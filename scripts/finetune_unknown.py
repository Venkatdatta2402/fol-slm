"""Fine-tune last N decoder layers on Unknown-heavy data with gradual unfreezing.

Strategy (ULMFiT-style):
  Phase 1 (steps 0 → unfreeze_at):     only last layer trainable, lr = base_lr
  Phase 2 (steps unfreeze_at → total):  unfreeze second-to-last layer with lr = base_lr / 3

Data mix: all Unknown + 10% True + 10% False (prevents catastrophic forgetting).

Usage:
    python scripts/finetune_unknown.py \
        --checkpoint outputs/final_v9/checkpoint_final.pt \
        --config configs/final.yaml \
        --base-lr 1e-4 \
        --max-steps 5000 \
        --unfreeze-at 2500 \
        --out-dir outputs/finetune_unknown/
"""

import sys
import argparse
import random
import os
import yaml
import torch
import torch.nn.functional as F
from pathlib import Path
from torch.amp import GradScaler
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.model import FOLModel
from src.data.dataset import ReasoningDataset
from src.data.collator import FOLCollator


# ── Freezing helpers ──────────────────────────────────────────────────────────

def freeze_all_decoder(model):
    for p in model.parameters():
        p.requires_grad = False


def unfreeze_layer(model, layer_idx: int):
    """Unfreeze decoder layer at layer_idx (0-indexed from start)."""
    for p in model.decoder.layers[layer_idx].parameters():
        p.requires_grad = True


def unfreeze_head(model):
    """Always unfreeze norm + lm_head."""
    for p in model.decoder.norm.parameters():
        p.requires_grad = True
    for p in model.decoder.lm_head.parameters():
        p.requires_grad = True


def count_trainable(model):
    t = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return t, total


# ── Evaluation ────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, val_loader, device, use_bf16):
    model.eval()
    total_loss, total_tokens = 0.0, 0
    for batch in val_loader:
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        decoder_input  = batch["decoder_input_ids"].to(device)
        labels         = batch["labels"].to(device)
        dtype = torch.bfloat16 if use_bf16 else torch.float16
        with torch.amp.autocast("cuda", dtype=dtype, enabled=use_bf16):
            logits = model(input_ids, attention_mask, decoder_input)
        B, T, V = logits.shape
        loss = F.cross_entropy(
            logits.float().view(B * T, V),
            labels.view(B * T),
            ignore_index=-100,
            reduction="sum",
        )
        n_tokens = (labels != -100).sum().item()
        total_loss   += loss.item()
        total_tokens += n_tokens
    model.train()
    return total_loss / total_tokens if total_tokens > 0 else float("inf")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint",  default="outputs/final_v8/checkpoint_final.pt")
    parser.add_argument("--config",      default="configs/final.yaml")
    parser.add_argument("--base-lr",     type=float, default=1e-4)
    parser.add_argument("--max-steps",   type=int,   default=5000)
    parser.add_argument("--unfreeze-at", type=int,   default=2500,
                        help="Step at which to unfreeze second-to-last decoder layer")
    parser.add_argument("--out-dir",     default="outputs/finetune_unknown/")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_bf16 = cfg["training"].get("bf16", False)
    grad_accum = cfg["training"]["grad_accum_steps"]
    batch_size = cfg["training"]["batch_size"]
    data_cfg   = cfg["data"]

    print(f"Device:     {device}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"base_lr={args.base_lr}  max_steps={args.max_steps}  unfreeze_at={args.unfreeze_at}")

    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["encoder_name"])
    collator  = FOLCollator(tokenizer.pad_token_id)

    # ── Dataset: confusion-pair balanced ─────────────────────────────────────
    # Negated:     neg_Unknown ≈ neg_False  (model confuses these)
    # Non-negated: pos_Unknown ≈ pos_True   (model confuses these)
    # Rare:        pos_False + neg_True     (included in full, counter-intuitive cases)
    full_ds = ReasoningDataset(
        data_cfg["train_path"], tokenizer,
        data_cfg["max_input_len"], data_cfg["max_target_len"],
    )

    def is_neg(s):
        q = s["premises"].split("<extra_id_0>", 1)[1].strip().lower() \
            if "<extra_id_0>" in s["premises"] else ""
        return any(w in q for w in ["not", "n't", "never", "no "])

    random.seed(42)
    pos_unknown = [s for s in full_ds.samples if s["answer"] == "Unknown" and not is_neg(s)]
    neg_unknown = [s for s in full_ds.samples if s["answer"] == "Unknown" and     is_neg(s)]
    pos_true    = [s for s in full_ds.samples if s["answer"] == "True"    and not is_neg(s)]
    neg_false   = [s for s in full_ds.samples if s["answer"] == "False"   and     is_neg(s)]
    pos_false   = [s for s in full_ds.samples if s["answer"] == "False"   and not is_neg(s)]
    neg_true    = [s for s in full_ds.samples if s["answer"] == "True"    and     is_neg(s)]

    K = 10_000  # cap per class for the confusion pairs
    mixed = (
        random.sample(pos_unknown, K) +
        random.sample(pos_true,    K) +
        random.sample(neg_unknown, K) +
        random.sample(neg_false,   K) +
        pos_false +
        neg_true
    )
    random.shuffle(mixed)
    full_ds.samples = mixed
    print(f"Train mix (K={K} per confusion-pair class):")
    print(f"  pos_Unknown: {K:,}  pos_True: {K:,}")
    print(f"  neg_Unknown: {K:,}  neg_False: {K:,}")
    print(f"  pos_False:   {len(pos_false):,}  neg_True: {len(neg_true):,}")
    print(f"  Total: {len(mixed):,}")

    train_loader = DataLoader(full_ds, batch_size=batch_size, shuffle=True,
                              collate_fn=collator, num_workers=4, pin_memory=True)

    val_ds = ReasoningDataset(
        data_cfg["val_path"], tokenizer,
        data_cfg["max_input_len"], data_cfg["max_target_len"],
    )
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            collate_fn=collator, num_workers=4, pin_memory=True)

    # ── Model ─────────────────────────────────────────────────────────────────
    model = FOLModel(cfg["model"], vocab_size=tokenizer.vocab_size)
    ckpt  = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model = model.to(device)

    n_layers  = len(model.decoder.layers)
    last_idx  = n_layers - 1       # e.g. 3
    second_idx = n_layers - 2      # e.g. 2

    # ── Phase 1 setup: freeze all, unfreeze last layer + head ─────────────────
    freeze_all_decoder(model)
    unfreeze_layer(model, last_idx)
    unfreeze_head(model)
    t, total = count_trainable(model)
    print(f"\nPhase 1 — trainable: {t:,} / {total:,} ({t/total*100:.1f}%)")
    print(f"  Unfrozen: layer[{last_idx}] + norm + lm_head  lr={args.base_lr:.1e}")

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.base_lr,
        weight_decay=cfg["training"].get("weight_decay", 0.01),
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(50, args.max_steps // 20) // grad_accum,
        num_training_steps=args.max_steps // grad_accum,
    )
    scaler = GradScaler("cuda", enabled=use_bf16)

    # ── W&B ───────────────────────────────────────────────────────────────────
    log_cfg = cfg.get("logging", {})
    use_wandb = log_cfg.get("use_wandb", False)
    wandb = None
    if use_wandb:
        import wandb as _wandb
        wandb = _wandb
        wandb.init(project=log_cfg.get("project_name", "fol-slm"),
                   name="finetune-v8-confusion-balanced",
                   config=vars(args))

    # ── Training loop ─────────────────────────────────────────────────────────
    os.makedirs(args.out_dir, exist_ok=True)
    model.train()
    optimizer.zero_grad()
    phase2_added = False
    data_iter = iter(train_loader)

    for step in range(args.max_steps):

        # ── Phase 2: unfreeze second-to-last layer with lower LR ──────────────
        if step == args.unfreeze_at and not phase2_added:
            unfreeze_layer(model, second_idx)
            lower_lr = args.base_lr / 3
            optimizer.add_param_group({
                "params": [p for p in model.decoder.layers[second_idx].parameters()],
                "lr": lower_lr,
                "weight_decay": cfg["training"].get("weight_decay", 0.01),
            })
            phase2_added = True
            t, total = count_trainable(model)
            print(f"\nPhase 2 — step {step}: unfroze layer[{second_idx}] with lr={lower_lr:.1e}")
            print(f"  Trainable: {t:,} / {total:,} ({t/total*100:.1f}%)")

        # ── Batch ─────────────────────────────────────────────────────────────
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            batch = next(data_iter)

        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        decoder_input  = batch["decoder_input_ids"].to(device)
        labels         = batch["labels"].to(device)

        dtype = torch.bfloat16 if use_bf16 else torch.float16
        with torch.amp.autocast("cuda", dtype=dtype, enabled=use_bf16):
            logits = model(input_ids, attention_mask, decoder_input)
            B, T, V = logits.shape
            loss = F.cross_entropy(
                logits.float().view(B * T, V),
                labels.view(B * T),
                ignore_index=-100,
            ) / grad_accum

        if not torch.isfinite(loss):
            optimizer.zero_grad()
            continue

        scaler.scale(loss).backward()

        if (step + 1) % grad_accum == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["training"]["clip_grad_norm"])
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()

        # ── Logging ───────────────────────────────────────────────────────────
        if step % 50 == 0:
            lr0 = optimizer.param_groups[0]["lr"]
            print(f"[step {step:5d}] loss: {loss.item()*grad_accum:.4f}  lr: {lr0:.2e}")
            if wandb:
                wandb.log({"train/loss": loss.item() * grad_accum, "train/lr": lr0}, step=step)

        # ── Eval + checkpoint ──────────────────────────────────────────────────
        if (step + 1) % 500 == 0:
            val_loss = evaluate(model, val_loader, device, use_bf16)
            print(f"[step {step+1:5d}] VAL LOSS: {val_loss:.4f}")
            if wandb:
                wandb.log({"val/loss": val_loss}, step=step)

        if (step + 1) % 1000 == 0:
            path = os.path.join(args.out_dir, f"checkpoint_{step+1}.pt")
            torch.save({"model_state": model.state_dict(), "step": step + 1}, path)
            print(f"  Saved {path}")

    # Final checkpoint
    val_loss = evaluate(model, val_loader, device, use_bf16)
    path = os.path.join(args.out_dir, "checkpoint_final.pt")
    torch.save({"model_state": model.state_dict(), "step": args.max_steps}, path)
    print(f"\nDone. Final val_loss: {val_loss:.4f}  Saved → {path}")
    if wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
