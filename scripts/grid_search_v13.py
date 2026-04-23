"""Targeted grid search: fix trial-13 hyperparams, vary layers and heads.

Trial 13 best params:
  lr=0.00053, weight_decay=0.00946, dropout=0.242, label_smoothing=0.000

Grid: layers ∈ {4, 6, 8} × heads ∈ {8, 16}  →  6 combinations

Usage:
    python scripts/grid_search_v13.py
"""

import sys
import yaml
import torch
import torch.nn.functional as F
import wandb
from pathlib import Path
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.model import FOLModelV3
from src.model.encoder import FrozenT5Encoder
from src.data.dataset import FOLDatasetV3
from src.data.collator import FOLCollatorV2
from src.utils.attention import build_premises_cross_attn_mask

# Fixed from trial 13
FIXED = {
    "lr":              0.00053,
    "weight_decay":    0.00946,
    "dropout":         0.242,
    "label_smoothing": 0.000,
}

GRID = [
    {"layers": layers, "heads": heads}
    for layers in [4, 6, 8]
    for heads in [8, 16]
]

SWEEP_STEPS = 5000
WARMUP_STEPS = 835   # 16.7% of 5k


@torch.no_grad()
def evaluate(model, val_loader, encoder, device, use_bf16, label_smoothing, extra_id_0_id):
    model.eval()
    total_loss, total_tokens = 0.0, 0
    for batch in val_loader:
        input_ids    = batch["input_ids"].to(device)
        attn_mask    = batch["attention_mask"].to(device)
        proof_input  = batch["proof_decoder_input_ids"].to(device)
        proof_labels = batch["proof_labels"].to(device)
        premises_mask = build_premises_cross_attn_mask(input_ids, extra_id_0_id)
        dtype = torch.bfloat16 if use_bf16 else torch.float32
        with torch.amp.autocast("cuda", dtype=dtype, enabled=use_bf16):
            encoder_out = encoder(input_ids, attn_mask)
            proof_logits = model.forward_proof(encoder_out, proof_input,
                                               premises_cross_attn_mask=premises_mask)
        loss = F.cross_entropy(
            proof_logits.float().view(-1, proof_logits.size(-1)),
            proof_labels.view(-1),
            ignore_index=-100,
            label_smoothing=label_smoothing,
            reduction="sum",
        )
        total_loss   += loss.item()
        total_tokens += (proof_labels != -100).sum().item()
    model.train()
    return total_loss / max(total_tokens, 1)


def run_combo(cfg, model_cfg, base_cfg, encoder, train_loader, val_loader,
              device, extra_id_0_id, combo_name):
    use_bf16 = base_cfg["training"].get("bf16", True)

    try:
        run = wandb.init(project="fol-slm", name=f"v13-grid-{combo_name}",
                         config=cfg, reinit="finish_previous")
        use_wandb = True
    except Exception as e:
        print(f"  [wandb] {e}")
        run = None
        use_wandb = False

    model = FOLModelV3(model_cfg, vocab_size=_tokenizer.vocab_size).to(device)
    trans_ckpt = base_cfg["training"]["translation_checkpoint"]
    ckpt = torch.load(trans_ckpt, map_location=device, weights_only=False)
    state = {k: v for k, v in ckpt["model_state"].items()
             if not k.startswith("proof_decoder") and not k.startswith("answer_cls_head")}
    model.load_state_dict(state, strict=False)
    for p in model.encoder.parameters():      p.requires_grad = False
    for p in model.translation_decoder.parameters(): p.requires_grad = False

    optimizer = torch.optim.AdamW(model.proof_decoder.parameters(),
                                  lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=WARMUP_STEPS, num_training_steps=SWEEP_STEPS)

    model.train()
    data_iter = iter(train_loader)
    best_val_loss = float("inf")

    for step in range(1, SWEEP_STEPS + 1):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            batch = next(data_iter)

        input_ids    = batch["input_ids"].to(device)
        attn_mask    = batch["attention_mask"].to(device)
        proof_input  = batch["proof_decoder_input_ids"].to(device)
        proof_labels = batch["proof_labels"].to(device)
        premises_mask = build_premises_cross_attn_mask(input_ids, extra_id_0_id)

        dtype = torch.bfloat16 if use_bf16 else torch.float32
        with torch.amp.autocast("cuda", dtype=dtype, enabled=use_bf16):
            with torch.no_grad():
                encoder_out = encoder(input_ids, attn_mask)
            proof_logits = model.forward_proof(encoder_out, proof_input,
                                               premises_cross_attn_mask=premises_mask)

        loss = F.cross_entropy(
            proof_logits.float().view(-1, proof_logits.size(-1)),
            proof_labels.view(-1),
            ignore_index=-100,
            label_smoothing=cfg["label_smoothing"],
        )

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.proof_decoder.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        if step % 50 == 0 and use_wandb:
            try: wandb.log({"train_loss": loss.item(), "lr": scheduler.get_last_lr()[0]}, step=step)
            except Exception: pass

        if step % 1000 == 0:
            val_loss = evaluate(model, val_loader, encoder, device, use_bf16,
                                cfg["label_smoothing"], extra_id_0_id)
            print(f"    [{combo_name}] step {step}: val_loss={val_loss:.4f}")
            if use_wandb:
                try: wandb.log({"val_loss": val_loss}, step=step)
                except Exception: pass
            if val_loss < best_val_loss:
                best_val_loss = val_loss

    if use_wandb and run:
        try: run.finish()
        except Exception: pass

    return best_val_loss


_tokenizer = None


def main():
    global _tokenizer

    with open("configs/v13.yaml") as f:
        base_cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    _tokenizer = AutoTokenizer.from_pretrained(base_cfg["model"]["encoder_name"])
    extra_id_0_id = _tokenizer.convert_tokens_to_ids("<extra_id_0>")

    data_cfg = base_cfg["data"]
    batch_size = base_cfg["training"]["batch_size"]
    collator = FOLCollatorV2(_tokenizer.pad_token_id)

    train_ds = FOLDatasetV3(data_cfg["train_path"], _tokenizer,
                            data_cfg["max_input_len"], data_cfg["max_target_len"], max_qdep=2)
    val_ds   = FOLDatasetV3(data_cfg["val_path"],   _tokenizer,
                            data_cfg["max_input_len"], data_cfg["max_target_len"], max_qdep=2)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              collate_fn=collator, num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              collate_fn=collator, num_workers=4, pin_memory=True)

    encoder = FrozenT5Encoder(base_cfg["model"]["encoder_name"]).to(device)

    results = []
    for g in GRID:
        combo_name = f"{g['layers']}L-{g['heads']}H"
        cfg = {**FIXED, **g}
        print(f"\n{'='*60}")
        print(f"Running: {combo_name}  (layers={g['layers']}, heads={g['heads']})")
        print(f"  lr={FIXED['lr']}, wd={FIXED['weight_decay']}, dropout={FIXED['dropout']}, ls={FIXED['label_smoothing']}")
        print(f"{'='*60}")

        model_cfg = dict(base_cfg["model"])
        model_cfg["proof_decoder"] = dict(base_cfg["model"]["proof_decoder"])
        model_cfg["proof_decoder"]["layers"]  = g["layers"]
        model_cfg["proof_decoder"]["heads"]   = g["heads"]
        model_cfg["proof_decoder"]["dropout"] = FIXED["dropout"]
        # ff_dim scales with dim to keep ratio consistent
        model_cfg["proof_decoder"]["ff_dim"]  = base_cfg["model"]["proof_decoder"]["ff_dim"]

        val_loss = run_combo(cfg, model_cfg, base_cfg, encoder, train_loader, val_loader,
                             device, extra_id_0_id, combo_name)
        results.append((combo_name, g["layers"], g["heads"], val_loss))
        print(f"  >>> {combo_name}: val_loss={val_loss:.4f}")

    print(f"\n{'='*60}")
    print("GRID SEARCH RESULTS (ranked by val_loss)")
    print(f"{'='*60}")
    results.sort(key=lambda x: x[3])
    print(f"{'combo':>12}  {'layers':>6}  {'heads':>5}  {'val_loss':>9}")
    print("-" * 40)
    for name, layers, heads, vl in results:
        print(f"{name:>12}  {layers:>6}  {heads:>5}  {vl:>9.4f}")


if __name__ == "__main__":
    main()
