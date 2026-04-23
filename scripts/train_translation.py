"""Phase 1: Train translation decoder (NL → FOL premises + FOL question).

Only translation_decoder weights are updated. Proof decoder and cls head are unused.
Translation decoder input is truncated at <extra_id_3> — it never sees proof tokens.

Usage:
    python scripts/train_translation.py --config configs/v12_translation.yaml
"""

import sys
import os
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from dotenv import load_dotenv
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, WeightedRandomSampler

load_dotenv()
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.model import FOLModelV2
from src.data.dataset import FOLDatasetV2
from src.data.collator import FOLCollatorV2


def evaluate(model, val_loader, device, use_bf16):
    model.eval()
    total_loss, total_tokens = 0.0, 0
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
    with torch.no_grad():
        for batch in val_loader:
            input_ids      = batch["input_ids"].to(device)
            attn_mask      = batch["attention_mask"].to(device)
            trans_input    = batch["trans_decoder_input_ids"].to(device)
            trans_labels   = batch["trans_labels"].to(device)

            dtype = torch.bfloat16 if use_bf16 else torch.float32
            with autocast("cuda", dtype=dtype, enabled=use_bf16):
                trans_logits, _, _ = model.forward_translation(input_ids, attn_mask, trans_input)

            B, T, V = trans_logits.shape
            loss = loss_fn(trans_logits.float().view(B * T, V), trans_labels.view(B * T))
            n_tok = (trans_labels != -100).sum().item()
            total_loss   += loss.item() * n_tok
            total_tokens += n_tok
    model.train()
    return total_loss / total_tokens if total_tokens > 0 else float("inf")


def main(config_path: str = "configs/v12_translation.yaml", resume_from: str = None):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_bf16 = cfg["training"].get("bf16", False)
    tcfg     = cfg["training"]
    data_cfg = cfg["data"]

    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["encoder_name"])
    collator  = FOLCollatorV2(tokenizer.pad_token_id)

    def make_loader(max_qdep=None):
        ds = FOLDatasetV2(data_cfg["train_path"], tokenizer,
                          data_cfg["max_input_len"], data_cfg["max_target_len"],
                          max_qdep=max_qdep)
        weights = ds.get_sample_weights()
        sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
        return DataLoader(ds, batch_size=tcfg["batch_size"], sampler=sampler,
                          collate_fn=collator, num_workers=4, pin_memory=True)

    val_ds = FOLDatasetV2(data_cfg["val_path"], tokenizer,
                          data_cfg["max_input_len"], data_cfg["max_target_len"])
    val_loader = DataLoader(val_ds, batch_size=tcfg["batch_size"], shuffle=False,
                            collate_fn=collator, num_workers=4, pin_memory=True)

    model = FOLModelV2(cfg["model"], vocab_size=tokenizer.vocab_size).to(device)

    # Freeze everything except translation_decoder
    for p in model.parameters():
        p.requires_grad = False
    for p in model.translation_decoder.parameters():
        p.requires_grad = True

    # Resume from checkpoint if provided
    resume_step = 0
    if resume_from:
        ckpt = torch.load(resume_from, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"], strict=False)
        resume_step = ckpt.get("step", 0)
        print(f"Resumed from: {resume_from}  (step {resume_step})")

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"Device: {device}")
    print(f"Trainable: {trainable:,} / {total:,} (translation decoder only)")

    optimizer = torch.optim.AdamW(
        model.translation_decoder.parameters(),
        lr=tcfg["learning_rate"], weight_decay=tcfg.get("weight_decay", 0.01)
    )
    grad_accum = tcfg["grad_accum_steps"]
    # When resuming, treat continuation as a fresh schedule over remaining steps only
    remaining_steps = tcfg["max_steps"] - resume_step
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=tcfg["warmup_steps"] // grad_accum,
        num_training_steps=max(remaining_steps, tcfg["max_steps"]) // grad_accum,
    )
    scaler   = GradScaler("cuda", enabled=False)
    loss_fn  = nn.CrossEntropyLoss(ignore_index=-100, label_smoothing=tcfg.get("label_smoothing", 0.0))

    # W&B
    log_cfg  = cfg.get("logging", {})
    use_wandb = log_cfg.get("use_wandb", False)
    wandb = None
    if use_wandb:
        import wandb as _wandb
        wandb = _wandb
        api_key = os.environ.get("WANDB_API_KEY") or os.environ.get("wandb_api_key")
        if api_key:
            wandb.login(key=api_key, relogin=True)
        wandb.init(project=log_cfg.get("project_name", "fol-slm"),
                   name=log_cfg.get("run_name"), config=tcfg)

    os.makedirs(tcfg["output_dir"], exist_ok=True)
    model.train()
    optimizer.zero_grad()

    # Curriculum — skip phases already passed when resuming
    train_loader = make_loader(max_qdep=2)
    curriculum = [(5000, make_loader(max_qdep=None))]
    # If resuming past the curriculum switch point, start with the full-data loader
    if resume_step >= 5000:
        train_loader = curriculum.pop(0)[1]
        print(f"Resuming past curriculum switch — using full data loader")
    data_iter = iter(train_loader)

    for step in range(resume_step, tcfg["max_steps"]):
        if curriculum and step >= curriculum[0][0]:
            _, train_loader = curriculum.pop(0)
            data_iter = iter(train_loader)
            print(f"[step {step}] Curriculum: switching to full data")

        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            batch = next(data_iter)

        input_ids    = batch["input_ids"].to(device)
        attn_mask    = batch["attention_mask"].to(device)
        trans_input  = batch["trans_decoder_input_ids"].to(device)
        trans_labels = batch["trans_labels"].to(device)

        dtype = torch.bfloat16 if use_bf16 else torch.float32
        with autocast("cuda", dtype=dtype, enabled=use_bf16):
            trans_logits, _, _ = model.forward_translation(input_ids, attn_mask, trans_input)
            B, T, V = trans_logits.shape
            loss = loss_fn(trans_logits.float().view(B * T, V), trans_labels.view(B * T)) / grad_accum

        if not torch.isfinite(loss):
            optimizer.zero_grad()
            continue

        loss.backward()

        if (step + 1) % grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg["clip_grad_norm"])
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        if step % log_cfg.get("log_every", 50) == 0:
            lr = scheduler.get_last_lr()[0]
            print(f"[step {step:5d}] loss: {loss.item()*grad_accum:.4f}  lr: {lr:.2e}")
            if wandb:
                wandb.log({"train/loss": loss.item() * grad_accum, "train/lr": lr}, step=step)

        if (step + 1) % tcfg["eval_every"] == 0:
            val_loss = evaluate(model, val_loader, device, use_bf16)
            print(f"[step {step+1:5d}] val_loss: {val_loss:.4f}")
            if wandb:
                wandb.log({"val/loss": val_loss}, step=step)

        if (step + 1) % tcfg["save_every"] == 0:
            path = os.path.join(tcfg["output_dir"], f"checkpoint_{step+1}.pt")
            torch.save({"model_state": model.state_dict(), "step": step + 1}, path)
            print(f"  Saved {path}")

    val_loss = evaluate(model, val_loader, device, use_bf16)
    path = os.path.join(tcfg["output_dir"], "checkpoint_final.pt")
    torch.save({"model_state": model.state_dict(), "step": tcfg["max_steps"]}, path)
    print(f"\nDone. Final val_loss: {val_loss:.4f}  Saved → {path}")
    if wandb:
        wandb.finish()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/v12_translation.yaml")
    parser.add_argument("--resume-from", default=None, help="Checkpoint path to resume from")
    args = parser.parse_args()
    main(args.config, args.resume_from)
