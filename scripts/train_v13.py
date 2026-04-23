"""V13 Phase 2: Train proof decoder cross-attending T5 encoder (premises only).

Translation decoder is frozen (loaded from v12 translation checkpoint).
Proof decoder:
  - Input: [<extra_id_2>, FOL_question, <extra_id_3>, gold_proof_{0..t-1}]
  - Self-attention: naturally sees FOL question + prior proof steps (no mask needed)
  - Cross-attention: T5 encoder restricted to NL premises (before <extra_id_0>)

Gradual curriculum: QDep<=1 -> QDep<=2 -> QDep<=3 -> full data.

Usage:
    python scripts/train_v13.py --config configs/v13.yaml
"""

import sys
import os
import yaml
import torch
import torch.nn as nn
from pathlib import Path
from dotenv import load_dotenv
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from torch.amp import autocast
from torch.utils.data import DataLoader, WeightedRandomSampler

load_dotenv()
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.model import FOLModelV3
from src.data.dataset import FOLDatasetV3
from src.data.collator import FOLCollatorV2
from src.utils.attention import build_premises_cross_attn_mask


def evaluate(model, val_loader, device, use_bf16, extra_id_0_id):
    model.eval()
    total_loss, total_tokens = 0.0, 0
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
    with torch.no_grad():
        for batch in val_loader:
            input_ids      = batch["input_ids"].to(device)
            attn_mask      = batch["attention_mask"].to(device)
            trans_input    = batch["trans_decoder_input_ids"].to(device)
            proof_input    = batch["proof_decoder_input_ids"].to(device)
            proof_labels   = batch["proof_labels"].to(device)

            premises_mask = build_premises_cross_attn_mask(input_ids, extra_id_0_id)

            dtype = torch.bfloat16 if use_bf16 else torch.float32
            with autocast("cuda", dtype=dtype, enabled=use_bf16):
                with torch.no_grad():
                    _, _, encoder_out = model.forward_translation(input_ids, attn_mask, trans_input)
                proof_logits = model.forward_proof(encoder_out, proof_input,
                                                   premises_cross_attn_mask=premises_mask)

            B, T, V = proof_logits.shape
            loss = loss_fn(proof_logits.float().view(B * T, V), proof_labels.view(B * T))
            n_tok = (proof_labels != -100).sum().item()
            total_loss   += loss.item() * n_tok
            total_tokens += n_tok
    model.train()
    return total_loss / total_tokens if total_tokens > 0 else float("inf")


def main(config_path: str = "configs/v13.yaml"):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_bf16 = cfg["training"].get("bf16", False)
    tcfg     = cfg["training"]
    data_cfg = cfg["data"]

    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["encoder_name"])
    extra_id_0_id = tokenizer.convert_tokens_to_ids("<extra_id_0>")
    collator = FOLCollatorV2(tokenizer.pad_token_id)

    curriculum_cfg = cfg.get("curriculum", {})
    phases = curriculum_cfg.get("phases", [])

    def make_loader(max_qdep=None):
        ds = FOLDatasetV3(data_cfg["train_path"], tokenizer,
                          data_cfg["max_input_len"], data_cfg["max_target_len"],
                          max_qdep=max_qdep)
        weights = ds.get_sample_weights()
        sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
        return DataLoader(ds, batch_size=tcfg["batch_size"], sampler=sampler,
                          collate_fn=collator, num_workers=4, pin_memory=True)

    val_ds = FOLDatasetV3(data_cfg["val_path"], tokenizer,
                          data_cfg["max_input_len"], data_cfg["max_target_len"])
    val_loader = DataLoader(val_ds, batch_size=tcfg["batch_size"], shuffle=False,
                            collate_fn=collator, num_workers=4, pin_memory=True)

    # Build model and load translation checkpoint
    model = FOLModelV3(cfg["model"], vocab_size=tokenizer.vocab_size).to(device)
    trans_ckpt_path = tcfg.get("translation_checkpoint")
    if trans_ckpt_path:
        ckpt = torch.load(trans_ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"], strict=False)
        print(f"Loaded translation decoder from: {trans_ckpt_path}")

    # Freeze encoder + translation decoder; train only proof decoder
    for p in model.parameters():
        p.requires_grad = False
    for p in model.proof_decoder.parameters():
        p.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"Device: {device}")
    print(f"Trainable: {trainable:,} / {total:,} (proof decoder only)")

    optimizer = torch.optim.AdamW(
        model.proof_decoder.parameters(),
        lr=tcfg["learning_rate"], weight_decay=tcfg.get("weight_decay", 0.01)
    )
    grad_accum = tcfg["grad_accum_steps"]
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=tcfg["warmup_steps"] // grad_accum,
        num_training_steps=tcfg["max_steps"] // grad_accum,
    )
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100, label_smoothing=tcfg.get("label_smoothing", 0.0))

    # W&B
    log_cfg   = cfg.get("logging", {})
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

    # No curriculum — full data from step 0
    train_loader = make_loader(max_qdep=None)
    data_iter = iter(train_loader)

    for step in range(tcfg["max_steps"]):

        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            batch = next(data_iter)

        input_ids      = batch["input_ids"].to(device)
        attn_mask      = batch["attention_mask"].to(device)
        trans_input    = batch["trans_decoder_input_ids"].to(device)
        proof_input    = batch["proof_decoder_input_ids"].to(device)
        proof_labels   = batch["proof_labels"].to(device)

        premises_mask = build_premises_cross_attn_mask(input_ids, extra_id_0_id)

        dtype = torch.bfloat16 if use_bf16 else torch.float32
        with autocast("cuda", dtype=dtype, enabled=use_bf16):
            with torch.no_grad():
                _, _, encoder_out = model.forward_translation(input_ids, attn_mask, trans_input)
            proof_logits = model.forward_proof(encoder_out, proof_input,
                                               premises_cross_attn_mask=premises_mask)
            B, T, V = proof_logits.shape
            loss = loss_fn(proof_logits.float().view(B * T, V), proof_labels.view(B * T)) / grad_accum

        if not torch.isfinite(loss):
            optimizer.zero_grad()
            continue

        loss.backward()

        if (step + 1) % grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(model.proof_decoder.parameters(), tcfg["clip_grad_norm"])
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        if step % log_cfg.get("log_every", 50) == 0:
            lr = scheduler.get_last_lr()[0]
            print(f"[step {step:5d}] loss: {loss.item()*grad_accum:.4f}  lr: {lr:.2e}")
            if wandb:
                wandb.log({"train/loss": loss.item() * grad_accum, "train/lr": lr}, step=step)

        if (step + 1) % tcfg["eval_every"] == 0:
            val_loss = evaluate(model, val_loader, device, use_bf16, extra_id_0_id)
            print(f"[step {step+1:5d}] val_loss: {val_loss:.4f}")
            if wandb:
                wandb.log({"val/loss": val_loss}, step=step)

        if (step + 1) % tcfg["save_every"] == 0:
            path = os.path.join(tcfg["output_dir"], f"checkpoint_{step+1}.pt")
            torch.save({"model_state": model.state_dict(), "step": step + 1}, path)
            print(f"  Saved {path}")

    val_loss = evaluate(model, val_loader, device, use_bf16, extra_id_0_id)
    path = os.path.join(tcfg["output_dir"], "checkpoint_final.pt")
    torch.save({"model_state": model.state_dict(), "step": tcfg["max_steps"]}, path)
    print(f"\nDone. Final val_loss: {val_loss:.4f}  Saved → {path}")
    if wandb:
        wandb.finish()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/v13.yaml")
    args = parser.parse_args()
    main(args.config)
