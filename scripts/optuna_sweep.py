"""Hyperparameter sweep for the translation decoder using Optuna.

Searches lr, weight_decay, label_smoothing, dropout, warmup_fraction,
decoder layers, and heads. Architecture dim is fixed at 512.
Each trial runs 3,000 steps (15% of full 20k training) — enough signal
to distinguish good configs without full training cost.

Known-good baseline (step 20k):
    lr=1.9028e-3, weight_decay=3e-3, label_smoothing=0.05,
    dropout=0.1, warmup_fraction=0.167, layers=4, heads=8

Usage:
    python scripts/optuna_sweep.py --config configs/v12_translation.yaml --n-trials 20
"""

import sys
import os
import copy
import yaml
import torch
import torch.nn as nn
import optuna
from dotenv import load_dotenv
from optuna.pruners import MedianPruner
from pathlib import Path
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, WeightedRandomSampler

load_dotenv()
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.model import FOLModelV2
from src.data.dataset import FOLDatasetV2
from src.data.collator import FOLCollatorV2

SWEEP_STEPS = 3000
EVAL_EVERY  = 300
GRAD_ACCUM  = 4

_base_cfg   = None
_tokenizer  = None
_val_loader = None
_device     = None


def evaluate(model, val_loader, device, use_bf16):
    model.eval()
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
    total_loss, total_tokens = 0.0, 0
    with torch.no_grad():
        for batch in val_loader:
            input_ids    = batch["input_ids"].to(device)
            attn_mask    = batch["attention_mask"].to(device)
            trans_input  = batch["trans_decoder_input_ids"].to(device)
            trans_labels = batch["trans_labels"].to(device)
            dtype = torch.bfloat16 if use_bf16 else torch.float32
            with autocast("cuda", dtype=dtype, enabled=use_bf16):
                trans_logits, _, _ = model.forward_translation(input_ids, attn_mask, trans_input)
            B, T, V = trans_logits.shape
            loss    = loss_fn(trans_logits.float().view(B * T, V), trans_labels.view(B * T))
            n_tok   = (trans_labels != -100).sum().item()
            total_loss   += loss.item() * n_tok
            total_tokens += n_tok
    model.train()
    return total_loss / total_tokens if total_tokens > 0 else float("inf")


def make_train_loader(cfg):
    data_cfg = cfg["data"]
    tcfg     = cfg["training"]
    ds = FOLDatasetV2(
        data_cfg["train_path"], _tokenizer,
        data_cfg["max_input_len"], data_cfg["max_target_len"],
    )
    weights = ds.get_sample_weights()
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
    return DataLoader(
        ds, batch_size=tcfg["batch_size"], sampler=sampler,
        collate_fn=FOLCollatorV2(_tokenizer.pad_token_id),
        num_workers=4, pin_memory=True,
    )


def objective(trial: optuna.Trial) -> float:
    cfg = copy.deepcopy(_base_cfg)

    # ── Hyperparameter search space ───────────────────────────────────────────
    lr               = trial.suggest_float("lr",               5e-4,  4e-3,  log=True)
    weight_decay     = trial.suggest_float("weight_decay",     5e-4,  1e-2,  log=True)
    label_smoothing  = trial.suggest_float("label_smoothing",  0.0,   0.15)
    dropout          = trial.suggest_float("dropout",          0.05,  0.3)
    warmup_fraction  = trial.suggest_float("warmup_fraction",  0.10,  0.25)
    n_layers         = trial.suggest_int("layers",             2,     6)
    n_heads          = trial.suggest_categorical("heads",      [4, 8])

    warmup_steps = int(warmup_fraction * SWEEP_STEPS)

    # Apply to config
    cfg["model"]["translation_decoder"]["layers"]  = n_layers
    cfg["model"]["translation_decoder"]["heads"]   = n_heads
    cfg["model"]["translation_decoder"]["dropout"] = dropout
    cfg["training"]["learning_rate"]   = lr
    cfg["training"]["weight_decay"]    = weight_decay
    cfg["training"]["label_smoothing"] = label_smoothing
    cfg["training"]["batch_size"]      = _base_cfg["training"]["batch_size"]

    use_bf16 = cfg["training"].get("bf16", True)

    model = FOLModelV2(cfg["model"], vocab_size=_tokenizer.vocab_size).to(_device)
    for p in model.parameters():
        p.requires_grad = False
    for p in model.translation_decoder.parameters():
        p.requires_grad = True

    optimizer = torch.optim.AdamW(
        model.translation_decoder.parameters(),
        lr=lr, weight_decay=weight_decay,
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps // GRAD_ACCUM,
        num_training_steps=SWEEP_STEPS // GRAD_ACCUM,
    )
    loss_fn   = nn.CrossEntropyLoss(ignore_index=-100, label_smoothing=label_smoothing)
    scaler    = GradScaler("cuda", enabled=False)

    train_loader = make_train_loader(cfg)
    data_iter    = iter(train_loader)

    model.train()
    optimizer.zero_grad()

    for step in range(SWEEP_STEPS):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            batch = next(data_iter)

        input_ids    = batch["input_ids"].to(_device)
        attn_mask    = batch["attention_mask"].to(_device)
        trans_input  = batch["trans_decoder_input_ids"].to(_device)
        trans_labels = batch["trans_labels"].to(_device)

        dtype = torch.bfloat16 if use_bf16 else torch.float32
        with autocast("cuda", dtype=dtype, enabled=use_bf16):
            trans_logits, _, _ = model.forward_translation(input_ids, attn_mask, trans_input)
            B, T, V = trans_logits.shape
            loss = loss_fn(trans_logits.float().view(B * T, V), trans_labels.view(B * T)) / GRAD_ACCUM

        if not torch.isfinite(loss):
            optimizer.zero_grad()
            continue

        loss.backward()

        if (step + 1) % GRAD_ACCUM == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        if (step + 1) % EVAL_EVERY == 0:
            val_loss = evaluate(model, _val_loader, _device, use_bf16)
            print(f"  [trial {trial.number}  step {step+1}] val_loss={val_loss:.4f}")
            trial.report(val_loss, step + 1)
            if trial.should_prune():
                raise optuna.TrialPruned()

    val_loss = evaluate(model, _val_loader, _device, use_bf16)
    trial.set_user_attr("final_val_loss", val_loss)
    return val_loss


def main(config_path: str = "configs/v12_translation.yaml", n_trials: int = 20):
    global _base_cfg, _tokenizer, _val_loader, _device

    with open(config_path) as f:
        _base_cfg = yaml.safe_load(f)

    _device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _tokenizer = AutoTokenizer.from_pretrained(_base_cfg["model"]["encoder_name"])
    print(f"Device: {_device}")

    data_cfg = _base_cfg["data"]
    val_ds   = FOLDatasetV2(
        data_cfg["val_path"], _tokenizer,
        data_cfg["max_input_len"], data_cfg["max_target_len"],
    )
    _val_loader = DataLoader(
        val_ds, batch_size=_base_cfg["training"]["batch_size"], shuffle=False,
        collate_fn=FOLCollatorV2(_tokenizer.pad_token_id),
        num_workers=4, pin_memory=True,
    )

    os.makedirs("outputs", exist_ok=True)
    study = optuna.create_study(
        direction="minimize",
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=EVAL_EVERY * 3),
        study_name="fol-slm-translation-hpo",
        storage="sqlite:///outputs/optuna_study.db",
        load_if_exists=True,
    )

    print(f"\nStarting sweep: {n_trials} trials × {SWEEP_STEPS} steps each")
    study.optimize(objective, n_trials=n_trials)

    print(f"\n{'='*60}")
    print("SWEEP COMPLETE")
    print(f"{'='*60}")
    print(f"Best trial : #{study.best_trial.number}")
    print(f"Best val_loss: {study.best_value:.4f}")
    print(f"Best params:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")

    trials = sorted(
        [t for t in study.trials if t.value is not None],
        key=lambda t: t.value,
    )
    print(f"\nTop 5 trials:")
    for t in trials[:5]:
        print(f"  #{t.number}: val_loss={t.value:.4f}  {t.params}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",   default="configs/v12_translation.yaml")
    parser.add_argument("--n-trials", type=int, default=20)
    args = parser.parse_args()
    main(args.config, args.n_trials)
