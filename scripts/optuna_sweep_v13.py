"""V13 Optuna hyperparameter sweep for proof decoder.

Sweeps: lr, weight_decay, dropout, label_smoothing, decoder_layers, decoder_heads.
No curriculum — full data from step 0 (V13 doesn't need it with explicit <extra_id_5> separators).
Each trial runs 5k steps. Uses shared frozen encoder + frozen translation decoder.

Usage:
    python scripts/optuna_sweep_v13.py --n-trials 30
"""

import sys
import copy
import yaml
import torch
import optuna
import wandb
from optuna.pruners import MedianPruner
from pathlib import Path
from transformers import AutoTokenizer
from torch.utils.data import DataLoader

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.model import FOLModelV3
from src.model.encoder import FrozenT5Encoder
from src.data.dataset import FOLDatasetV3
from src.data.collator import FOLCollatorV2
from src.utils.attention import build_premises_cross_attn_mask

# Module-level references set in main()
_base_cfg = None
_tokenizer = None
_train_loader = None
_val_loader = None
_shared_encoder = None
_frozen_trans_decoder = None
_device = None
_extra_id_0_id = None

SWEEP_STEPS = 5000
WARMUP_STEPS = 835   # 16.7% of 5k


def run_trial(model, train_loader, val_loader, cfg, device, trial):
    """Train proof decoder for SWEEP_STEPS and return final val_loss."""
    import torch.nn.functional as F
    from transformers import get_linear_schedule_with_warmup

    try:
        run = wandb.init(
            project="fol-slm",
            name=f"v13-sweep-trial-{trial.number}",
            config=cfg,
            reinit="finish_previous",
        )
        use_wandb = True
    except Exception as e:
        print(f"  [wandb] init failed ({e}), continuing without logging")
        run = None
        use_wandb = False

    optimizer = torch.optim.AdamW(
        model.proof_decoder.parameters(),
        lr=cfg["lr"],
        weight_decay=cfg["weight_decay"],
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=WARMUP_STEPS,
        num_training_steps=SWEEP_STEPS,
    )

    use_bf16 = _base_cfg["training"].get("bf16", True)
    label_smoothing = cfg["label_smoothing"]
    clip_grad = _base_cfg["training"].get("clip_grad_norm", 1.0)

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

        premises_mask = build_premises_cross_attn_mask(input_ids, _extra_id_0_id)

        dtype = torch.bfloat16 if use_bf16 else torch.float32
        with torch.amp.autocast("cuda", dtype=dtype, enabled=use_bf16):
            with torch.no_grad():
                encoder_out = _shared_encoder(input_ids, attn_mask)

            proof_logits = model.forward_proof(
                encoder_out, proof_input,
                premises_cross_attn_mask=premises_mask,
            )

        loss = F.cross_entropy(
            proof_logits.float().view(-1, proof_logits.size(-1)),
            proof_labels.view(-1),
            ignore_index=-100,
            label_smoothing=label_smoothing,
        )

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.proof_decoder.parameters(), clip_grad)
        optimizer.step()
        scheduler.step()

        if step % 50 == 0 and use_wandb:
            try:
                wandb.log({"train_loss": loss.item(), "lr": scheduler.get_last_lr()[0]}, step=step)
            except Exception:
                pass

        # Eval + pruning check every 1k steps
        if step % 1000 == 0:
            val_loss = evaluate(model, val_loader, device, use_bf16, label_smoothing)
            if use_wandb:
                try:
                    wandb.log({"val_loss": val_loss}, step=step)
                except Exception:
                    pass
            model.train()
            if val_loss < best_val_loss:
                best_val_loss = val_loss
            trial.report(val_loss, step)
            if trial.should_prune():
                if use_wandb and run:
                    try: run.finish()
                    except Exception: pass
                return None  # pruned

    if use_wandb and run:
        try: run.finish()
        except Exception: pass
    return best_val_loss


@torch.no_grad()
def evaluate(model, val_loader, device, use_bf16, label_smoothing):
    import torch.nn.functional as F
    model.eval()
    total_loss, total_tokens = 0.0, 0
    for batch in val_loader:
        input_ids    = batch["input_ids"].to(device)
        attn_mask    = batch["attention_mask"].to(device)
        proof_input  = batch["proof_decoder_input_ids"].to(device)
        proof_labels = batch["proof_labels"].to(device)

        premises_mask = build_premises_cross_attn_mask(input_ids, _extra_id_0_id)

        dtype = torch.bfloat16 if use_bf16 else torch.float32
        with torch.amp.autocast("cuda", dtype=dtype, enabled=use_bf16):
            encoder_out = _shared_encoder(input_ids, attn_mask)
            proof_logits = model.forward_proof(
                encoder_out, proof_input,
                premises_cross_attn_mask=premises_mask,
            )

        mask = proof_labels != -100
        loss = F.cross_entropy(
            proof_logits.float().view(-1, proof_logits.size(-1)),
            proof_labels.view(-1),
            ignore_index=-100,
            label_smoothing=label_smoothing,
            reduction="sum",
        )
        total_loss += loss.item()
        total_tokens += mask.sum().item()

    return total_loss / max(total_tokens, 1)


def objective(trial: optuna.Trial) -> float:
    cfg = {
        "lr":              trial.suggest_float("lr", 5e-4, 3e-3, log=True),
        "weight_decay":    trial.suggest_float("weight_decay", 1e-3, 1e-2, log=True),
        "dropout":         trial.suggest_float("dropout", 0.1, 0.4),
        "label_smoothing": trial.suggest_float("label_smoothing", 0.0, 0.1),
        "layers":          trial.suggest_categorical("layers", [4, 6]),
        "heads":           trial.suggest_categorical("heads", [8, 16]),
    }

    # Build proof decoder config
    model_cfg = copy.deepcopy(_base_cfg["model"])
    model_cfg["proof_decoder"]["layers"]  = cfg["layers"]
    model_cfg["proof_decoder"]["heads"]   = cfg["heads"]
    model_cfg["proof_decoder"]["dropout"] = cfg["dropout"]

    model = FOLModelV3(model_cfg, vocab_size=_tokenizer.vocab_size).to(_device)

    # Load frozen translation decoder weights
    trans_ckpt = _base_cfg["training"]["translation_checkpoint"]
    ckpt = torch.load(trans_ckpt, map_location=_device, weights_only=False)
    # Load only translation decoder + encoder weights, leave proof decoder fresh
    state = {k: v for k, v in ckpt["model_state"].items()
             if not k.startswith("proof_decoder") and not k.startswith("answer_cls_head")}
    model.load_state_dict(state, strict=False)

    # Freeze encoder + translation decoder
    for p in model.encoder.parameters():
        p.requires_grad = False
    for p in model.translation_decoder.parameters():
        p.requires_grad = False

    result = run_trial(model, _train_loader, _val_loader, cfg, _device, trial)

    if result is None:
        raise optuna.TrialPruned()

    trial.set_user_attr("val_loss", result)
    print(f"  Trial #{trial.number}: val_loss={result:.4f}  params={cfg}")
    return result


def main():
    global _base_cfg, _tokenizer, _train_loader, _val_loader
    global _shared_encoder, _device, _extra_id_0_id

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",   default="configs/v13.yaml")
    parser.add_argument("--n-trials", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Override batch size (default: from config)")
    args = parser.parse_args()

    with open(args.config) as f:
        _base_cfg = yaml.safe_load(f)

    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {_device}")

    _tokenizer = AutoTokenizer.from_pretrained(_base_cfg["model"]["encoder_name"])
    _extra_id_0_id = _tokenizer.convert_tokens_to_ids("<extra_id_0>")

    batch_size = args.batch_size or _base_cfg["training"]["batch_size"]
    data_cfg = _base_cfg["data"]

    # Use QDep≤2 subset for sweep speed — covers all separator patterns
    from src.data.dataset import FOLDatasetV3 as DS
    train_ds = DS(data_cfg["train_path"], _tokenizer,
                  data_cfg["max_input_len"], data_cfg["max_target_len"], max_qdep=2)
    val_ds   = DS(data_cfg["val_path"],   _tokenizer,
                  data_cfg["max_input_len"], data_cfg["max_target_len"], max_qdep=2)

    collator = FOLCollatorV2(_tokenizer.pad_token_id)
    _train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                               collate_fn=collator, num_workers=4, pin_memory=True)
    _val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                               collate_fn=collator, num_workers=4, pin_memory=True)

    # Shared frozen encoder — loaded once, reused across all trials
    _shared_encoder = FrozenT5Encoder(_base_cfg["model"]["encoder_name"]).to(_device)

    print(f"Train: {len(train_ds):,}  Val: {len(val_ds):,}  Batch: {batch_size}")
    print(f"Sweep steps per trial: {SWEEP_STEPS}  Warmup: {WARMUP_STEPS}")
    print(f"Starting {args.n_trials} trials...\n")

    study = optuna.create_study(
        direction="minimize",
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=1000),
        study_name="fol-slm-v13-hpo",
        storage="sqlite:///outputs/optuna_v13.db",
        load_if_exists=True,
    )

    study.optimize(objective, n_trials=args.n_trials)

    print(f"\n{'='*60}")
    print("SWEEP COMPLETE")
    print(f"{'='*60}")
    print(f"Best trial: #{study.best_trial.number}")
    print(f"Best val_loss: {study.best_value:.4f}")
    print(f"Best params:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")

    print(f"\nTop 5 trials:")
    trials = sorted(study.trials, key=lambda t: t.value if t.value is not None else float("inf"))
    for t in trials[:5]:
        print(f"  Trial #{t.number}: val_loss={t.value:.4f}  params={t.params}")


if __name__ == "__main__":
    main()
