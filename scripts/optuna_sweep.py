"""Phase 3: Automated hyperparameter optimization with Optuna.

Performs fine-grained search over decoder hyperparameters using ranges
informed by Phase 2 pilot sweeps.

Usage:
    python scripts/optuna_sweep.py --config configs/phase2_sweep.yaml --n-trials 35
"""

import sys
import copy
import yaml
import torch
import optuna
from dotenv import load_dotenv
load_dotenv()
from optuna.pruners import MedianPruner
from pathlib import Path
from transformers import AutoTokenizer
from torch.utils.data import DataLoader

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.model.encoder import FrozenT5Encoder
from src.data.dataset import ReasoningDataset
from src.data.collator import FOLCollator
from src.training.trainer import Trainer
from src.utils.factory import create_model_fresh_decoder


# Module-level references set in main()
_base_cfg = None
_tokenizer = None
_train_loader = None
_val_loader = None
_shared_encoder = None
_device = None


def objective(trial: optuna.Trial) -> float:
    cfg = copy.deepcopy(_base_cfg)

    # Only LR is searched — all other params fixed to Trial #10 best values.
    # Trial #10: val_loss=0.0977, entropy=3.551, layers=4, heads=8, dropout=0.35,
    #            WD=3e-3, LS=0.05. Those are locked in; only LR varies here.
    lr = trial.suggest_float("lr", 1.2e-3, 2.5e-3, log=True)
    warmup = 500          # 16.7% of 3000 sweep steps
    n_layers = 4
    n_heads = 8
    dropout = 0.35
    weight_decay = 3.00e-3
    label_smoothing = 0.05

    # Apply to config
    cfg["training"]["learning_rate"] = lr
    cfg["training"]["warmup_steps"] = warmup
    cfg["model"]["decoder_layers"] = n_layers
    cfg["model"]["decoder_heads"] = n_heads
    cfg["model"]["decoder_dropout"] = dropout
    cfg["training"]["weight_decay"] = weight_decay
    cfg["training"]["label_smoothing"] = label_smoothing

    run_name = f"optuna-trial-{trial.number}"
    cfg["logging"]["run_name"] = run_name
    cfg["training"]["output_dir"] = f"outputs/optuna/{run_name}/"

    # Create model with fresh decoder, shared encoder
    model = create_model_fresh_decoder(
        cfg["model"], vocab_size=_tokenizer.vocab_size,
        shared_encoder=_shared_encoder,
    )
    model = model.to(_device)

    # Callback for Optuna pruning — report same score formula used for final ranking
    def trial_callback(step: int, val_loss: float, val_entropy: float) -> bool:
        upper_penalty = max(0.0, val_entropy - 3.5) * 0.4287
        lower_penalty = max(0.0, 3.0 - val_entropy) * 0.4287
        score = val_loss + upper_penalty + lower_penalty
        trial.report(score, step)
        return trial.should_prune()

    trainer = Trainer(
        model, _train_loader, _val_loader,
        cfg["training"], _device,
        logging_config=cfg.get("logging"),
        diagnostics_config=cfg.get("diagnostics"),
    )

    result = trainer.train(trial_callback=trial_callback)

    if result["pruned"]:
        raise optuna.TrialPruned()

    val_loss = result["final_val_loss"]
    ca_entropy = result["final_val_entropy"]

    # Two-sided entropy penalty: sweet zone is [2.0, 3.5] nats.
    # Below 2.0 = collapsed to single token (too sharp); above 3.5 = uniform/unfocused.
    # coeff=0.4287 carried over from Phase 3a calibration (std(val_loss)/std(val_entropy)).
    upper_penalty = max(0.0, ca_entropy - 3.5) * 0.4287
    lower_penalty = max(0.0, 2.0 - ca_entropy) * 0.4287
    score = val_loss + upper_penalty + lower_penalty

    trial.set_user_attr("val_loss", val_loss)
    trial.set_user_attr("cross_attn_entropy", ca_entropy)

    return score


def main(config_path: str = "configs/phase2_sweep.yaml", n_trials: int = 35,
         calibration: bool = False):
    global _base_cfg, _tokenizer, _train_loader, _val_loader, _shared_encoder, _device

    with open(config_path) as f:
        _base_cfg = yaml.safe_load(f)

    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {_device}")

    _tokenizer = AutoTokenizer.from_pretrained(_base_cfg["model"]["encoder_name"])

    train_dataset = ReasoningDataset(
        _base_cfg["data"]["train_path"], _tokenizer,
        _base_cfg["data"]["max_input_len"], _base_cfg["data"]["max_target_len"],
    )
    val_dataset = ReasoningDataset(
        _base_cfg["data"]["val_path"], _tokenizer,
        _base_cfg["data"]["max_input_len"], _base_cfg["data"]["max_target_len"],
    )

    collator = FOLCollator(_tokenizer.pad_token_id)
    _train_loader = DataLoader(train_dataset, batch_size=_base_cfg["training"]["batch_size"],
                               shuffle=True, collate_fn=collator, num_workers=4, pin_memory=True)
    _val_loader = DataLoader(val_dataset, batch_size=_base_cfg["training"]["batch_size"],
                             shuffle=False, collate_fn=collator, num_workers=4, pin_memory=True)

    _shared_encoder = FrozenT5Encoder(_base_cfg["model"]["encoder_name"]).to(_device)

    # Calibration mode: disable pruning so all trials run to completion
    n_startup = 999 if calibration else 5
    if calibration:
        print("CALIBRATION MODE: pruning disabled, all trials will run to completion")

    # Create Optuna study with persistent storage
    study = optuna.create_study(
        direction="minimize",
        pruner=MedianPruner(n_startup_trials=n_startup, n_warmup_steps=500),
        study_name="fol-slm-hpo",
        storage="sqlite:///outputs/optuna_study.db",
        load_if_exists=True,
    )

    print(f"\nStarting Optuna optimization with {n_trials} trials")
    study.optimize(objective, n_trials=n_trials)

    # Print results
    print(f"\n{'='*60}")
    print("OPTUNA OPTIMIZATION COMPLETE")
    print(f"{'='*60}")
    print(f"Best trial: #{study.best_trial.number}")
    print(f"Best score: {study.best_value:.4f}")
    print(f"Best params:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    if study.best_trial.user_attrs:
        print(f"Best trial attrs:")
        for key, value in study.best_trial.user_attrs.items():
            print(f"  {key}: {value:.4f}")

    # Top 5 trials
    print(f"\nTop 5 trials:")
    trials = sorted(study.trials, key=lambda t: t.value if t.value is not None else float("inf"))
    for t in trials[:5]:
        print(f"  Trial #{t.number}: score={t.value:.4f} params={t.params}")

    # Calibration summary: compute coeff from non-diverged trials
    if calibration:
        import numpy as np
        completed = [t for t in study.trials
                     if t.state.name == "COMPLETE"
                     and t.user_attrs.get("val_loss", 999) < 5.0]
        if len(completed) >= 2:
            val_losses = [t.user_attrs["val_loss"] for t in completed]
            val_entropies = [t.user_attrs["cross_attn_entropy"] for t in completed]
            std_loss = np.std(val_losses)
            std_entropy = np.std(val_entropies)
            coeff = std_loss / std_entropy if std_entropy > 0 else 0.0

            print(f"\n{'='*60}")
            print("CALIBRATION RESULTS")
            print(f"{'='*60}")
            print(f"Non-diverged trials: {len(completed)}/{len(study.trials)}")
            print(f"\nPer-trial breakdown:")
            for t in completed:
                vl = t.user_attrs['val_loss']
                ve = t.user_attrs['cross_attn_entropy']
                print(f"  Trial #{t.number}: val_loss={vl:.4f}, val_entropy={ve:.4f}")
            print(f"\nval_loss  — mean={np.mean(val_losses):.4f}, std={std_loss:.4f}")
            print(f"val_entropy — mean={np.mean(val_entropies):.4f}, std={std_entropy:.4f}")
            print(f"\ncoeff (equal contribution) = std(val_loss) / std(val_entropy) = {coeff:.4f}")
            print(f"To make entropy dominant, multiply coeff by factor > 1")
        else:
            print(f"\nOnly {len(completed)} non-diverged trials — need at least 2 for calibration")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/phase2_sweep.yaml")
    parser.add_argument("--n-trials", type=int, default=35)
    parser.add_argument("--calibration", action="store_true",
                        help="Disable pruning and print coeff calibration summary")
    args = parser.parse_args()
    main(args.config, args.n_trials, args.calibration)
