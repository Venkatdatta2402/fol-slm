"""Phase 4: K-fold cross-validation.

Trains a fresh decoder for each fold using the same frozen encoder
and optimized hyperparameters. Reports mean and std of validation loss.

Usage:
    python scripts/cross_validate.py --config configs/final.yaml --folds 5
"""

import sys
import json
import copy
import yaml
import torch
import numpy as np
from dotenv import load_dotenv
load_dotenv()
from pathlib import Path
from transformers import AutoTokenizer
from torch.utils.data import DataLoader, Subset

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.model.encoder import FrozenT5Encoder
from src.data.dataset import ReasoningDataset
from src.data.collator import FOLCollator
from src.training.trainer import Trainer
from src.utils.factory import create_model_fresh_decoder


def main(config_path: str = "configs/final.yaml", n_folds: int = 5):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["encoder_name"])

    # Load full training dataset
    full_dataset = ReasoningDataset(
        cfg["data"]["train_path"], tokenizer,
        cfg["data"]["max_input_len"], cfg["data"]["max_target_len"],
    )

    n_samples = len(full_dataset)
    indices = np.arange(n_samples)
    np.random.seed(42)
    np.random.shuffle(indices)

    fold_size = n_samples // n_folds
    collator = FOLCollator(tokenizer.pad_token_id)

    # Load encoder once
    shared_encoder = FrozenT5Encoder(cfg["model"]["encoder_name"]).to(device)

    fold_results = []

    print(f"\n{'='*60}")
    print(f"{n_folds}-FOLD CROSS VALIDATION")
    print(f"Dataset size: {n_samples}")
    print(f"{'='*60}")

    for fold in range(n_folds):
        print(f"\n--- Fold {fold + 1}/{n_folds} ---")

        # Split indices
        val_start = fold * fold_size
        val_end = val_start + fold_size if fold < n_folds - 1 else n_samples
        val_indices = indices[val_start:val_end].tolist()
        train_indices = np.concatenate([indices[:val_start], indices[val_end:]]).tolist()

        train_subset = Subset(full_dataset, train_indices)
        val_subset = Subset(full_dataset, val_indices)

        train_loader = DataLoader(
            train_subset, batch_size=cfg["training"]["batch_size"],
            shuffle=True, collate_fn=collator, num_workers=4, pin_memory=True,
        )
        val_loader = DataLoader(
            val_subset, batch_size=cfg["training"]["batch_size"],
            shuffle=False, collate_fn=collator, num_workers=4, pin_memory=True,
        )

        fold_cfg = copy.deepcopy(cfg)
        fold_cfg["training"]["output_dir"] = f"outputs/cv/fold_{fold + 1}/"
        fold_cfg["logging"]["run_name"] = f"cv-fold-{fold + 1}"

        # Fresh decoder, shared encoder
        model = create_model_fresh_decoder(
            fold_cfg["model"], vocab_size=tokenizer.vocab_size,
            shared_encoder=shared_encoder,
        )
        model = model.to(device)

        trainer = Trainer(
            model, train_loader, val_loader,
            fold_cfg["training"], device,
            logging_config=fold_cfg.get("logging"),
            diagnostics_config=fold_cfg.get("diagnostics"),
        )

        result = trainer.train()
        val_loss = result["final_val_loss"]
        fold_results.append(val_loss)
        print(f"  Fold {fold + 1} val_loss: {val_loss:.4f}")

    # Summary
    mean_loss = np.mean(fold_results)
    std_loss = np.std(fold_results)
    cv_pct = (std_loss / mean_loss) * 100 if mean_loss > 0 else 0

    print(f"\n{'='*60}")
    print("CROSS-VALIDATION RESULTS")
    print(f"{'='*60}")
    for i, loss in enumerate(fold_results):
        print(f"  Fold {i + 1}: {loss:.4f}")
    print(f"\n  Mean val_loss: {mean_loss:.4f}")
    print(f"  Std val_loss:  {std_loss:.4f}")
    print(f"  CV%:           {cv_pct:.1f}%")

    if cv_pct < 10:
        print("\n  Training is STABLE (CV% < 10%)")
    else:
        print("\n  Training is UNSTABLE (CV% >= 10%) — consider reducing LR or adding warmup")

    print(f"{'='*60}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/final.yaml")
    parser.add_argument("--folds", type=int, default=5)
    args = parser.parse_args()
    main(args.config, args.folds)
