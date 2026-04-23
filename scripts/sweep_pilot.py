"""Phase 2: Short-run pilot sweeps.

Runs sequential sweeps over key hyperparameters, each for 15% of full training.
Each sweep dimension uses the best result from the previous sweep.

Usage:
    python scripts/sweep_pilot.py --config configs/phase2_sweep.yaml
"""

import sys
import copy
import yaml
import torch
from dotenv import load_dotenv
load_dotenv()
from pathlib import Path
from transformers import AutoTokenizer
from torch.utils.data import DataLoader

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.model import FOLModel
from src.model.encoder import FrozenT5Encoder
from src.data.dataset import ReasoningDataset
from src.data.collator import FOLCollator
from src.training.trainer import Trainer
from src.utils.factory import create_model_fresh_decoder


def run_sweep(
    sweep_name: str,
    param_key: str,
    param_values: list,
    base_cfg: dict,
    tokenizer,
    train_loader,
    val_loader,
    shared_encoder,
    device,
):
    """Run a single sweep dimension and return the best value."""
    print(f"\n{'='*60}")
    print(f"SWEEP: {sweep_name}")
    print(f"Values: {param_values}")
    print(f"{'='*60}")

    results = []

    for value in param_values:
        cfg = copy.deepcopy(base_cfg)

        # Set the sweep parameter — handle nested keys
        if "." in param_key:
            section, key = param_key.split(".", 1)
            cfg[section][key] = value
        else:
            cfg["training"][param_key] = value

        run_name = f"sweep-{sweep_name}-{value}"
        cfg["logging"]["run_name"] = run_name
        cfg["training"]["output_dir"] = f"outputs/phase2_sweeps/{run_name}/"

        print(f"\n--- {run_name} ---")

        model = create_model_fresh_decoder(
            cfg["model"], vocab_size=tokenizer.vocab_size,
            shared_encoder=shared_encoder,
        )
        model = model.to(device)

        trainer = Trainer(
            model, train_loader, val_loader,
            cfg["training"], device,
            logging_config=cfg.get("logging"),
            diagnostics_config=cfg.get("diagnostics"),
        )

        result = trainer.train()
        val_loss = result["final_val_loss"]
        ca_entropy = result["final_val_entropy"]

        results.append({
            "value": value,
            "val_loss": val_loss,
            "cross_attn_entropy": ca_entropy,
        })
        print(f"  val_loss: {val_loss:.4f}  cross_attn_entropy: {ca_entropy:.4f}")

    # Find best: within 0.01 val_loss of the minimum, prefer lower cross-attention entropy
    VAL_LOSS_GAP = 0.01
    min_val_loss = min(r["val_loss"] for r in results)
    candidates = [r for r in results if r["val_loss"] - min_val_loss <= VAL_LOSS_GAP]
    best = min(candidates, key=lambda r: r["cross_attn_entropy"])
    print(f"\nBest {sweep_name}: {best['value']} (val_loss={best['val_loss']:.4f}, entropy={best['cross_attn_entropy']:.4f})")

    # Print summary table
    print(f"\n{'Value':<15} {'Val Loss':<12} {'CA Entropy':<12}")
    print("-" * 39)
    for r in results:
        marker = " <-- best" if r["value"] == best["value"] else ""
        print(f"{str(r['value']):<15} {r['val_loss']:<12.4f} {r['cross_attn_entropy']:<12.4f}{marker}")

    return best["value"]


def main(config_path: str = "configs/phase2_sweep.yaml"):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["encoder_name"])

    train_dataset = ReasoningDataset(
        cfg["data"]["train_path"], tokenizer,
        cfg["data"]["max_input_len"], cfg["data"]["max_target_len"],
    )
    val_dataset = ReasoningDataset(
        cfg["data"]["val_path"], tokenizer,
        cfg["data"]["max_input_len"], cfg["data"]["max_target_len"],
    )

    collator = FOLCollator(tokenizer.pad_token_id)
    train_loader = DataLoader(train_dataset, batch_size=cfg["training"]["batch_size"],
                              shuffle=True, collate_fn=collator, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg["training"]["batch_size"],
                            shuffle=False, collate_fn=collator, num_workers=4, pin_memory=True)

    # Load encoder once, share across all sweep trials
    shared_encoder = FrozenT5Encoder(cfg["model"]["encoder_name"]).to(device)

    # Sweep 1: Learning rate
    best_lr = run_sweep(
        "lr", "learning_rate",
        [5e-5, 1e-4, 3e-4, 1e-3],
        cfg, tokenizer, train_loader, val_loader, shared_encoder, device,
    )
    cfg["training"]["learning_rate"] = best_lr

    # Sweep 2: Warmup steps
    best_warmup = run_sweep(
        "warmup", "warmup_steps",
        [0, 250, 500, 1000],  # proportional to 3000-step pilots (4000/8000 would exceed max_steps)
        cfg, tokenizer, train_loader, val_loader, shared_encoder, device,
    )
    cfg["training"]["warmup_steps"] = best_warmup

    # Sweep 3: Decoder layers
    best_layers = run_sweep(
        "layers", "model.decoder_layers",
        [2, 4, 6, 8],
        cfg, tokenizer, train_loader, val_loader, shared_encoder, device,
    )
    cfg["model"]["decoder_layers"] = best_layers

    # Sweep 4: Cross-attention heads
    best_heads = run_sweep(
        "heads", "model.decoder_heads",
        [4, 8, 16],
        cfg, tokenizer, train_loader, val_loader, shared_encoder, device,
    )
    cfg["model"]["decoder_heads"] = best_heads

    # Final summary
    print(f"\n{'='*60}")
    print("SWEEP COMPLETE — Best hyperparameters:")
    print(f"  Learning rate:   {best_lr}")
    print(f"  Warmup steps:    {best_warmup}")
    print(f"  Decoder layers:  {best_layers}")
    print(f"  Decoder heads:   {best_heads}")
    print(f"{'='*60}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/phase2_sweep.yaml")
    args = parser.parse_args()
    main(args.config)
