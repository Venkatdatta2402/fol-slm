"""Final training script.

Trains on all data with optimized hyperparameters from Phase 3 (Optuna)
after cross-validation confirms training stability.

Usage:
    python scripts/train_final.py --config configs/final.yaml
"""

import sys
import yaml
import torch
from dotenv import load_dotenv
load_dotenv()
from pathlib import Path
from transformers import AutoTokenizer
from torch.utils.data import DataLoader, WeightedRandomSampler

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.model import FOLModel
from src.data.dataset import ReasoningDataset
from src.data.collator import FOLCollator
from src.training.trainer import Trainer


def main(config_path: str = "configs/final.yaml"):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["encoder_name"])

    collator = FOLCollator(tokenizer.pad_token_id)
    data_cfg = cfg["data"]
    curriculum_cfg = cfg.get("curriculum", {})
    phases = curriculum_cfg.get("phases", [])

    def make_loader(max_qdep=None):
        ds = ReasoningDataset(
            data_cfg["train_path"], tokenizer,
            data_cfg["max_input_len"], data_cfg["max_target_len"],
            max_qdep=max_qdep,
        )
        weights = ds.get_sample_weights()
        sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
        return DataLoader(ds, batch_size=cfg["training"]["batch_size"],
                          sampler=sampler, collate_fn=collator, num_workers=4, pin_memory=True)

    val_dataset = ReasoningDataset(
        data_cfg["val_path"], tokenizer,
        data_cfg["max_input_len"], data_cfg["max_target_len"],
    )
    val_loader = DataLoader(val_dataset, batch_size=cfg["training"]["batch_size"],
                            shuffle=False, collate_fn=collator, num_workers=4, pin_memory=True)

    if phases:
        # Build phase-0 loader (first phase) and curriculum_loaders for subsequent phases
        first_max_qdep = phases[0].get("max_qdep")
        train_loader = make_loader(max_qdep=first_max_qdep)
        print(f"Curriculum phase 1: max_qdep={first_max_qdep}, {len(train_loader.dataset):,} examples")
        curriculum_loaders = []
        for i, phase in enumerate(phases[1:], 1):
            switch_at_step = phases[i - 1]["until_step"]  # end of previous phase = start of this one
            max_qdep = phase.get("max_qdep")
            loader = make_loader(max_qdep=max_qdep)
            curriculum_loaders.append((switch_at_step, loader))
            print(f"Curriculum phase {i + 1}: starts at step {switch_at_step}, max_qdep={max_qdep}, {len(loader.dataset):,} examples")
    else:
        train_loader = make_loader()
        curriculum_loaders = None
        print(f"Training on {len(train_loader.dataset):,} samples")

    print(f"Validating on {len(val_dataset):,} samples")

    model = FOLModel(cfg["model"], vocab_size=tokenizer.vocab_size)
    print(f"Trainable params: {model.trainable_params():,}  /  Total: {model.total_params():,}")

    proof_sentinel_id = tokenizer.convert_tokens_to_ids("<extra_id_3>")
    answer_cls_id = tokenizer.convert_tokens_to_ids("<extra_id_4>")
    extra_id_2_id = tokenizer.convert_tokens_to_ids("<extra_id_2>")
    extra_id_3_id = tokenizer.convert_tokens_to_ids("<extra_id_3>")
    # Map first token of each answer word → class index (True=0, False=1, Unknown=2)
    answer_tok_to_cls = {}
    for label, cls_idx in [("True", 0), ("False", 1), ("Unknown", 2)]:
        toks = tokenizer.encode(label, add_special_tokens=False)
        if toks:
            answer_tok_to_cls[toks[0]] = cls_idx

    trainer = Trainer(
        model, train_loader, val_loader,
        cfg["training"], device,
        logging_config=cfg.get("logging"),
        diagnostics_config=cfg.get("diagnostics"),
        proof_sentinel_id=proof_sentinel_id,
        curriculum_loaders=curriculum_loaders,
        answer_cls_id=answer_cls_id,
        answer_tok_to_cls=answer_tok_to_cls,
        extra_id_2_id=extra_id_2_id,
        extra_id_3_id=extra_id_3_id,
    )

    start_step = 0
    if cfg["training"].get("resume_from"):
        start_step = trainer.resume_from(cfg["training"]["resume_from"])

    result = trainer.train(start_step=start_step)
    print(f"\nFinal training complete.")
    print(f"  Steps: {result['steps_completed']}")
    print(f"  Monitoring val_loss: {result['final_val_loss']:.4f}")
    print(f"  Checkpoint: {cfg['training']['output_dir']}checkpoint_final.pt")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/final.yaml")
    args = parser.parse_args()
    main(args.config)
