import sys
import yaml
import torch
from pathlib import Path
from transformers import AutoTokenizer
from torch.utils.data import DataLoader

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.model import FOLModel
from src.data.dataset import FOLIODataset
from src.data.collator import FOLCollator
from src.training.trainer import Trainer


def main(config_path: str = "configs/base_config.yaml"):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["encoder_name"])

    train_dataset = FOLIODataset(
        cfg["data"]["train_path"], tokenizer,
        cfg["data"]["max_input_len"], cfg["data"]["max_target_len"],
    )
    val_dataset = FOLIODataset(
        cfg["data"]["val_path"], tokenizer,
        cfg["data"]["max_input_len"], cfg["data"]["max_target_len"],
    )

    collator = FOLCollator(tokenizer.pad_token_id)
    train_loader = DataLoader(train_dataset, batch_size=cfg["training"]["batch_size"],
                              shuffle=True, collate_fn=collator, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg["training"]["batch_size"],
                            shuffle=False, collate_fn=collator, num_workers=4, pin_memory=True)

    model = FOLModel(cfg["model"], vocab_size=tokenizer.vocab_size)
    print(f"Trainable params: {model.trainable_params():,}  /  Total: {model.total_params():,}")

    trainer = Trainer(model, train_loader, val_loader, cfg["training"], device)
    trainer.train()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base_config.yaml")
    args = parser.parse_args()
    main(args.config)
