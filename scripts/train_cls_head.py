"""Train only the answer classification head on a frozen v8 decoder.

The entire encoder + decoder is frozen. Only answer_cls_head (Linear 512→3)
is trainable. This fully decouples answer prediction from proof generation:
- Decoder learned proofs via CE loss during v8 training
- Cls head learns to read the proof representation and classify

At <extra_id_4>, the hidden state has attended over:
  - premises (via frozen T5 encoder + cross-attention)
  - FOL premises + full proof chain (via causal self-attention, teacher-forced)

Usage:
    python scripts/train_cls_head.py \
        --checkpoint outputs/final_v8/checkpoint_final.pt \
        --config configs/final.yaml \
        --lr 1e-3 \
        --max-steps 3000 \
        --out-dir outputs/cls_head_v8/
"""

import sys
import argparse
import os
import yaml
import torch
import torch.nn.functional as F
from pathlib import Path
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.model import FOLModel
from src.data.dataset import ReasoningDataset

ANSWER_TO_IDX = {"True": 0, "False": 1, "Unknown": 2}


# ── Dataset wrapper ───────────────────────────────────────────────────────────

class ClsDataset(Dataset):
    """Wraps ReasoningDataset to add integer answer_label to each item."""
    def __init__(self, base_ds: ReasoningDataset):
        self.base = base_ds

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        item = self.base[idx]
        answer = self.base.samples[idx].get("answer", "Unknown")
        item["answer_label"] = torch.tensor(ANSWER_TO_IDX[answer], dtype=torch.long)
        return item


class ClsCollator:
    def __init__(self, pad_id: int):
        self.pad_id = pad_id

    def __call__(self, batch):
        answer_labels = torch.stack([x.pop("answer_label") for x in batch])
        input_ids = pad_sequence(
            [x["input_ids"] for x in batch], batch_first=True, padding_value=self.pad_id
        )
        attention_mask = pad_sequence(
            [x["attention_mask"] for x in batch], batch_first=True, padding_value=0
        )
        decoder_input_ids = pad_sequence(
            [x["decoder_input_ids"] for x in batch], batch_first=True, padding_value=self.pad_id
        )
        labels = pad_sequence(
            [x["labels"] for x in batch], batch_first=True, padding_value=-100
        )
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "decoder_input_ids": decoder_input_ids,
            "labels": labels,
            "answer_label": answer_labels,
        }


# ── Helpers ───────────────────────────────────────────────────────────────────

def get_cls_hidden(model, decoder_input_ids, extra_id_4_token):
    """Extract hidden state at <extra_id_4> position for each sample in batch.
    Falls back to last token position if sentinel is truncated out."""
    B, T = decoder_input_ids.shape
    # last_hidden: (B, T, d_model) — stored by decoder after forward()
    hidden = model.decoder.last_hidden  # (B, T, d_model)
    positions = []
    for i in range(B):
        matches = (decoder_input_ids[i] == extra_id_4_token).nonzero(as_tuple=True)[0]
        pos = matches[0].item() if len(matches) > 0 else T - 1
        positions.append(pos)
    pos_tensor = torch.tensor(positions, device=hidden.device)
    return hidden[torch.arange(B, device=hidden.device), pos_tensor]  # (B, d_model)


@torch.no_grad()
def evaluate(model, val_loader, device, use_bf16, extra_id_4_token):
    model.eval()
    total, correct = 0, 0
    per_class_correct = {0: 0, 1: 0, 2: 0}
    per_class_total   = {0: 0, 1: 0, 2: 0}

    for batch in val_loader:
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        decoder_input  = batch["decoder_input_ids"].to(device)
        answer_labels  = batch["answer_label"].to(device)

        dtype = torch.bfloat16 if use_bf16 else torch.float32
        with torch.amp.autocast("cuda", dtype=dtype, enabled=use_bf16):
            model(input_ids, attention_mask, decoder_input)

        cls_hidden = get_cls_hidden(model, decoder_input, extra_id_4_token).detach()
        cls_logits = model.answer_cls_head(cls_hidden.float())  # (B, 3)
        preds = cls_logits.argmax(dim=-1)

        correct += (preds == answer_labels).sum().item()
        total   += answer_labels.size(0)
        for c in range(3):
            mask = answer_labels == c
            per_class_correct[c] += (preds[mask] == c).sum().item()
            per_class_total[c]   += mask.sum().item()

    model.train()
    idx_to_ans = {0: "True", 1: "False", 2: "Unknown"}
    print(f"  Val accuracy: {correct/total*100:.1f}%")
    for c in range(3):
        n = per_class_total[c]
        acc = per_class_correct[c] / n * 100 if n > 0 else 0
        print(f"    {idx_to_ans[c]:8s}: {per_class_correct[c]:4d}/{n:4d} = {acc:.1f}%")
    return correct / total


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="outputs/final_v8/checkpoint_final.pt")
    parser.add_argument("--config",     default="configs/final.yaml")
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--max-steps",  type=int,   default=3000)
    parser.add_argument("--out-dir",    default="outputs/cls_head_v8/")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_bf16 = cfg["training"].get("bf16", False)
    batch_size = cfg["training"]["batch_size"]
    data_cfg   = cfg["data"]

    print(f"Device:     {device}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"lr={args.lr}  max_steps={args.max_steps}")

    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["encoder_name"])
    extra_id_4_token = tokenizer.convert_tokens_to_ids("<extra_id_4>")
    print(f"<extra_id_4> token id: {extra_id_4_token}")

    collator = ClsCollator(tokenizer.pad_token_id)

    train_base = ReasoningDataset(
        data_cfg["train_path"], tokenizer,
        data_cfg["max_input_len"], data_cfg["max_target_len"],
    )
    train_ds = ClsDataset(train_base)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              collate_fn=collator, num_workers=4, pin_memory=True)

    val_base = ReasoningDataset(
        data_cfg["val_path"], tokenizer,
        data_cfg["max_input_len"], data_cfg["max_target_len"],
    )
    val_ds = ClsDataset(val_base)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            collate_fn=collator, num_workers=4, pin_memory=True)

    # ── Load model, freeze all except cls head ────────────────────────────────
    model = FOLModel(cfg["model"], vocab_size=tokenizer.vocab_size)
    ckpt  = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"], strict=False)
    model = model.to(device)

    for p in model.parameters():
        p.requires_grad = False
    for p in model.answer_cls_head.parameters():
        p.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"\nTrainable: {trainable:,} / {total:,} params (cls head only)")

    optimizer = torch.optim.AdamW(model.answer_cls_head.parameters(), lr=args.lr, weight_decay=1e-2)

    # ── Training loop ─────────────────────────────────────────────────────────
    os.makedirs(args.out_dir, exist_ok=True)
    model.train()
    data_iter = iter(train_loader)
    label_names = {0: "True", 1: "False", 2: "Unknown"}

    for step in range(args.max_steps):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            batch = next(data_iter)

        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        decoder_input  = batch["decoder_input_ids"].to(device)
        answer_labels  = batch["answer_label"].to(device)

        dtype = torch.bfloat16 if use_bf16 else torch.float32
        with torch.amp.autocast("cuda", dtype=dtype, enabled=use_bf16):
            with torch.no_grad():
                model(input_ids, attention_mask, decoder_input)

        # Detach: cls head trains on frozen proof representation
        cls_hidden = get_cls_hidden(model, decoder_input, extra_id_4_token).detach()
        cls_logits = model.answer_cls_head(cls_hidden.float())  # (B, 3)

        loss = F.cross_entropy(cls_logits, answer_labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 50 == 0:
            preds = cls_logits.argmax(dim=-1)
            acc = (preds == answer_labels).float().mean().item()
            print(f"[step {step:5d}] loss: {loss.item():.4f}  batch_acc: {acc*100:.1f}%  lr: {args.lr:.1e}")

        if (step + 1) % 500 == 0:
            print(f"\n[step {step+1}] Validation:")
            evaluate(model, val_loader, device, use_bf16, extra_id_4_token)
            print()

        if (step + 1) % 1000 == 0:
            path = os.path.join(args.out_dir, f"checkpoint_{step+1}.pt")
            torch.save({"model_state": model.state_dict(), "step": step + 1}, path)
            print(f"  Saved {path}")

    # Final
    print("\nFinal validation:")
    evaluate(model, val_loader, device, use_bf16, extra_id_4_token)
    path = os.path.join(args.out_dir, "checkpoint_final.pt")
    torch.save({"model_state": model.state_dict(), "step": args.max_steps}, path)
    print(f"Saved → {path}")


if __name__ == "__main__":
    main()
