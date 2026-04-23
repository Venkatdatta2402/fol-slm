"""Train answer classification head on frozen FOLModelV3 (V13).

Encoder + translation_decoder + proof_decoder are all frozen.
Only answer_cls_head (Linear 512→3) is trainable.

Hidden state source: proof_decoder.last_hidden at <extra_id_4> position.
The proof decoder input is [<extra_id_2>, FOL_question, <extra_id_3>, gold_proof_{0..t-1}],
so <extra_id_4> appears in proof_labels — we look for it in proof_decoder_input_ids
(it is the last real token shifted in, just before the label <extra_id_4>).
Actually <extra_id_4> is the last label, meaning it is NOT in proof_decoder_input_ids.
We therefore take the hidden state at the last non-pad position (last proof token).

Usage:
    python scripts/train_cls_head_v13.py \
        --checkpoint outputs/v13/checkpoint_final.pt \
        --config configs/v13.yaml \
        --lr 1e-3 \
        --max-steps 3000 \
        --out-dir outputs/cls_head_v13/
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

from src.model import FOLModelV3
from src.data.dataset import FOLDatasetV3
from src.data.collator import FOLCollatorV2
from src.utils.attention import build_premises_cross_attn_mask

ANSWER_TO_IDX = {"True": 0, "False": 1, "Unknown": 2}


class ClsDatasetV3(Dataset):
    """Wraps FOLDatasetV3 to add integer answer_label."""
    def __init__(self, base_ds: FOLDatasetV3):
        self.base = base_ds

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        item = self.base[idx]
        answer = self.base.samples[idx].get("answer", "Unknown")
        item["answer_label"] = torch.tensor(ANSWER_TO_IDX.get(answer, 2), dtype=torch.long)
        return item


class ClsCollatorV3:
    def __init__(self, pad_id: int):
        self.base = FOLCollatorV2(pad_id)

    def __call__(self, batch):
        answer_labels = torch.stack([x.pop("answer_label") for x in batch])
        collated = self.base(batch)
        collated["answer_label"] = answer_labels
        return collated


def get_cls_hidden(model, proof_decoder_input_ids):
    """Extract hidden state at last non-pad token in proof decoder output.

    <extra_id_4> is the last label (target), not in decoder input.
    The last token of proof_decoder_input_ids is the last proof step token.
    We take the hidden state there — it has attended over the full proof chain.
    """
    hidden = model.proof_decoder.last_hidden  # (B, T, d_model)
    B, T, _ = hidden.shape
    # Last position in each sequence (all are the same length post-collation, padded)
    # Use last position — proof decoder input ends just before <extra_id_4>
    return hidden[:, -1, :]  # (B, d_model)


@torch.no_grad()
def evaluate(model, val_loader, device, use_bf16, extra_id_0_id):
    model.eval()
    total, correct = 0, 0
    per_class_correct = {0: 0, 1: 0, 2: 0}
    per_class_total   = {0: 0, 1: 0, 2: 0}

    for batch in val_loader:
        input_ids    = batch["input_ids"].to(device)
        attn_mask    = batch["attention_mask"].to(device)
        trans_input  = batch["trans_decoder_input_ids"].to(device)
        proof_input  = batch["proof_decoder_input_ids"].to(device)
        answer_labels = batch["answer_label"].to(device)

        premises_mask = build_premises_cross_attn_mask(input_ids, extra_id_0_id)

        dtype = torch.bfloat16 if use_bf16 else torch.float32
        with torch.amp.autocast("cuda", dtype=dtype, enabled=use_bf16):
            _, _, encoder_out = model.forward_translation(input_ids, attn_mask, trans_input)
            model.forward_proof(encoder_out, proof_input, premises_cross_attn_mask=premises_mask)

        cls_hidden = get_cls_hidden(model, proof_input).detach()
        cls_logits = model.answer_cls_head(cls_hidden.float())
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="outputs/v13/checkpoint_final.pt")
    parser.add_argument("--config",     default="configs/v13.yaml")
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--max-steps",  type=int,   default=3000)
    parser.add_argument("--out-dir",    default="outputs/cls_head_v13/")
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
    extra_id_0_id = tokenizer.convert_tokens_to_ids("<extra_id_0>")
    collator = ClsCollatorV3(tokenizer.pad_token_id)

    train_base = FOLDatasetV3(data_cfg["train_path"], tokenizer,
                              data_cfg["max_input_len"], data_cfg["max_target_len"])
    train_ds = ClsDatasetV3(train_base)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              collate_fn=collator, num_workers=4, pin_memory=True)

    val_base = FOLDatasetV3(data_cfg["val_path"], tokenizer,
                            data_cfg["max_input_len"], data_cfg["max_target_len"])
    val_ds = ClsDatasetV3(val_base)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            collate_fn=collator, num_workers=4, pin_memory=True)

    model = FOLModelV3(cfg["model"], vocab_size=tokenizer.vocab_size)
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

    optimizer = torch.optim.AdamW(model.answer_cls_head.parameters(),
                                  lr=args.lr, weight_decay=1e-2)

    os.makedirs(args.out_dir, exist_ok=True)
    model.train()
    data_iter = iter(train_loader)

    for step in range(args.max_steps):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            batch = next(data_iter)

        input_ids    = batch["input_ids"].to(device)
        attn_mask    = batch["attention_mask"].to(device)
        trans_input  = batch["trans_decoder_input_ids"].to(device)
        proof_input  = batch["proof_decoder_input_ids"].to(device)
        answer_labels = batch["answer_label"].to(device)

        premises_mask = build_premises_cross_attn_mask(input_ids, extra_id_0_id)

        dtype = torch.bfloat16 if use_bf16 else torch.float32
        with torch.amp.autocast("cuda", dtype=dtype, enabled=use_bf16):
            with torch.no_grad():
                _, _, encoder_out = model.forward_translation(input_ids, attn_mask, trans_input)
                model.forward_proof(encoder_out, proof_input,
                                    premises_cross_attn_mask=premises_mask)

        cls_hidden = get_cls_hidden(model, proof_input).detach()
        cls_logits = model.answer_cls_head(cls_hidden.float())

        loss = F.cross_entropy(cls_logits, answer_labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 50 == 0:
            preds = cls_logits.argmax(dim=-1)
            acc = (preds == answer_labels).float().mean().item()
            print(f"[step {step:5d}] loss: {loss.item():.4f}  batch_acc: {acc*100:.1f}%")

        if (step + 1) % 500 == 0:
            print(f"\n[step {step+1}] Validation:")
            evaluate(model, val_loader, device, use_bf16, extra_id_0_id)
            print()

        if (step + 1) % 1000 == 0:
            path = os.path.join(args.out_dir, f"checkpoint_{step+1}.pt")
            torch.save({"model_state": model.state_dict(), "step": step + 1}, path)
            print(f"  Saved {path}")

    print("\nFinal validation:")
    evaluate(model, val_loader, device, use_bf16, extra_id_0_id)
    path = os.path.join(args.out_dir, "checkpoint_final.pt")
    torch.save({"model_state": model.state_dict(), "step": args.max_steps}, path)
    print(f"Saved → {path}")


if __name__ == "__main__":
    main()
