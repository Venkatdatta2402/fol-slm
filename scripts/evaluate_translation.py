"""Evaluate translation decoder (Phase 1) of FOLModelV2.

Generates FOL premises + question from NL input and scores against gold.

Usage:
    python scripts/evaluate_translation.py \
        --checkpoint outputs/v12_translation_4L512/checkpoint_final.pt \
        --config configs/v12_translation.yaml \
        --input-file data/processed/test.jsonl \
        --batch-size 64
"""

import sys
import json
import argparse
import yaml
import torch
from pathlib import Path
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.model import FOLModelV2
from transformers import AutoTokenizer

from scripts.evaluate import (  # type: ignore
    _split_fol_statements,
    score_premises_fol,
)


def generate_batch(model, tokenizer, input_texts, cfg, device):
    """Batched greedy decode: returns one decoded string per input."""
    extra_id_1 = tokenizer.convert_tokens_to_ids("<extra_id_1>")
    extra_id_3 = tokenizer.convert_tokens_to_ids("<extra_id_3>")
    eos        = tokenizer.eos_token_id
    pad        = tokenizer.pad_token_id
    max_len    = cfg["data"]["max_target_len"]

    enc = tokenizer(
        input_texts,
        return_tensors="pt",
        max_length=cfg["data"]["max_input_len"],
        truncation=True,
        padding=True,
    ).to(device)

    B = len(input_texts)
    decoder_ids = torch.full((B, 1), extra_id_1, dtype=torch.long, device=device)
    finished    = torch.zeros(B, dtype=torch.bool, device=device)

    with torch.no_grad():
        encoder_out = model.encoder(enc["input_ids"], enc["attention_mask"])
        for _ in range(max_len):
            logits      = model.translation_decoder(decoder_ids, encoder_out, enc["attention_mask"])
            next_tokens = logits[:, -1].argmax(-1)
            next_tokens[finished] = pad
            decoder_ids = torch.cat([decoder_ids, next_tokens.unsqueeze(1)], dim=1)
            finished   |= (next_tokens == extra_id_3) | (next_tokens == eos)
            if finished.all():
                break

    return [tokenizer.decode(seq, skip_special_tokens=False) for seq in decoder_ids]


def parse_gold(logic_str):
    try:
        prem_start = logic_str.index("<extra_id_1>") + len("<extra_id_1>")
        prem_end   = logic_str.index("<extra_id_2>")
        q_start    = prem_end + len("<extra_id_2>")
        q_end      = logic_str.index("<extra_id_3>")
        return logic_str[prem_start:prem_end].strip(), logic_str[q_start:q_end].strip()
    except ValueError:
        return "", ""


def parse_pred(decoded_str):
    try:
        prem_start = decoded_str.index("<extra_id_1>") + len("<extra_id_1>")
        prem_end   = decoded_str.index("<extra_id_2>")
        q_start    = prem_end + len("<extra_id_2>")
        try:
            q_end  = decoded_str.index("<extra_id_3>")
            question = decoded_str[q_start:q_end].strip()
        except ValueError:
            question = decoded_str[q_start:].strip()
        return decoded_str[prem_start:prem_end].strip(), question
    except ValueError:
        return "", ""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint",  default="outputs/v12_translation_4L512/checkpoint_final.pt")
    parser.add_argument("--config",      default="configs/v12_translation.yaml")
    parser.add_argument("--input-file",  default="data/processed/test.jsonl")
    parser.add_argument("--max-samples", type=int, default=0, help="0 = full dataset")
    parser.add_argument("--batch-size",  type=int, default=64)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["encoder_name"])
    model     = FOLModelV2(cfg["model"], vocab_size=tokenizer.vocab_size).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"], strict=False)
    model.eval()
    print(f"Loaded: {args.checkpoint}  (step {ckpt.get('step', '?')})  device={device}")

    samples = []
    with open(args.input_file) as f:
        for line in f:
            samples.append(json.loads(line))
            if args.max_samples and len(samples) >= args.max_samples:
                break

    prem_scores, q_exact = [], []

    batches = range(0, len(samples), args.batch_size)
    for start in tqdm(batches, desc="Evaluating"):
        batch   = samples[start : start + args.batch_size]
        texts   = [s["premises"] for s in batch]
        decoded = generate_batch(model, tokenizer, texts, cfg, device)

        for sample, dec in zip(batch, decoded):
            gold_prem, gold_q = parse_gold(sample["logic"])
            pred_prem, pred_q = parse_pred(dec)

            gold_lines = _split_fol_statements(gold_prem)
            pred_lines = _split_fol_statements(pred_prem)
            score, _   = score_premises_fol(gold_lines, pred_lines)
            prem_scores.append(score)
            q_exact.append(1 if pred_q.strip() == gold_q.strip() else 0)

    n        = len(prem_scores)
    avg_prem = sum(prem_scores) / n * 100
    q_acc    = sum(q_exact)    / n * 100
    print(f"\n=== Translation Decoder Evaluation ({n} samples) ===")
    print(f"Premises-FOL avg score : {avg_prem:.1f}%")
    print(f"Question FOL exact match: {q_acc:.1f}%  ({sum(q_exact)}/{n})")


if __name__ == "__main__":
    main()
