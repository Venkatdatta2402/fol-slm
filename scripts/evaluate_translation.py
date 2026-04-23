"""Evaluate translation decoder (Phase 1) of FOLModelV2.

Generates FOL premises + question from NL input and scores against gold.

Usage:
    python scripts/evaluate_translation.py \
        --checkpoint outputs/v12_translation_2L256/checkpoint_final.pt \
        --config configs/v12_translation.yaml \
        --input-file data/processed/test.jsonl \
        --max-samples 200
"""

import sys
import re
import json
import argparse
import yaml
import torch
from pathlib import Path
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.model import FOLModelV2
from transformers import AutoTokenizer

# Import the same fuzzy scoring used by evaluate.py
from scripts.evaluate import (  # type: ignore
    _split_fol_statements,
    score_premises_fol,
)


def generate_translation(model, tokenizer, premises_nl, question_nl, cfg, device):
    """Greedy decode translation decoder output up to <extra_id_3>."""
    extra_id_1 = tokenizer.convert_tokens_to_ids("<extra_id_1>")
    extra_id_3 = tokenizer.convert_tokens_to_ids("<extra_id_3>")

    input_text = f"{premises_nl} {question_nl}"
    enc = tokenizer(input_text, return_tensors="pt",
                    max_length=cfg["data"]["max_input_len"],
                    truncation=True, padding=True).to(device)

    decoder_ids = torch.tensor([[extra_id_1]], device=device)
    max_len = cfg["data"]["max_target_len"]

    model.eval()
    with torch.no_grad():
        encoder_out = model.encoder(enc["input_ids"], enc["attention_mask"])
        for _ in range(max_len):
            logits = model.translation_decoder(decoder_ids, encoder_out, enc["attention_mask"])
            next_token = logits[0, -1].argmax(-1).item()
            decoder_ids = torch.cat([decoder_ids,
                                     torch.tensor([[next_token]], device=device)], dim=1)
            if next_token == extra_id_3 or next_token == tokenizer.eos_token_id:
                break

    decoded = tokenizer.decode(decoder_ids[0], skip_special_tokens=False)
    return decoded


def parse_gold(logic_str):
    """Extract gold FOL premises and question from logic field."""
    extra_id_1 = "<extra_id_1>"
    extra_id_2 = "<extra_id_2>"
    extra_id_3 = "<extra_id_3>"

    try:
        prem_start = logic_str.index(extra_id_1) + len(extra_id_1)
        prem_end   = logic_str.index(extra_id_2)
        q_start    = logic_str.index(extra_id_2) + len(extra_id_2)
        q_end      = logic_str.index(extra_id_3)
        premises = logic_str[prem_start:prem_end].strip()
        question = logic_str[q_start:q_end].strip()
        return premises, question
    except ValueError:
        return "", ""


def parse_pred(decoded_str):
    """Extract predicted FOL premises and question from decoded output."""
    extra_id_1 = "<extra_id_1>"
    extra_id_2 = "<extra_id_2>"
    extra_id_3 = "<extra_id_3>"

    try:
        prem_start = decoded_str.index(extra_id_1) + len(extra_id_1)
        prem_end   = decoded_str.index(extra_id_2)
        q_start    = decoded_str.index(extra_id_2) + len(extra_id_2)
        q_end      = decoded_str.index(extra_id_3)
        premises = decoded_str[prem_start:prem_end].strip()
        question = decoded_str[q_start:q_end].strip()
        return premises, question
    except ValueError:
        # Try without extra_id_3
        try:
            prem_start = decoded_str.index(extra_id_1) + len(extra_id_1)
            prem_end   = decoded_str.index(extra_id_2)
            q_start    = decoded_str.index(extra_id_2) + len(extra_id_2)
            premises = decoded_str[prem_start:prem_end].strip()
            question = decoded_str[q_start:].strip()
            return premises, question
        except ValueError:
            return "", ""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="outputs/v12_translation_2L256/checkpoint_final.pt")
    parser.add_argument("--config",     default="configs/v12_translation.yaml")
    parser.add_argument("--input-file", default="data/processed/test.jsonl")
    parser.add_argument("--max-samples", type=int, default=200)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["encoder_name"])
    model = FOLModelV2(cfg["model"], vocab_size=tokenizer.vocab_size).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"], strict=False)
    print(f"Loaded: {args.checkpoint}  (step {ckpt.get('step', '?')})")
    model.eval()

    samples = []
    with open(args.input_file) as f:
        for line in f:
            samples.append(json.loads(line))
            if len(samples) >= args.max_samples:
                break

    prem_scores, q_exact = [], []
    for sample in tqdm(samples, desc="Evaluating"):
        gold_prem_str, gold_q = parse_gold(sample["logic"])
        gold_q = gold_q.strip()

        decoded = generate_translation(model, tokenizer, sample["premises"], "", cfg, device)
        pred_prem_str, pred_q = parse_pred(decoded)
        pred_q = pred_q.strip()

        gold_lines = _split_fol_statements(gold_prem_str)
        pred_lines = _split_fol_statements(pred_prem_str)
        score, _ = score_premises_fol(gold_lines, pred_lines)
        prem_scores.append(score)
        q_exact.append(1 if pred_q == gold_q else 0)

    avg_prem = sum(prem_scores) / len(prem_scores) * 100
    q_acc    = sum(q_exact) / len(q_exact) * 100
    print(f"\n=== Translation Decoder Evaluation ({len(samples)} samples) ===")
    print(f"Premises-FOL avg score : {avg_prem:.1f}%")
    print(f"Question FOL exact match: {q_acc:.1f}%  ({sum(q_exact)}/{len(q_exact)})")


if __name__ == "__main__":
    main()
