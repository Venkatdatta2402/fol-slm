"""Evaluate translation decoder (Phase 1) of FOLModelV2.

Generates FOL premises + question from NL input and scores against gold.
Runs all GPU inference first, then scores in parallel on CPU.

Usage:
    python scripts/evaluate_translation.py \
        --checkpoint outputs/v12_translation_4L512/checkpoint_final.pt \
        --config configs/v12_translation.yaml \
        --input-file data/processed/test.jsonl \
        --batch-size 64 \
        --workers 8
"""

import sys
import json
import argparse
import yaml
import torch
from pathlib import Path
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.model import FOLModelV2
from transformers import AutoTokenizer

from scripts.evaluate import (  # type: ignore
    _split_fol_statements,
    score_premises_fol,
)


def generate_batch(model, tokenizer, input_texts, cfg, device):
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
    # Start token
    cur_ids  = torch.full((B, 1), extra_id_1, dtype=torch.long, device=device)
    finished = torch.zeros(B, dtype=torch.bool, device=device)
    all_tokens = [cur_ids]

    with torch.no_grad():
        encoder_out = model.encoder(enc["input_ids"], enc["attention_mask"])

        # First step: full prefix, build KV cache
        logits, past_kv = model.translation_decoder(
            cur_ids, encoder_out, enc["attention_mask"], use_cache=True
        )
        next_tokens = logits[:, -1].argmax(-1)
        finished   |= (next_tokens == extra_id_3) | (next_tokens == eos)
        all_tokens.append(next_tokens.unsqueeze(1))

        # Subsequent steps: one new token at a time, reuse KV cache
        for _ in range(max_len - 1):
            if finished.all():
                break
            cur_ids = next_tokens.unsqueeze(1).clone()
            cur_ids[finished] = pad

            logits, past_kv = model.translation_decoder(
                cur_ids, encoder_out, enc["attention_mask"],
                past_key_values=past_kv, use_cache=True,
            )
            next_tokens = logits[:, -1].argmax(-1)
            next_tokens[finished] = pad
            finished   |= (next_tokens == extra_id_3) | (next_tokens == eos)
            all_tokens.append(next_tokens.unsqueeze(1))

    decoder_ids = torch.cat(all_tokens, dim=1)  # (B, total_len)
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
            q_end    = decoded_str.index("<extra_id_3>")
            question = decoded_str[q_start:q_end].strip()
        except ValueError:
            question = decoded_str[q_start:].strip()
        return decoded_str[prem_start:prem_end].strip(), question
    except ValueError:
        return "", ""


def score_one(args):
    gold_prem, gold_q, pred_prem, pred_q = args
    gold_lines = _split_fol_statements(gold_prem)
    pred_lines = _split_fol_statements(pred_prem)
    prem_score, _ = score_premises_fol(gold_lines, pred_lines)
    q_match = 1 if pred_q == gold_q else 0
    return prem_score, q_match


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint",  default="outputs/v12_translation_4L512/checkpoint_final.pt")
    parser.add_argument("--config",      default="configs/v12_translation.yaml")
    parser.add_argument("--input-file",  default="data/processed/test.jsonl")
    parser.add_argument("--max-samples", type=int, default=0, help="0 = full dataset")
    parser.add_argument("--batch-size",  type=int, default=64)
    parser.add_argument("--workers",     type=int, default=min(8, cpu_count()),
                        help="CPU workers for parallel scoring")
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
    print(f"Samples: {len(samples)}  |  batch_size: {args.batch_size}  |  workers: {args.workers}")

    # ── Phase 1: GPU inference (all batches, no scoring) ─────────────────────
    score_args = []
    batches = range(0, len(samples), args.batch_size)
    for start in tqdm(batches, desc="GPU inference"):
        batch   = samples[start : start + args.batch_size]
        texts   = [s["premises"] for s in batch]
        decoded = generate_batch(model, tokenizer, texts, cfg, device)

        for sample, dec in zip(batch, decoded):
            gold_prem, gold_q = parse_gold(sample["logic"])
            pred_prem, pred_q = parse_pred(dec)
            score_args.append((gold_prem, gold_q, pred_prem, pred_q.strip()))

    # ── Phase 2: parallel CPU scoring ────────────────────────────────────────
    print(f"Scoring {len(score_args)} samples with {args.workers} workers...")
    with Pool(processes=args.workers) as pool:
        results = list(tqdm(
            pool.imap(score_one, score_args, chunksize=64),
            total=len(score_args),
            desc="Scoring",
        ))

    prem_scores = [r[0] for r in results]
    q_exact     = [r[1] for r in results]

    n        = len(prem_scores)
    avg_prem = sum(prem_scores) / n * 100
    q_acc    = sum(q_exact)    / n * 100
    print(f"\n=== Translation Decoder Evaluation ({n} samples) ===")
    print(f"Premises-FOL avg score : {avg_prem:.1f}%")
    print(f"Question FOL exact match: {q_acc:.1f}%  ({sum(q_exact)}/{n})")


if __name__ == "__main__":
    main()
