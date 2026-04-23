"""Evaluate full FOLModelV2 pipeline (translation + proof decoder).

Generates proof chain autoregressively:
  1. Translation decoder: NL → FOL premises + question
  2. Proof decoder: FOL → proof chain (cross-attends to translation hidden states)

Answer is extracted from the generated proof output (greedy token after <extra_id_4>).

Usage:
    python scripts/evaluate_v12.py \
        --proof-checkpoint outputs/v12_proof/checkpoint_final.pt \
        --config configs/v12_proof.yaml \
        --input-file data/processed/test.jsonl \
        --max-samples 200 \
        --output-file outputs/eval_v12_200.jsonl
"""

import sys
import json
import argparse
import yaml
import torch
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

import nltk
nltk.download("wordnet", quiet=True)

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.model import FOLModelV2
from transformers import AutoTokenizer

from scripts.evaluate import (  # type: ignore
    parse_output,
    score_premises_fol,
    score_proof,
    extract_answer_word,
)

IDX_TO_ANS = {0: "true", 1: "false", 2: "unknown"}


def generate_v12(model, tokenizer, premises_nl, cfg, device):
    """Full two-decoder generation: NL → FOL translation → proof chain."""
    extra_id_1 = tokenizer.convert_tokens_to_ids("<extra_id_1>")
    extra_id_3 = tokenizer.convert_tokens_to_ids("<extra_id_3>")
    extra_id_4 = tokenizer.convert_tokens_to_ids("<extra_id_4>")
    max_len = cfg["data"]["max_target_len"]
    max_input_len = cfg["data"]["max_input_len"]

    enc = tokenizer(premises_nl, return_tensors="pt",
                    max_length=max_input_len, truncation=True, padding=True).to(device)

    with torch.no_grad():
        # Phase 1: translation decoder — NL → FOL
        encoder_out = model.encoder(enc["input_ids"], enc["attention_mask"])
        trans_ids = torch.tensor([[extra_id_1]], device=device)
        for _ in range(max_len):
            logits = model.translation_decoder(trans_ids, encoder_out, enc["attention_mask"])
            next_tok = logits[0, -1].argmax(-1).item()
            trans_ids = torch.cat([trans_ids, torch.tensor([[next_tok]], device=device)], dim=1)
            if next_tok == extra_id_3 or next_tok == tokenizer.eos_token_id:
                break

        # Re-run translation decoder on the full generated sequence in one shot
        # to get teacher-forced hidden states — matches training distribution
        _ = model.translation_decoder(trans_ids, encoder_out, enc["attention_mask"])
        trans_hidden = model.translation_decoder.last_hidden  # (1, T_trans, d)

        # Build padding mask — format must match training: int64, 1=real token, 0=pad
        # The decoder internally inverts this (== 0) to get True=ignore for MHA
        trans_pad_mask = torch.ones(1, trans_hidden.shape[1], dtype=torch.long, device=device)

        # Phase 2: proof decoder — FOL → proof chain
        proof_ids = torch.tensor([[extra_id_3]], device=device)
        for _ in range(max_len):
            proof_logits = model.forward_proof(trans_hidden, proof_ids, trans_pad_mask)
            next_tok = proof_logits[0, -1].argmax(-1).item()
            proof_ids = torch.cat([proof_ids, torch.tensor([[next_tok]], device=device)], dim=1)
            if next_tok == extra_id_4 or next_tok == tokenizer.eos_token_id:
                break

    # Decode full output: translation part + proof part
    full_ids = torch.cat([trans_ids, proof_ids[:, 1:]], dim=1)  # skip duplicate extra_id_3
    raw_output = tokenizer.decode(full_ids[0], skip_special_tokens=False)
    return raw_output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--proof-checkpoint", default="outputs/v12_proof/checkpoint_final.pt")
    parser.add_argument("--config",           default="configs/v12_proof.yaml")
    parser.add_argument("--input-file",       default="data/processed/test.jsonl")
    parser.add_argument("--max-samples",      type=int, default=200)
    parser.add_argument("--output-file",      default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["encoder_name"])
    model = FOLModelV2(cfg["model"], vocab_size=tokenizer.vocab_size).to(device)

    ckpt = torch.load(args.proof_checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"], strict=False)
    print(f"Loaded: {args.proof_checkpoint}  (step {ckpt.get('step', '?')})")
    model.eval()

    samples = []
    with open(args.input_file) as f:
        for line in f:
            samples.append(json.loads(line))
            if len(samples) >= args.max_samples:
                break

    LABELS = ["true", "false", "unknown"]
    qdep_correct = defaultdict(int)
    qdep_total   = defaultdict(int)
    total_correct = 0
    prem_fol_scores, proof_scores = [], []
    cm_count = {g: {p: 0   for p in LABELS} for g in LABELS}
    cm_prem  = {g: {p: []  for p in LABELS} for g in LABELS}
    cm_proof = {g: {p: []  for p in LABELS} for g in LABELS}

    out_fh = open(args.output_file, "w") if args.output_file else None

    for sample in tqdm(samples, desc="Evaluating"):
        premises   = sample["premises"]
        gold_answer = sample.get("answer", "").strip().lower()
        qdep       = sample.get("qdep", -1)
        gold_logic = sample.get("logic", "")

        raw_output = generate_v12(model, tokenizer, premises, cfg, device)

        parsed_hyp = parse_output(raw_output)
        pred_answer = extract_answer_word(parsed_hyp["answer_text"])

        is_correct = (pred_answer == gold_answer)
        if is_correct:
            total_correct += 1
        qdep_correct[qdep] += int(is_correct)
        qdep_total[qdep]   += 1

        parsed_ref = parse_output(gold_logic)
        prem_score, pred_mapping = score_premises_fol(
            parsed_ref["premises_fol"], parsed_hyp["premises_fol"]
        )
        proof_score = score_proof(
            parsed_ref["proof"], parsed_hyp["proof"], pred_mapping
        )
        prem_fol_scores.append(prem_score)
        proof_scores.append(proof_score)

        g = gold_answer if gold_answer in LABELS else "unknown"
        p = pred_answer if pred_answer in LABELS else "unknown"
        cm_count[g][p] += 1
        cm_prem[g][p].append(prem_score)
        cm_proof[g][p].append(proof_score)

        if out_fh:
            out_fh.write(json.dumps({
                "source": sample.get("source", ""),
                "qdep": qdep,
                "premises": premises,
                "gold_answer": gold_answer,
                "pred_answer": pred_answer,
                "answer_correct": is_correct,
                "prem_fol_score": prem_score,
                "proof_score": proof_score,
                "raw_output": raw_output,
            }) + "\n")

    if out_fh:
        out_fh.close()

    n = len(samples)
    print()
    print("=== EVALUATION RESULTS (V12 Two-Decoder) ===")
    print(f"Samples: {n}")
    print()
    print("Metric 1 — Answer Accuracy")
    print(f"  Overall: {total_correct/n*100:.1f}%")
    for qdep_val in sorted(qdep_total.keys()):
        cnt = qdep_total[qdep_val]
        acc = qdep_correct[qdep_val] / cnt * 100 if cnt > 0 else 0.0
        print(f"  QDep {qdep_val}: {acc:.1f}% ({cnt} samples)")
    print()
    print(f"Metric 2a — Premises-FOL: {sum(prem_fol_scores)/len(prem_fol_scores)*100:.1f}%")
    print(f"Metric 2b — Proof Match:  {sum(proof_scores)/len(proof_scores)*100:.1f}%")
    print()

    col_w = 10
    header = f"{'':10}" + "".join(f"{'pred:'+lbl:>{col_w}}" for lbl in LABELS)
    def _avg(vals): return f"{sum(vals)/len(vals)*100:.1f}%" if vals else "  -"

    print("Confusion Matrix — Answer Counts")
    print(header)
    for g in LABELS:
        print(f"{'gold:'+g:10}" + "".join(f"{cm_count[g][p]:>{col_w}}" for p in LABELS))
    print()

    print("Confusion Matrix — Premises-FOL Avg Score")
    print(header)
    for g in LABELS:
        print(f"{'gold:'+g:10}" + "".join(f"{_avg(cm_prem[g][p]):>{col_w}}" for p in LABELS))
    print()

    print("Confusion Matrix — Proof Avg Score")
    print(header)
    for g in LABELS:
        print(f"{'gold:'+g:10}" + "".join(f"{_avg(cm_proof[g][p]):>{col_w}}" for p in LABELS))


if __name__ == "__main__":
    main()
