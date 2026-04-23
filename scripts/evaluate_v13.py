"""Evaluate full FOLModelV3 pipeline (translation decoder → proof decoder).

Inference procedure:
  1. Translation decoder: NL → [<extra_id_1> FOL_premises <extra_id_2> FOL_question <extra_id_3>]
  2. Extract slice from <extra_id_2> onward as proof decoder prefix
  3. Proof decoder: autoregressively generates proof chain, cross-attending T5 encoder (premises only)

No hidden-state passing between decoders — proof decoder only receives token IDs as prefix.
No train/inference mismatch (unlike V12).

Usage:
    python scripts/evaluate_v13.py \
        --checkpoint outputs/v13/checkpoint_final.pt \
        --config configs/v13.yaml \
        --input-file data/processed/test.jsonl \
        --max-samples 200 \
        --output-file outputs/eval_v13_200.jsonl
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

from src.model import FOLModelV3
from src.utils.attention import build_premises_cross_attn_mask
from transformers import AutoTokenizer

from scripts.evaluate import (  # type: ignore
    parse_output,
    score_premises_fol,
    score_proof,
    extract_answer_word,
)


def get_cls_answer(model, proof_ids, device):
    """Run cls head on proof decoder last hidden state → answer string."""
    hidden = model.proof_decoder.last_hidden  # (1, T, d_model)
    cls_hidden = hidden[0, -1, :].float()     # last proof position
    logits = model.answer_cls_head(cls_hidden)
    idx = logits.argmax().item()
    return {0: "true", 1: "false", 2: "unknown"}[idx]


def generate_v13(model, tokenizer, premises_nl, cfg, device,
                 extra_id_0, extra_id_1, extra_id_2, extra_id_3, extra_id_4):
    """Full V13 two-decoder generation.

    Step 1: Translation decoder generates FOL sequence (NL → FOL premises + question).
    Step 2: Extract <extra_id_2>-onward slice as proof decoder prefix.
    Step 3: Proof decoder generates proof chain autoregressively,
            cross-attending T5 encoder restricted to NL premises only.
    """
    max_len = cfg["data"]["max_target_len"]
    max_input_len = cfg["data"]["max_input_len"]

    enc = tokenizer(premises_nl, return_tensors="pt",
                    max_length=max_input_len, truncation=True, padding=True).to(device)

    with torch.no_grad():
        # Encode NL input once — shared by both decoders
        encoder_out = model.encoder(enc["input_ids"], enc["attention_mask"])

        # Premises-only cross-attn mask: True = ignore (positions at/after <extra_id_0>)
        premises_mask = build_premises_cross_attn_mask(enc["input_ids"], extra_id_0)

        # --- Phase 1: Translation decoder ---
        # Generates: <extra_id_1> FOL_premises <extra_id_2> FOL_question <extra_id_3>
        trans_ids = torch.tensor([[extra_id_1]], device=device)
        for _ in range(max_len):
            logits = model.translation_decoder(
                trans_ids, encoder_out, enc["attention_mask"]
            )
            next_tok = logits[0, -1].argmax(-1).item()
            trans_ids = torch.cat(
                [trans_ids, torch.tensor([[next_tok]], device=device)], dim=1
            )
            if next_tok == extra_id_3 or next_tok == tokenizer.eos_token_id:
                break

        # --- Phase 2: Proof decoder ---
        # Prefix: slice of trans_ids from <extra_id_2> (inclusive) to end
        # This gives: [<extra_id_2>, FOL_question_tokens, <extra_id_3>]
        id2_positions = (trans_ids[0] == extra_id_2).nonzero(as_tuple=True)[0]
        if len(id2_positions) > 0:
            proof_prefix = trans_ids[:, id2_positions[0]:]  # (1, prefix_len)
        else:
            # Fallback: translation failed to produce <extra_id_2>, use <extra_id_3> only
            id3_positions = (trans_ids[0] == extra_id_3).nonzero(as_tuple=True)[0]
            if len(id3_positions) > 0:
                proof_prefix = trans_ids[:, id3_positions[0]:]
            else:
                proof_prefix = torch.tensor([[extra_id_3]], device=device)

        proof_ids = proof_prefix
        for _ in range(max_len):
            proof_logits = model.forward_proof(
                encoder_out, proof_ids,
                premises_cross_attn_mask=premises_mask,
            )
            next_tok = proof_logits[0, -1].argmax(-1).item()
            proof_ids = torch.cat(
                [proof_ids, torch.tensor([[next_tok]], device=device)], dim=1
            )
            if next_tok == extra_id_4 or next_tok == tokenizer.eos_token_id:
                break

    # Decode full output: translation + proof (proof prefix already in trans_ids, skip overlap)
    id2_in_trans = (trans_ids[0] == extra_id_2).nonzero(as_tuple=True)[0]
    if len(id2_in_trans) > 0:
        trans_part = trans_ids[:, :id2_in_trans[0]]   # up to (not incl.) <extra_id_2>
        full_ids = torch.cat([trans_part, proof_ids], dim=1)
    else:
        full_ids = torch.cat([trans_ids, proof_ids[:, 1:]], dim=1)

    raw_output = tokenizer.decode(full_ids[0], skip_special_tokens=False)

    # Cls head answer prediction from last proof decoder hidden state
    cls_answer = get_cls_answer(model, proof_ids, device)

    return raw_output, cls_answer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint",       default="outputs/v13/checkpoint_final.pt")
    parser.add_argument("--cls-checkpoint",   default=None,
                        help="Cls head checkpoint (loads answer_cls_head weights). "
                             "If omitted, uses cls head weights already in --checkpoint.")
    parser.add_argument("--config",           default="configs/v13.yaml")
    parser.add_argument("--input-file",       default="data/processed/test.jsonl")
    parser.add_argument("--max-samples",      type=int, default=200)
    parser.add_argument("--output-file",      default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["encoder_name"])
    extra_id_0 = tokenizer.convert_tokens_to_ids("<extra_id_0>")
    extra_id_1 = tokenizer.convert_tokens_to_ids("<extra_id_1>")
    extra_id_2 = tokenizer.convert_tokens_to_ids("<extra_id_2>")
    extra_id_3 = tokenizer.convert_tokens_to_ids("<extra_id_3>")
    extra_id_4 = tokenizer.convert_tokens_to_ids("<extra_id_4>")

    model = FOLModelV3(cfg["model"], vocab_size=tokenizer.vocab_size).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"], strict=False)
    print(f"Loaded: {args.checkpoint}  (step {ckpt.get('step', '?')})")

    if args.cls_checkpoint:
        cls_ckpt = torch.load(args.cls_checkpoint, map_location=device, weights_only=False)
        # Load only the cls head weights
        cls_state = {k.replace("answer_cls_head.", ""): v
                     for k, v in cls_ckpt["model_state"].items()
                     if k.startswith("answer_cls_head.")}
        model.answer_cls_head.load_state_dict(cls_state)
        print(f"Loaded cls head: {args.cls_checkpoint}  (step {cls_ckpt.get('step', '?')})")

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
    cm_count = {g: {p: 0  for p in LABELS} for g in LABELS}
    cm_prem  = {g: {p: [] for p in LABELS} for g in LABELS}
    cm_proof = {g: {p: [] for p in LABELS} for g in LABELS}

    out_fh = open(args.output_file, "w") if args.output_file else None

    for sample in tqdm(samples, desc="Evaluating"):
        premises    = sample["premises"]
        gold_answer = sample.get("answer", "").strip().lower()
        qdep        = sample.get("qdep", -1)
        gold_logic  = sample.get("logic", "")

        raw_output, pred_answer = generate_v13(
            model, tokenizer, premises, cfg, device,
            extra_id_0, extra_id_1, extra_id_2, extra_id_3, extra_id_4,
        )

        parsed_hyp = parse_output(raw_output)

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
                "source":        sample.get("source", ""),
                "qdep":          qdep,
                "premises":      premises,
                "gold_answer":   gold_answer,
                "pred_answer":   pred_answer,
                "answer_correct": is_correct,
                "prem_fol_score": prem_score,
                "proof_score":    proof_score,
                "raw_output":     raw_output,
            }) + "\n")

    if out_fh:
        out_fh.close()

    n = len(samples)
    print()
    print("=== EVALUATION RESULTS (V13 Two-Decoder) ===")
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
