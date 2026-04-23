"""Analyze evaluation results: failure case txt files + confusion matrices.

Usage:
    python scripts/analyze_failures.py \
        --eval-file outputs/eval_v8_200.jsonl \
        --test-file data/processed/test.jsonl \
        --out-dir outputs/analysis_v8
"""

import sys
import json
import argparse
from pathlib import Path
from collections import defaultdict

sys.path.append(str(Path(__file__).resolve().parents[1]))
from scripts.evaluate import parse_output  # type: ignore


LABELS = ["true", "false", "unknown"]


def load_gold_by_premises(test_file: str) -> dict:
    gold = {}
    with open(test_file) as f:
        for line in f:
            r = json.loads(line.strip())
            gold[r["premises"]] = r
    return gold


def _raw_section(text: str, start: str, end: str) -> str:
    if start not in text:
        return ""
    after = text.split(start, 1)[1]
    return after.split(end, 1)[0].strip() if end in after else after.strip()


def format_record(rec: dict, gold_rec: dict | None) -> str:
    parsed_pred = parse_output(rec["raw_output"])
    gold_logic = gold_rec["logic"] if gold_rec else ""
    parsed_gold = parse_output(gold_logic) if gold_logic else {}

    gold_prem_fol = parsed_gold.get("premises_fol", [])
    gold_q_fol    = parsed_gold.get("question_fol", "")
    pred_prem_fol = parsed_pred.get("premises_fol", [])
    pred_q_fol    = parsed_pred.get("question_fol", "")

    # Use raw proof sections to preserve full reasoning traces
    gold_proof_raw = _raw_section(gold_logic, "<extra_id_3>", "<extra_id_4>")
    pred_proof_raw = _raw_section(rec["raw_output"], "<extra_id_3>", "<extra_id_4>")
    pred_answer_raw = _raw_section(rec["raw_output"], "<extra_id_4>", "\n\n")
    if not pred_answer_raw:
        pred_answer_raw = rec["raw_output"].split("<extra_id_4>", 1)[-1].strip() if "<extra_id_4>" in rec["raw_output"] else ""

    lines = []
    lines.append("=" * 80)
    lines.append(f"SOURCE:      {rec['source']}")
    lines.append(f"QDEP:        {rec['qdep']}")
    lines.append(f"GOLD ANSWER: {rec['gold_answer'].upper()}")
    lines.append(f"PRED ANSWER: {rec['pred_answer'].upper()}")
    lines.append(f"PREM SCORE:  {rec['prem_fol_score']:.2f}   PROOF SCORE: {rec['proof_score']:.2f}")
    lines.append("")
    lines.append("PREMISES:")
    lines.append(f"  {rec['premises']}")
    lines.append("")
    lines.append("GOLD PREMISES FOL:")
    for s in gold_prem_fol:
        lines.append(f"  {s}")
    lines.append("")
    lines.append(f"GOLD QUESTION FOL:  {gold_q_fol}")
    lines.append("")
    lines.append("GOLD PROOF:")
    for step in (gold_proof_raw.splitlines() if gold_proof_raw else []):
        lines.append(f"  {step}")
    if not gold_proof_raw:
        lines.append("  (empty)")
    lines.append("")
    lines.append("PRED PREMISES FOL:")
    for s in pred_prem_fol:
        lines.append(f"  {s}")
    lines.append("")
    lines.append(f"PRED QUESTION FOL:  {pred_q_fol}")
    lines.append("")
    lines.append("PRED PROOF:")
    for step in (pred_proof_raw.splitlines() if pred_proof_raw else []):
        lines.append(f"  {step}")
    if not pred_proof_raw:
        lines.append("  (empty)")
    lines.append("")
    lines.append(f"PRED ANSWER: {pred_answer_raw}")
    lines.append(f"GOLD ANSWER: {rec['gold_answer']}")
    lines.append("")

    return "\n".join(lines)


def write_cases(records: list, gold_by_source: dict, path: Path, label: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write(f"{label}  ({len(records)} cases)\n\n")
        for rec in records:
            gold_rec = gold_by_source.get(rec["premises"])
            f.write(format_record(rec, gold_rec))
            f.write("\n")
    print(f"  Wrote {len(records):3d} cases → {path}")


def is_negated(premises: str) -> bool:
    """Detect if the question (after <extra_id_0>) contains negation words."""
    if "<extra_id_0>" not in premises:
        return False
    question = premises.split("<extra_id_0>", 1)[1].strip().lower()
    return any(w in question for w in ["not", "n't", "never", "no "])


def render_confusion_matrices(cm_count, cm_prem, cm_proof, title: str) -> str:
    col_w = 12
    header = f"{'':12}" + "".join(f"{'pred:'+lbl:>{col_w}}" for lbl in LABELS)

    def _avg(vals):
        return f"{sum(vals)/len(vals)*100:.1f}%" if vals else "  -"

    lines = []
    lines.append(title)
    lines.append("=" * 60)

    lines.append("\nAnswer Counts (rows=gold, cols=pred)")
    lines.append(header)
    for g in LABELS:
        total = sum(cm_count[g].values())
        correct = cm_count[g][g]
        acc = correct / total * 100 if total > 0 else 0
        row = f"{'gold:'+g:12}" + "".join(f"{cm_count[g][p]:>{col_w}}" for p in LABELS)
        row += f"   total={total}  acc={acc:.1f}%"
        lines.append(row)

    lines.append("\nPremises-FOL Avg Score (rows=gold, cols=pred)")
    lines.append(header)
    for g in LABELS:
        row = f"{'gold:'+g:12}" + "".join(f"{_avg(cm_prem[g][p]):>{col_w}}" for p in LABELS)
        lines.append(row)

    lines.append("\nProof Avg Score (rows=gold, cols=pred)")
    lines.append(header)
    for g in LABELS:
        row = f"{'gold:'+g:12}" + "".join(f"{_avg(cm_proof[g][p]):>{col_w}}" for p in LABELS)
        lines.append(row)

    lines.append("")
    return "\n".join(lines)


def build_cms(records):
    cm_count = {g: {p: 0  for p in LABELS} for g in LABELS}
    cm_prem  = {g: {p: [] for p in LABELS} for g in LABELS}
    cm_proof = {g: {p: [] for p in LABELS} for g in LABELS}
    for rec in records:
        g = rec["gold_answer"] if rec["gold_answer"] in LABELS else "unknown"
        p = rec["pred_answer"] if rec["pred_answer"] in LABELS else "unknown"
        cm_count[g][p] += 1
        cm_prem[g][p].append(rec["prem_fol_score"])
        cm_proof[g][p].append(rec["proof_score"])
    return cm_count, cm_prem, cm_proof


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-file", default="outputs/eval_v8_200.jsonl")
    parser.add_argument("--test-file", default="data/processed/test.jsonl")
    parser.add_argument("--out-dir",   default="outputs/analysis_v8")
    args = parser.parse_args()

    print(f"Loading eval results: {args.eval_file}")
    records = []
    with open(args.eval_file) as f:
        for line in f:
            records.append(json.loads(line.strip()))

    print(f"Loading gold data:    {args.test_file}")
    gold_by_source = load_gold_by_premises(args.test_file)

    out_dir = Path(args.out_dir)

    # --- Split by negation ---
    neg_records  = [r for r in records if is_negated(r["premises"])]
    pos_records  = [r for r in records if not is_negated(r["premises"])]

    # --- Bucket records by gold×pred ---
    buckets: dict[str, dict[str, list]] = {g: {p: [] for p in LABELS} for g in LABELS}
    for rec in records:
        g = rec["gold_answer"] if rec["gold_answer"] in LABELS else "unknown"
        p = rec["pred_answer"] if rec["pred_answer"] in LABELS else "unknown"
        buckets[g][p].append(rec)

    # --- Write all gold×pred txt files ---
    print(f"\nWriting case files to {out_dir}/by_answer/")
    for g in LABELS:
        for p in LABELS:
            recs = buckets[g][p]
            fname = out_dir / "by_answer" / f"gold_{g}__pred_{p}.txt"
            label = f"GOLD={g.upper()}  PRED={p.upper()}"
            write_cases(recs, gold_by_source, fname, label)

    # --- Write Unknown failure files ---
    print(f"\nUnknown failure breakdown:")
    for p in ["true", "false"]:
        recs = buckets["unknown"][p]
        fname = out_dir / "unknown_failures" / f"unknown_pred_{p}.txt"
        label = f"GOLD=UNKNOWN  PRED={p.upper()}  — FAILURES"
        write_cases(recs, gold_by_source, fname, label)

    # --- Confusion matrices: overall, negated, non-negated ---
    cm_out = out_dir / "confusion_matrices.txt"
    cm_out.parent.mkdir(parents=True, exist_ok=True)
    with open(cm_out, "w") as f:
        overall = render_confusion_matrices(*build_cms(records),   title=f"OVERALL  ({len(records)} samples)")
        negated = render_confusion_matrices(*build_cms(neg_records), title=f"NEGATED QUESTIONS  ({len(neg_records)} samples)")
        nonneg  = render_confusion_matrices(*build_cms(pos_records), title=f"NON-NEGATED QUESTIONS  ({len(pos_records)} samples)")
        f.write(overall + "\n\n" + negated + "\n\n" + nonneg)
    print(f"\nConfusion matrices saved → {cm_out}")

    # Print to console too
    print("\n" + overall)
    print(negated)
    print(nonneg)

    print(f"Done. All files in {out_dir}/")


if __name__ == "__main__":
    main()
