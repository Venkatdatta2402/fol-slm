"""Side-by-side inspection of raw ProofWriter data vs preprocessed output.

Groups examples by: answer (True/False/Unknown) x negated/non-negated x QDep
Shows 1-2 examples per unique combination, with full raw and preprocessed data.

Usage:
    python scripts/inspect_preprocessing.py --out-file outputs/preprocessing_inspection.txt
"""

import sys
import json
import re
import argparse
from pathlib import Path
from collections import defaultdict

sys.path.append(str(Path(__file__).resolve().parents[1]))

from scripts.preprocess_proofwriter import (  # type: ignore
    process_question, triple_to_fol, rule_to_fol
)

DEPTHS = ["depth-2", "depth-3", "depth-3ext"]


def is_negated(question_text: str) -> bool:
    return bool(re.search(r'\bnot\b', question_text, re.IGNORECASE))


def load_raw_by_source(depths=DEPTHS, split="test"):
    """Return dict: source_id → record."""
    records = {}
    for depth in depths:
        p = Path(f"data/proofwriter/{depth}/meta-{split}.jsonl")
        if not p.exists():
            continue
        with open(p) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                records[rec["id"]] = rec
    return records


def format_raw(record: dict, question: dict) -> str:
    """Format the raw ProofWriter data for display."""
    lines = []
    lines.append(f"  THEORY: {record['theory']}")
    lines.append("")
    lines.append("  TRIPLES:")
    for k, v in record["triples"].items():
        lines.append(f"    {k}: {v['text']}  →  rep={v['representation']}")
    lines.append("")
    lines.append("  RULES:")
    for k, v in record["rules"].items():
        lines.append(f"    {k}: {v['text']}  →  rep={v['representation']}")
    lines.append("")
    lines.append(f"  QUESTION: {question['question']}")
    lines.append(f"  ANSWER:   {question['answer']}")
    lines.append(f"  QDEP:     {question['QDep']}")
    lines.append(f"  PROOFS:   {question.get('proofs','')[:300]}")
    lines.append("")
    pwi = question.get("proofsWithIntermediates") or []
    if pwi:
        lines.append("  PROOFS_WITH_INTERMEDIATES:")
        for i, p in enumerate(pwi[:2]):  # show first 2 proofs
            lines.append(f"    proof[{i}] representation: {p.get('representation','')}")
            ints = p.get("intermediates") or {}
            for k, v in sorted(ints.items()):
                lines.append(f"      {k}: text={v.get('text','')}  rep={v.get('representation','')}")
    else:
        lines.append("  PROOFS_WITH_INTERMEDIATES: (none)")
    return "\n".join(lines)


def format_preprocessed(ex: dict) -> str:
    lines = []
    lines.append(f"  PREMISES (encoder input):")
    # split at <extra_id_0>
    parts = ex["premises"].split("<extra_id_0>")
    lines.append(f"    NL theory:   {parts[0].strip()}")
    lines.append(f"    NL question: {parts[1].strip() if len(parts)>1 else ''}")
    lines.append("")
    lines.append("  LOGIC (decoder target):")
    for line in ex["logic"].split("\n"):
        lines.append(f"    {line}")
    lines.append("")
    lines.append(f"  ANSWER: {ex['answer']}  QDEP: {ex['qdep']}")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split",    default="test")
    parser.add_argument("--out-file", default="outputs/preprocessing_inspection.txt")
    parser.add_argument("--max-per-cell", type=int, default=2)
    args = parser.parse_args()

    print("Loading raw records...")
    raw_records = load_raw_by_source(split=args.split)
    print(f"  {len(raw_records)} records loaded")

    # Collect all (record, question) pairs and group them
    # Group key: (answer, negated, qdep)
    groups = defaultdict(list)

    for rec in raw_records.values():
        for q in rec.get("questions", {}).values():
            question_text = q.get("question", "")
            answer = q.get("answer")
            qdep = q.get("QDep", 0)
            negated = is_negated(question_text)

            if answer is True:
                answer_str = "True"
            elif answer is False:
                answer_str = "False"
            elif answer == "Unknown":
                answer_str = "Unknown"
            else:
                continue

            neg_str = "negated" if negated else "non-negated"
            key = (answer_str, neg_str, qdep)
            groups[key].append((rec, q))

    # Build output
    out_lines = []
    out_lines.append("=" * 80)
    out_lines.append("RAW vs PREPROCESSED: Side-by-side inspection")
    out_lines.append("Groups: answer × negated × QDep, up to 2 examples per cell")
    out_lines.append("=" * 80)
    out_lines.append("")

    for answer_str in ["True", "False", "Unknown"]:
        for neg_str in ["non-negated", "negated"]:
            # Collect all qdeps for this answer × negated combo
            qdeps = sorted(set(k[2] for k in groups if k[0]==answer_str and k[1]==neg_str))

            out_lines.append("=" * 80)
            out_lines.append(f"  ANSWER={answer_str}  |  {neg_str.upper()}")
            out_lines.append("=" * 80)

            if not qdeps:
                out_lines.append("  (no examples found)")
                out_lines.append("")
                continue

            for qdep in qdeps:
                key = (answer_str, neg_str, qdep)
                examples = groups[key]
                out_lines.append(f"\n  --- QDep={qdep}  ({len(examples)} total examples) ---\n")

                for i, (rec, q) in enumerate(examples[:args.max_per_cell]):
                    out_lines.append(f"  [Example {i+1}]  source={rec['id']}")
                    out_lines.append("")
                    out_lines.append("  >>> RAW <<<")
                    out_lines.append(format_raw(rec, q))
                    out_lines.append("")

                    # Run preprocessing on this example
                    ex = process_question(rec, q)
                    if ex:
                        out_lines.append("  >>> PREPROCESSED <<<")
                        out_lines.append(format_preprocessed(ex))
                    else:
                        out_lines.append("  >>> PREPROCESSED: (skipped / returned None) <<<")

                    out_lines.append("")
                    out_lines.append("  " + "-" * 60)
                    out_lines.append("")

    output = "\n".join(out_lines)

    Path(args.out_file).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_file, "w") as f:
        f.write(output)

    print(f"\nWritten → {args.out_file}")
    print(f"Groups found: {len(groups)}")
    for key in sorted(groups.keys()):
        print(f"  {key}: {len(groups[key])} examples")


if __name__ == "__main__":
    main()
