"""Evaluation script for the FOL SLM seq2seq model.

Computes three metrics against the test set:
  Metric 1  — Answer accuracy (overall + per-QDep breakdown)
  Metric 2a — Premises-FOL match (sorted, predicate-normalised)
  Metric 2b — Proof match (in-order, predicate-normalised)

Usage:
    python scripts/evaluate.py
    python scripts/evaluate.py --checkpoint outputs/final/checkpoint_final.pt \
        --config configs/final.yaml --input-file data/processed/test.jsonl
    python scripts/evaluate.py --max-samples 100 --output-file results.jsonl
"""

import sys
import json
import re
import argparse
import yaml
import torch
from pathlib import Path
from collections import defaultdict

import nltk
nltk.download("wordnet", quiet=True)
from nltk.corpus import wordnet

from tqdm import tqdm

# Make sure project root is on the path so we can import src.*
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.model import FOLModel
from transformers import AutoTokenizer

# Re-use the existing generate() helper
from scripts.generate import generate  # type: ignore


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

# Matches individual FOL statements regardless of newline vs space separation:
#   - universal rule:        forall x (...)
#   - specific entity rule:  Pred(args) -> Pred(args)
#   - negated atom:          not Pred(args)
#   - simple atom:           Pred(args)
_FOL_ATOM_RE = re.compile(
    r'forall\s+x\s+\([^()]*(?:\([^()]*\)[^()]*)*\)'           # universal rule
    r'|(?:not\s+)?[A-Z][a-zA-Z]*\([^)]+\)'                    # atom (pos or neg)
    r'(?:\s*->\s*(?:not\s+)?[A-Z][a-zA-Z]*\([^)]+\))?'       # optional -> conclusion
)

# Matches proof steps: "therefore Pred(args)"
_PROOF_STEP_RE = re.compile(
    r'therefore\s+(?:not\s+)?[A-Z][a-zA-Z]*\([^)]+\)'
)


def _extract_raw(text: str, start_token: str, end_token: str) -> str:
    """Return raw text between start_token and end_token."""
    if start_token not in text or end_token not in text:
        return ""
    return text.split(start_token, 1)[1].split(end_token, 1)[0]


def _split_fol_statements(text: str) -> list[str]:
    """Extract individual FOL statements from text — works on both newline and space separated output."""
    return [m.strip() for m in _FOL_ATOM_RE.findall(text) if m.strip()]


def _split_proof_steps(text: str) -> list[str]:
    """Extract proof steps (therefore ...) from text."""
    return [m.strip() for m in _PROOF_STEP_RE.findall(text) if m.strip()]


def parse_output(text: str) -> dict:
    """Split a model output (or ground-truth logic string) into its three parts.

    Returns:
        {
            "premises_fol": [str, ...],   # FOL statements between <extra_id_1> and <extra_id_2>
            "question_fol": str,          # FOL question between <extra_id_2> and <extra_id_3>
            "proof":        [str, ...],   # proof steps between <extra_id_3> and <extra_id_4>
            "answer_text":  str,          # everything after <extra_id_4>
        }
    """
    prem_raw     = _extract_raw(text, "<extra_id_1>", "<extra_id_2>")
    question_raw = _extract_raw(text, "<extra_id_2>", "<extra_id_3>")
    proof_raw    = _extract_raw(text, "<extra_id_3>", "<extra_id_4>")

    premises_fol = _split_fol_statements(prem_raw)
    question_fol = question_raw.strip()
    proof        = _split_proof_steps(proof_raw)

    answer_text = ""
    if "<extra_id_4>" in text:
        answer_text = text.split("<extra_id_4>", 1)[1].strip()

    return {
        "premises_fol": premises_fol,
        "question_fol": question_fol,
        "proof": proof,
        "answer_text": answer_text,
    }


def extract_answer_word(answer_text: str) -> str:
    """Return the first word of the answer (True / False / Unknown), lowercased."""
    if not answer_text:
        return ""
    return answer_text.strip().split()[0].lower().rstrip(".,;:")


# ---------------------------------------------------------------------------
# Predicate normalisation
# ---------------------------------------------------------------------------

_PREDICATE_RE = re.compile(r"\b([A-Z][a-z]*)\b")


def extract_predicates(line: str) -> list[str]:
    """Return all capitalised word-tokens that look like predicate names.

    Variable names (x, y, z) and entity names that appear in parentheses as
    arguments are also capitalised but we want *predicate* names — those that
    appear before '(' or that stand alone outside parens.  We take a pragmatic
    approach: any capitalised token is a candidate; true entity names (proper
    nouns used as arguments) are typically lower-cased in the dataset after
    normalisation, so this works well in practice.
    """
    return _PREDICATE_RE.findall(line)


def _synsets(word: str) -> set[str]:
    """Return the set of synset names for *word* (all POS)."""
    return {s.name() for s in wordnet.synsets(word.lower())}


def _wordnet_equivalent(p1: str, p2: str) -> bool:
    """True when both predicates share at least one WordNet synset."""
    s1 = _synsets(p1)
    s2 = _synsets(p2)
    return bool(s1 and s2 and s1 & s2)


def _edit_distance(a: str, b: str) -> int:
    """Standard Levenshtein edit distance."""
    a, b = a.lower(), b.lower()
    if a == b:
        return 0
    m, n = len(a), len(b)
    # One-row DP
    prev = list(range(n + 1))
    for i in range(1, m + 1):
        curr = [i] + [0] * n
        for j in range(1, n + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            curr[j] = min(curr[j - 1] + 1, prev[j] + 1, prev[j - 1] + cost)
        prev = curr
    return prev[n]


def build_predicate_mapping(
    pred_list_ref: list[str], pred_list_hyp: list[str]
) -> dict[str, str]:
    """Build a substitution mapping from hypothesis predicates to reference predicates.

    Strategy:
      1. Exact match — no substitution needed.
      2. WordNet synset overlap — treat as equivalent.
      3. Edit distance ≤ 2 fallback for invented/out-of-vocabulary predicates.

    Returns a dict {hyp_pred: ref_pred} only for non-identical pairs.
    """
    ref_set = set(pred_list_ref)
    hyp_set = set(pred_list_hyp)

    # Predicates that differ between hypothesis and reference
    new_in_hyp = hyp_set - ref_set
    only_in_ref = ref_set - hyp_set

    mapping: dict[str, str] = {}
    for hp in new_in_hyp:
        # Try WordNet first
        matched = None
        for rp in only_in_ref:
            if _wordnet_equivalent(hp, rp):
                matched = rp
                break
        if matched is None:
            # Edit-distance fallback
            candidates = [(rp, _edit_distance(hp, rp)) for rp in only_in_ref]
            candidates.sort(key=lambda x: x[1])
            if candidates and candidates[0][1] <= 2:
                matched = candidates[0][0]
        if matched is not None:
            mapping[hp] = matched

    return mapping


def apply_mapping(line: str, mapping: dict[str, str]) -> str:
    """Replace each hypothesis predicate with its mapped reference predicate."""
    if not mapping:
        return line

    def _replace(m: re.Match) -> str:
        word = m.group(1)
        return mapping.get(word, word)

    return _PREDICATE_RE.sub(_replace, line)


def normalise_line(line: str) -> str:
    """Strip whitespace and lowercase predicate names only.

    Keeps variable names (x, y, z) and entity names (anne, bob …) as-is
    because they are already lower-cased in the dataset.
    """
    # Lowercase the predicate-name tokens (capitalised words)
    def _lower_pred(m: re.Match) -> str:
        return m.group(1).lower()

    return _PREDICATE_RE.sub(_lower_pred, line).strip()


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------

def score_premises_fol(
    ref_lines: list[str], hyp_lines: list[str]
) -> tuple[float, dict[str, str]]:
    """Compute Metric 2a and return (score, predicate_mapping).

    score = fraction of reference lines that have a matching hypothesis line
    after predicate substitution (order-independent, sorted comparison).
    """
    if not ref_lines:
        return 1.0, {}

    # Collect all predicate names from both sides
    ref_preds = [p for ln in ref_lines for p in extract_predicates(ln)]
    hyp_preds = [p for ln in hyp_lines for p in extract_predicates(ln)]
    mapping = build_predicate_mapping(ref_preds, hyp_preds)

    # Normalise + apply mapping to both sets
    norm_ref = sorted(normalise_line(ln) for ln in ref_lines)
    norm_hyp = sorted(normalise_line(apply_mapping(ln, mapping)) for ln in hyp_lines)

    # Count how many reference lines appear in hypothesis
    hyp_multiset: dict[str, int] = defaultdict(int)
    for ln in norm_hyp:
        hyp_multiset[ln] += 1

    matched = 0
    for ln in norm_ref:
        if hyp_multiset[ln] > 0:
            matched += 1
            hyp_multiset[ln] -= 1

    return matched / len(norm_ref), mapping


def score_proof(
    ref_lines: list[str], hyp_lines: list[str], mapping: dict[str, str]
) -> float:
    """Compute Metric 2b.

    score = fraction of reference proof steps covered by the longest common
    subsequence (LCS) with the hypothesis. Extra steps in the hypothesis are
    tolerated as long as the matched steps appear in the same relative order.

    Examples:
        GOLD [A, B],    PRED [X, A, B]  -> LCS=2, score=1.0  (extra step ok)
        GOLD [A, B],    PRED [A]        -> LCS=1, score=0.5   (missing step)
        GOLD [A, B],    PRED [B, A]     -> LCS=1, score=0.5   (wrong order)
        GOLD [A, B, C], PRED [A, C, B]  -> LCS=2, score=0.67  (partial order)
    """
    if not ref_lines:
        return 1.0

    norm_ref = [normalise_line(ln) for ln in ref_lines]
    norm_hyp = [normalise_line(apply_mapping(ln, mapping)) for ln in hyp_lines]

    # LCS via DP
    m, n = len(norm_ref), len(norm_hyp)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if norm_ref[i - 1] == norm_hyp[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    return dp[m][n] / len(ref_lines)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model_and_tokenizer(
    checkpoint_path: str, config_path: str, device: torch.device,
    cls_head_checkpoint: str | None = None,
):
    """Load a FOLModel from checkpoint and its matching tokenizer."""
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["encoder_name"])
    model = FOLModel(cfg["model"], vocab_size=tokenizer.vocab_size)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])

    if cls_head_checkpoint is not None:
        cls_ckpt = torch.load(cls_head_checkpoint, map_location=device, weights_only=False)
        cls_state = {
            k.replace("answer_cls_head.", ""): v
            for k, v in cls_ckpt["model_state"].items()
            if k.startswith("answer_cls_head.")
        }
        model.answer_cls_head.load_state_dict(cls_state)
        print(f"Loaded cls head weights from: {cls_head_checkpoint}")

    model = model.to(device)
    model.eval()
    return model, tokenizer, cfg


# ---------------------------------------------------------------------------
# Cls head inference
# ---------------------------------------------------------------------------

IDX_TO_ANS = {0: "true", 1: "false", 2: "unknown"}


def predict_with_cls_head(
    model, tokenizer, premises: str, raw_output: str,
    device: torch.device, max_input_len: int, extra_id_4_token: int,
) -> str | None:
    """Teacher-force decoder up to <extra_id_4> and classify with cls head.

    Returns predicted answer string ("true"/"false"/"unknown") or None if
    the <extra_id_4> sentinel is not present in the generated output.
    """
    enc = tokenizer(
        premises, max_length=max_input_len, truncation=True, return_tensors="pt"
    )
    input_ids      = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    target = tokenizer(
        raw_output, max_length=512, truncation=True, return_tensors="pt"
    )
    target_ids = target["input_ids"].to(device)  # (1, T)

    matches = (target_ids[0] == extra_id_4_token).nonzero(as_tuple=True)[0]
    if len(matches) == 0:
        return None

    cls_pos = matches[0].item()
    decoder_input = target_ids[:, : cls_pos + 1]  # (1, cls_pos+1)

    with torch.no_grad():
        encoder_out = model.encoder(input_ids, attention_mask)
        model.decoder(decoder_input, encoder_out, attention_mask)

    hidden     = model.decoder.last_hidden          # (1, T, d_model)
    cls_hidden = hidden[0, cls_pos].unsqueeze(0)    # (1, d_model)
    cls_logits = model.answer_cls_head(cls_hidden.float())
    return IDX_TO_ANS[cls_logits.argmax(dim=-1).item()]


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def evaluate(
    checkpoint_path: str,
    config_path: str,
    input_file: str,
    max_samples: int | None = None,
    output_file: str | None = None,
    max_len: int = 512,
    use_cls_head: bool = False,
    cls_head_checkpoint: str | None = None,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Config:     {config_path}")
    print(f"Input file: {input_file}")
    if use_cls_head:
        print(f"Answer mode: cls head  (checkpoint: {cls_head_checkpoint or 'from main ckpt'})")
    print()

    model, tokenizer, cfg = load_model_and_tokenizer(
        checkpoint_path, config_path, device, cls_head_checkpoint
    )
    max_input_len = cfg["data"].get("max_input_len", 512)
    extra_id_4_token = tokenizer.convert_tokens_to_ids("<extra_id_4>") if use_cls_head else None

    # Load samples
    samples = []
    with open(input_file) as f:
        for line in f:
            samples.append(json.loads(line.strip()))

    if max_samples is not None:
        samples = samples[:max_samples]

    # Per-QDep accumulators for answer accuracy
    qdep_correct: dict[int, int] = defaultdict(int)
    qdep_total: dict[int, int] = defaultdict(int)
    total_correct = 0

    # Structural metric accumulators
    prem_fol_scores: list[float] = []
    proof_scores: list[float] = []

    # Confusion matrix accumulators: [gold][pred]
    LABELS = ["true", "false", "unknown"]
    cm_count: dict[str, dict[str, int]]   = {g: {p: 0   for p in LABELS} for g in LABELS}
    cm_prem:  dict[str, dict[str, list]]  = {g: {p: []  for p in LABELS} for g in LABELS}
    cm_proof: dict[str, dict[str, list]]  = {g: {p: []  for p in LABELS} for g in LABELS}

    out_fh = open(output_file, "w") if output_file else None

    for sample in tqdm(samples, desc="Evaluating", unit="sample"):
        premises = sample["premises"]
        gold_answer = sample.get("answer", "").strip().lower()
        qdep = sample.get("qdep", -1)
        gold_logic = sample.get("logic", "")

        # --- Inference ---
        raw_output = generate(
            model, tokenizer, premises, device,
            max_len=max_len,
            max_input_len=max_input_len,
        )

        # --- Metric 1: answer accuracy ---
        parsed_hyp = parse_output(raw_output)
        if use_cls_head:
            cls_pred = predict_with_cls_head(
                model, tokenizer, premises, raw_output,
                device, max_input_len, extra_id_4_token,
            )
            pred_answer = cls_pred if cls_pred is not None else extract_answer_word(parsed_hyp["answer_text"])
        else:
            pred_answer = extract_answer_word(parsed_hyp["answer_text"])
        is_correct = (pred_answer == gold_answer)
        if is_correct:
            total_correct += 1
        qdep_correct[qdep] += int(is_correct)
        qdep_total[qdep] += 1

        # --- Metric 2: structural accuracy ---
        parsed_ref = parse_output(gold_logic)

        prem_score, pred_mapping = score_premises_fol(
            parsed_ref["premises_fol"], parsed_hyp["premises_fol"]
        )
        proof_score = score_proof(
            parsed_ref["proof"], parsed_hyp["proof"], pred_mapping
        )
        prem_fol_scores.append(prem_score)
        proof_scores.append(proof_score)

        # --- Confusion matrix accumulation ---
        g = gold_answer if gold_answer in LABELS else "unknown"
        p = pred_answer if pred_answer in LABELS else "unknown"
        cm_count[g][p] += 1
        cm_prem[g][p].append(prem_score)
        cm_proof[g][p].append(proof_score)

        # --- Optional per-sample output ---
        if out_fh is not None:
            record = {
                "source": sample.get("source", ""),
                "qdep": qdep,
                "premises": premises,
                "gold_answer": gold_answer,
                "pred_answer": pred_answer,
                "answer_correct": is_correct,
                "prem_fol_score": prem_score,
                "proof_score": proof_score,
                "raw_output": raw_output,
            }
            out_fh.write(json.dumps(record) + "\n")

    if out_fh is not None:
        out_fh.close()

    n = len(samples)

    # --- Print summary ---
    print()
    print("=== EVALUATION RESULTS ===")
    print(f"Samples evaluated: {n}")
    print()
    print("Metric 1 — Answer Accuracy")
    overall_acc = total_correct / n * 100 if n > 0 else 0.0
    print(f"  Overall: {overall_acc:.1f}%")
    for qdep_val in sorted(qdep_total.keys()):
        cnt = qdep_total[qdep_val]
        acc = qdep_correct[qdep_val] / cnt * 100 if cnt > 0 else 0.0
        label = f"QDep {qdep_val}" if qdep_val >= 0 else "QDep unknown"
        print(f"  {label}:  {acc:.1f}% ({cnt} samples)")
    print()
    print("Metric 2a — Premises-FOL Match")
    prem_avg = sum(prem_fol_scores) / len(prem_fol_scores) * 100 if prem_fol_scores else 0.0
    print(f"  Overall: {prem_avg:.1f}%")
    print()
    print("Metric 2b — Proof Match")
    proof_avg = sum(proof_scores) / len(proof_scores) * 100 if proof_scores else 0.0
    print(f"  Overall: {proof_avg:.1f}%")
    print()

    # --- Confusion matrices ---
    col_w = 10
    header = f"{'':10}" + "".join(f"{'pred:'+lbl:>{col_w}}" for lbl in LABELS)

    def _avg(vals): return f"{sum(vals)/len(vals)*100:.1f}%" if vals else "  -"

    print("Confusion Matrix 1 — Answer Counts (rows=gold, cols=pred)")
    print(header)
    for g in LABELS:
        row = f"{'gold:'+g:10}" + "".join(f"{cm_count[g][p]:>{col_w}}" for p in LABELS)
        print(row)
    print()

    print("Confusion Matrix 2 — Premises-FOL Avg Score (rows=gold, cols=pred)")
    print(header)
    for g in LABELS:
        row = f"{'gold:'+g:10}" + "".join(f"{_avg(cm_prem[g][p]):>{col_w}}" for p in LABELS)
        print(row)
    print()

    print("Confusion Matrix 3 — Proof Avg Score (rows=gold, cols=pred)")
    print(header)
    for g in LABELS:
        row = f"{'gold:'+g:10}" + "".join(f"{_avg(cm_proof[g][p]):>{col_w}}" for p in LABELS)
        print(row)
    print()

    return {
        "n_samples": n,
        "answer_accuracy": overall_acc,
        "prem_fol_match": prem_avg,
        "proof_match": proof_avg,
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained FOL SLM checkpoint on a test set."
    )
    parser.add_argument(
        "--checkpoint",
        default="outputs/final_v8/checkpoint_final.pt",
        help="Path to .pt checkpoint file (default: outputs/final/checkpoint_final.pt)",
    )
    parser.add_argument(
        "--config",
        default="configs/final.yaml",
        help="Path to YAML config file (default: configs/final.yaml)",
    )
    parser.add_argument(
        "--input-file",
        default="data/processed/test.jsonl",
        help="JSONL test file (default: data/processed/test.jsonl)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Limit evaluation to first N samples (useful for quick testing)",
    )
    parser.add_argument(
        "--output-file",
        default=None,
        help="Optional path to save per-sample results as JSONL",
    )
    parser.add_argument(
        "--max-len",
        type=int,
        default=512,
        help="Maximum generation length (default: 512)",
    )
    parser.add_argument(
        "--use-cls-head",
        action="store_true",
        help="Use classification head for answer prediction instead of greedy decoding",
    )
    parser.add_argument(
        "--cls-head-checkpoint",
        default=None,
        help="Path to cls head checkpoint to load answer_cls_head weights from",
    )
    args = parser.parse_args()

    evaluate(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        input_file=args.input_file,
        max_samples=args.max_samples,
        output_file=args.output_file,
        max_len=args.max_len,
        use_cls_head=args.use_cls_head,
        cls_head_checkpoint=args.cls_head_checkpoint,
    )


if __name__ == "__main__":
    main()
