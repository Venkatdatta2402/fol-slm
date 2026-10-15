"""Shared FOL parsing and evaluation scoring utilities.

Used by evaluate.py, analyze.py, and inspect_data.py.
"""

import re
from collections import defaultdict

import nltk
nltk.download("wordnet", quiet=True)
from nltk.corpus import wordnet


# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------

_FOL_ATOM_RE = re.compile(
    r'forall\s+x\s+\([^()]*(?:\([^()]*\)[^()]*)*\)'           # universal rule
    r'|(?:not\s+)?[A-Z][a-zA-Z]*\([^)]+\)'                    # atom (pos or neg)
    r'(?:\s*->\s*(?:not\s+)?[A-Z][a-zA-Z]*\([^)]+\))?'       # optional -> conclusion
)

_PROOF_STEP_RE = re.compile(
    r'therefore\s+(?:not\s+)?[A-Z][a-zA-Z]*\([^)]+\)'
)

_PREDICATE_RE = re.compile(r"\b([A-Z][a-z]*)\b")


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def _extract_raw(text: str, start_token: str, end_token: str) -> str:
    """Return raw text between start_token and end_token."""
    if start_token not in text or end_token not in text:
        return ""
    return text.split(start_token, 1)[1].split(end_token, 1)[0]


def _split_fol_statements(text: str) -> list[str]:
    """Extract individual FOL statements from text."""
    return [m.strip() for m in _FOL_ATOM_RE.findall(text) if m.strip()]


def _split_proof_steps(text: str) -> list[str]:
    """Extract proof steps (therefore ...) from text."""
    return [m.strip() for m in _PROOF_STEP_RE.findall(text) if m.strip()]


def parse_output(text: str) -> dict:
    """Split a model output (or ground-truth logic string) into its parts.

    Returns:
        {
            "premises_fol": [str, ...],
            "question_fol": str,
            "proof":        [str, ...],
            "answer_text":  str,
        }
    """
    prem_raw     = _extract_raw(text, "<extra_id_1>", "<extra_id_2>")
    question_raw = _extract_raw(text, "<extra_id_2>", "<extra_id_3>")
    proof_raw    = _extract_raw(text, "<extra_id_3>", "<extra_id_4>")

    answer_text = ""
    if "<extra_id_4>" in text:
        answer_text = text.split("<extra_id_4>", 1)[1].strip()

    return {
        "premises_fol": _split_fol_statements(prem_raw),
        "question_fol": question_raw.strip(),
        "proof":        _split_proof_steps(proof_raw),
        "answer_text":  answer_text,
    }


def extract_answer_word(answer_text: str) -> str:
    """Return the first word of the answer (True/False/Unknown), lowercased."""
    if not answer_text:
        return ""
    return answer_text.strip().split()[0].lower().rstrip(".,;:")


# ---------------------------------------------------------------------------
# Predicate normalisation
# ---------------------------------------------------------------------------

def extract_predicates(line: str) -> list[str]:
    return _PREDICATE_RE.findall(line)


def _synsets(word: str) -> set[str]:
    return {s.name() for s in wordnet.synsets(word.lower())}


def _wordnet_equivalent(p1: str, p2: str) -> bool:
    s1 = _synsets(p1)
    s2 = _synsets(p2)
    return bool(s1 and s2 and s1 & s2)


def _edit_distance(a: str, b: str) -> int:
    a, b = a.lower(), b.lower()
    if a == b:
        return 0
    m, n = len(a), len(b)
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
    """Map hypothesis predicates to reference predicates via WordNet / edit-distance."""
    ref_set = set(pred_list_ref)
    hyp_set = set(pred_list_hyp)
    new_in_hyp = hyp_set - ref_set
    only_in_ref = ref_set - hyp_set

    mapping: dict[str, str] = {}
    for hp in new_in_hyp:
        matched = None
        for rp in only_in_ref:
            if _wordnet_equivalent(hp, rp):
                matched = rp
                break
        if matched is None:
            candidates = [(rp, _edit_distance(hp, rp)) for rp in only_in_ref]
            candidates.sort(key=lambda x: x[1])
            if candidates and candidates[0][1] <= 2:
                matched = candidates[0][0]
        if matched is not None:
            mapping[hp] = matched

    return mapping


def apply_mapping(line: str, mapping: dict[str, str]) -> str:
    """Replace hypothesis predicates with their mapped reference predicates."""
    if not mapping:
        return line

    def _replace(m: re.Match) -> str:
        return mapping.get(m.group(1), m.group(1))

    return _PREDICATE_RE.sub(_replace, line)


def normalise_line(line: str) -> str:
    """Lowercase predicate names only."""
    return _PREDICATE_RE.sub(lambda m: m.group(1).lower(), line).strip()


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def score_premises_fol(
    ref_lines: list[str], hyp_lines: list[str]
) -> tuple[float, dict[str, str]]:
    """Metric 2a: fraction of reference FOL premises matched by hypothesis.

    Returns (score, predicate_mapping).
    """
    if not ref_lines:
        return 1.0, {}

    ref_preds = [p for ln in ref_lines for p in extract_predicates(ln)]
    hyp_preds = [p for ln in hyp_lines for p in extract_predicates(ln)]
    mapping = build_predicate_mapping(ref_preds, hyp_preds)

    norm_ref = sorted(normalise_line(ln) for ln in ref_lines)
    norm_hyp = sorted(normalise_line(apply_mapping(ln, mapping)) for ln in hyp_lines)

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
    """Metric 2b: LCS-based proof step match, normalised by gold length."""
    if not ref_lines:
        return 1.0

    norm_ref = [normalise_line(ln) for ln in ref_lines]
    norm_hyp = [normalise_line(apply_mapping(ln, mapping)) for ln in hyp_lines]

    m, n = len(norm_ref), len(norm_hyp)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if norm_ref[i - 1] == norm_hyp[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    return dp[m][n] / len(ref_lines)
