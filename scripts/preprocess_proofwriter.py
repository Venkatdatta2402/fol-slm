#!/usr/bin/env python3
"""
Preprocess ProofWriter (OWA) dataset into FOL CoT training format.

Reads:  data/proofwriter/{depth-2,depth-3,depth-3ext}/meta-{train,dev,test}.jsonl
Writes: data/processed/{train,dev,test}.jsonl

Output schema (matches ReasoningDataset):
    {"premises": str, "logic": str, "qdep": int, "answer": str, "source": str}

    premises  — NL theory string (encoder input)
    logic     — FOL premises + derivation steps + NL answer (decoder target)
    qdep      — reasoning depth (useful for filtering in pilots/curriculum)
    answer    — "True" / "False" / "Unknown"
    source    — original record id

Usage:
    python scripts/preprocess_proofwriter.py                  # all depths, all splits
    python scripts/preprocess_proofwriter.py --pilot          # QDep=1 only, train split
    python scripts/preprocess_proofwriter.py --min-qdep 1     # exclude trivial QDep=0
"""

import argparse
import json
import random
import re
from pathlib import Path

# ── Constants ────────────────────────────────────────────────────────────────

UNIVERSAL_SUBJECTS = {"someone", "something"}

# ── Name / predicate pools for random substitution ───────────────────────────
# ProofWriter's fixed vocabulary — excluded from all pools to prevent leakage.
_PW_ENTITIES    = {"anne","bob","charlie","dave","erin","fiona","gary","harry",
                   "bear","cat","cow","dog","lion","mouse","rabbit","squirrel","tiger","wolf"}
_PW_PROPERTIES  = {"kind","cold","rough","blue","green","white","red","young","round",
                   "quiet","furry","smart","big","nice","black","brown","dark","fast","good"}
_PW_RELATIONS   = {"visits","chases","sees","eats","needs","likes"}


def _build_pools() -> tuple[list, list, list]:
    """Build large substitution pools from NLTK corpora (downloaded on first call).

    Returns:
        (entity_pool, property_pool, relation_pool) — each a sorted list of
        lowercase strings that do NOT overlap with ProofWriter's fixed vocab.

    Pool sizes (approximate, after exclusions):
        entities   ~7 000  (NLTK names corpus)
        properties ~13 000 (WordNet adjectives, 4–10 chars, single-word lemmas)
        relations  ~7 400  (WordNet verbs,      4–10 chars, single-word lemmas)
    """
    import nltk
    for pkg in ("names", "wordnet", "omw-1.4"):
        nltk.download(pkg, quiet=True)
    from nltk.corpus import names as _names
    from nltk.corpus import wordnet as wn

    entities = {
        n.lower() for n in _names.words()
        if n.isalpha() and 3 <= len(n) <= 9
    } - _PW_ENTITIES

    properties = {
        lemma.name().lower()
        for syn in wn.all_synsets(pos="a")
        for lemma in syn.lemmas()
        if lemma.name().isalpha() and 4 <= len(lemma.name()) <= 10
    } - _PW_PROPERTIES

    relations = {
        lemma.name().lower()
        for syn in wn.all_synsets(pos="v")
        for lemma in syn.lemmas()
        if lemma.name().isalpha() and 4 <= len(lemma.name()) <= 10
    } - _PW_RELATIONS

    return sorted(entities), sorted(properties), sorted(relations)


ENTITY_POOL, PROPERTY_POOL, RELATION_POOL = _build_pools()


# ── Per-question substitution map ─────────────────────────────────────────────

def _collect_vocab(record: dict) -> tuple[set, set, set]:
    """Return (entities, properties, relations) found in a record's triples+rules."""
    entities   = set()
    properties = set()
    relations  = set()
    for triple in record["triples"].values():
        toks = parse_tokens(triple["representation"])
        if len(toks) == 4:
            subj, verb, obj_prop, _ = toks
            if subj.lower() not in UNIVERSAL_SUBJECTS:
                entities.add(subj.lower())
            if verb == "is":
                properties.add(obj_prop.lower())
            else:
                relations.add(verb.lower())
                if obj_prop.lower() not in UNIVERSAL_SUBJECTS:
                    entities.add(obj_prop.lower())
    for rule in record["rules"].values():
        # rules don't introduce new entities beyond triples, but may reference them
        rep = rule["representation"]
        for tok_group in re.findall(r'\("[^)]+\)', rep):
            toks = parse_tokens(tok_group)
            if len(toks) == 4:
                subj, verb, obj_prop, _ = toks
                if subj.lower() not in UNIVERSAL_SUBJECTS:
                    entities.add(subj.lower())
                if verb == "is":
                    properties.add(obj_prop.lower())
                else:
                    relations.add(verb.lower())
                    if obj_prop.lower() not in UNIVERSAL_SUBJECTS:
                        entities.add(obj_prop.lower())
    return entities, properties, relations


def make_subst_map(record: dict, rng: random.Random) -> dict:
    """Build a fresh random substitution map for one question (called per question).

    Returns:
        {
          "entities":   {original_lower: replacement_lower},
          "properties": {original_lower: replacement_lower},
          "relations":  {original_lower: replacement_lower},
        }
    """
    entities, properties, relations = _collect_vocab(record)
    subst_entities   = {e: rng.choice(ENTITY_POOL)   for e in sorted(entities)}
    subst_properties = {p: rng.choice(PROPERTY_POOL) for p in sorted(properties)}
    subst_relations  = {r: rng.choice(RELATION_POOL) for r in sorted(relations)}
    return {"entities": subst_entities, "properties": subst_properties, "relations": subst_relations}


def apply_nl_substitution(text: str, subst: dict) -> str:
    """Apply substitution map to a NL string using whole-word regex replacement.

    Entities are title-cased in NL (Anne → Jessica).
    Properties and relations stay lowercase.
    Longest originals replaced first to avoid partial overlaps.
    """
    replacements = {}
    for orig, repl in subst["entities"].items():
        # NL uses capitalised entity names; multi-word joined by space
        nl_orig = orig.replace("_", " ").title()
        replacements[nl_orig] = repl.title()
        replacements[orig]    = repl          # lowercase fallback
    for orig, repl in subst["properties"].items():
        replacements[orig] = repl
    for orig, repl in subst["relations"].items():
        replacements[orig] = repl

    # Sort by length descending so "bald eagle" is replaced before "eagle"
    for orig in sorted(replacements, key=len, reverse=True):
        text = re.sub(r'\b' + re.escape(orig) + r'\b', replacements[orig], text)
    return text

DEPTHS = ["depth-2", "depth-3", "depth-3ext"]

# ── Entity / Predicate Normalisation ────────────────────────────────────────

def normalize_entity(name: str) -> str:
    """'bald eagle' → 'bald_eagle',  'Anne' → 'anne'"""
    return name.lower().replace(" ", "_")


def normalize_predicate(name: str) -> str:
    """'kind' → 'Kind',  'bald eagle' → 'BaldEagle',  'visits' → 'Visits'"""
    return "".join(part.capitalize() for part in name.split())


# ── Atom-level FOL translation ───────────────────────────────────────────────

def parse_tokens(rep: str) -> list:
    """Extract all double-quoted tokens from a representation string."""
    return re.findall(r'"([^"]*)"', rep)


def atom_to_fol(tokens: list, var: str = "x", subst: dict = None) -> str:
    """
    Convert a 4-token atom [subject, verb, obj/prop, polarity] to a FOL string.

    Polarity:
        '+'  → positive
        '-'  → negated (stated false fact or negated rule conclusion)
        '~'  → negated condition inside a rule antecedent

    Examples:
        ["Anne",    "is",     "kind",      "+"]  → Kind(anne)
        ["Anne",    "is",     "kind",      "-"]  → ¬Kind(anne)
        ["someone", "is",     "young",     "+"]  → Young(x)
        ["bear",    "needs",  "mouse",     "+"]  → Needs(bear, mouse)
        ["someone", "needs",  "dog",       "~"]  → ¬Needs(x, dog)
        ["bald eagle","is",   "rough",     "+"]  → Rough(bald_eagle)
    """
    if len(tokens) != 4:
        return str(tokens)

    subject, verb, obj_or_prop, polarity = tokens
    negated = polarity in ("-", "~")
    neg = "not " if negated else ""

    # Apply substitution before normalisation
    if subst:
        subj_key = subject.lower()
        if subj_key in subst["entities"]:
            subject = subst["entities"][subj_key]
        if verb == "is":
            prop_key = obj_or_prop.lower()
            if prop_key in subst["properties"]:
                obj_or_prop = subst["properties"][prop_key]
        else:
            rel_key = verb.lower()
            if rel_key in subst["relations"]:
                verb = subst["relations"][rel_key]
            obj_key = obj_or_prop.lower()
            if obj_key in subst["entities"]:
                obj_or_prop = subst["entities"][obj_key]

    subj_fol = var if subject.lower() in UNIVERSAL_SUBJECTS else normalize_entity(subject)

    if verb == "is":
        pred = normalize_predicate(obj_or_prop)
        return f"{neg}{pred}({subj_fol})"
    else:
        pred = normalize_predicate(verb)
        obj_fol = normalize_entity(obj_or_prop)
        return f"{neg}{pred}({subj_fol}, {obj_fol})"


def triple_to_fol(rep: str, subst: dict = None) -> str:
    """Convert a triple representation string to a FOL atom string."""
    tokens = parse_tokens(rep)
    if len(tokens) != 4:
        return rep  # fallback: return raw
    return atom_to_fol(tokens, subst=subst)


# ── Rule-level FOL translation ───────────────────────────────────────────────

def rule_to_fol(rep: str, subst: dict = None) -> str:
    """
    Convert a rule representation to a FOL implication string.

    Input format:  ((conditions) -> conclusion)
        conditions — one or more atoms: ("X" "v" "Y" "p") ...
        conclusion — single atom: ("X" "v" "Y" "p")

    Handles:
        - single-condition rules
        - multi-condition rules (conjunction)
        - negated conditions ('~')
        - negated conclusions ('-')
        - universal rules ('someone'/'something' → ∀x)
        - specific-entity rules (named subject, no quantifier)
    """
    rep = rep.strip()

    arrow_idx = rep.find(" -> ")
    if arrow_idx == -1:
        return rep  # fallback

    conditions_str = rep[:arrow_idx]
    conclusion_str = rep[arrow_idx + 4:].strip()
    # Remove the trailing ')' that closes the outer rule wrapper
    if conclusion_str.endswith(")"):
        conclusion_str = conclusion_str[:-1]

    # Extract individual condition atoms: patterns like ("..." "..." "..." "...")
    condition_atoms = re.findall(r'\("[^)]+\)', conditions_str)
    conclusion_tokens = parse_tokens(conclusion_str)

    # Determine whether this is a universal rule
    all_subjects = [
        parse_tokens(a)[0] for a in condition_atoms if parse_tokens(a)
    ]
    if conclusion_tokens:
        all_subjects.append(conclusion_tokens[0])
    is_universal = any(s in UNIVERSAL_SUBJECTS for s in all_subjects)

    var = "x"

    # Build condition FOL strings
    cond_fols = []
    for atom in condition_atoms:
        tokens = parse_tokens(atom)
        if len(tokens) == 4:
            cond_fols.append(atom_to_fol(tokens, var, subst=subst))

    # Build conclusion FOL string
    if len(conclusion_tokens) == 4:
        concl_fol = atom_to_fol(conclusion_tokens, var, subst=subst)
    else:
        concl_fol = conclusion_str  # fallback

    antecedent = " and ".join(cond_fols) if cond_fols else "?"
    implication = f"{antecedent} -> {concl_fol}"

    return f"forall x ({implication})" if is_universal else implication


# ── Proof Step Extraction ────────────────────────────────────────────────────

def _int_num(key: str) -> int:
    m = re.match(r"int(\d+)", key)
    return int(m.group(1)) if m else 0


# ── Proof Tree Parser ────────────────────────────────────────────────────────

class _ProofParser:
    """Parse ProofWriter proof representation into ordered derivation steps.

    Each step is (input_fols: list[str], rule_fol: str, output_fol: str).

    The proof representation is a nested expression of the form:
        (INPUTS -> (ruleN % intM))
    where INPUTS can be identifiers (tripleK, intK) or nested expressions.

    Example:
        ((((triple1) -> (rule5 % int2))) -> (rule2 % int1))
    yields:
        Step 1: [Kind(anne)], forall x (Kind(x) -> Furry(x)) -> Furry(anne)
        Step 2: [Furry(anne)], forall x (Furry(x) -> Green(x)) -> Green(anne)
    """

    def __init__(self, proof_rep: str, triples: dict, rules: dict, intermediates: dict,
                 subst: dict = None):
        self.tokens = re.findall(r'\(|\)|->|%|[A-Za-z]\w*', proof_rep)
        self.pos = 0
        self.triples = triples
        self.rules = rules
        self.intermediates = intermediates
        self.subst = subst
        self.steps: list[tuple[list[str], str, str]] = []

    def _peek(self):
        return self.tokens[self.pos] if self.pos < len(self.tokens) else None

    def _consume(self):
        tok = self.tokens[self.pos]
        self.pos += 1
        return tok

    def _fol_of(self, id_str: str) -> str:
        if id_str.startswith("triple"):
            return triple_to_fol(self.triples[id_str]["representation"], subst=self.subst)
        if id_str.startswith("int"):
            return triple_to_fol(self.intermediates[id_str]["representation"], subst=self.subst)
        if id_str.startswith("rule"):
            return rule_to_fol(self.rules[id_str]["representation"], subst=self.subst)
        return id_str

    def _flatten(self, items) -> list[str]:
        result = []
        for item in items:
            if isinstance(item, str):
                result.append(item)
            elif isinstance(item, list):
                result.extend(self._flatten(item))
        return result

    def parse(self):
        """Parse an expression; return resolved output id (str) or list of ids."""
        if self._peek() == '(':
            return self._parse_paren()
        return self._consume()

    def _parse_paren(self):
        self._consume()  # '('
        operands = []
        while self._peek() and self._peek() not in ('->', '%', ')'):
            operands.append(self.parse())

        if self._peek() == '->':
            self._consume()  # '->'
            right = self.parse()
            self._consume()  # ')'
            if isinstance(right, tuple) and right[0] == 'prod':
                _, rule_id, int_id = right
                input_ids = self._flatten(operands)
                input_fols = [self._fol_of(i) for i in input_ids]
                rule_fol = self._fol_of(rule_id)
                output_fol = self._fol_of(int_id)
                self.steps.append((input_fols, rule_fol, output_fol))
                return int_id
            flat = self._flatten(operands)
            return flat[0] if len(flat) == 1 else flat

        if self._peek() == '%':
            rule_id = self._flatten(operands)[0]
            self._consume()  # '%'
            int_id = self._consume()
            self._consume()  # ')'
            return ('prod', rule_id, int_id)

        # bare grouping
        self._consume()  # ')'
        flat = self._flatten(operands)
        return flat[0] if len(flat) == 1 else flat


def extract_steps(proof_with_intermediates: dict,
                  triples: dict | None = None,
                  rules: dict | None = None,
                  subst: dict = None) -> list:
    """Return ordered proof steps as (input_fols, rule_fol, output_fol) triples.

    If triples/rules are provided the full derivation structure is extracted from
    the proof representation tree.  Otherwise falls back to conclusions-only ordering.
    """
    intermediates = proof_with_intermediates.get("intermediates") or {}
    if not intermediates:
        return []

    proof_rep = proof_with_intermediates.get("representation", "")

    if triples is not None and rules is not None and proof_rep:
        try:
            parser = _ProofParser(proof_rep, triples, rules, intermediates, subst=subst)
            parser.parse()
            if parser.steps:
                return parser.steps
        except Exception:
            pass  # fall through to conclusions-only on any parse error

    # Fallback: conclusions only (no inputs/rule), sorted by int number
    ordered = sorted(intermediates.items(), key=lambda kv: _int_num(kv[0]), reverse=True)
    return [([], "", triple_to_fol(val["representation"], subst=subst)) for _, val in ordered]


# ── Unknown Failure Trace ─────────────────────────────────────────────────────

def parse_unknown_proof(proofs_str: str, rules: dict, question_fol: str,
                        subst: dict = None) -> str:
    """Translate a ProofWriter failure trace string into a FOL proof section.

    Input format:
        "@DEPTH: Fact text.[CWA. Example of deepest failure = (rule1 <- rule2 <- FAIL)]"

    Output: FOL rule chain showing which rules were tried and why they failed,
    followed by "Cannot be determined from given premises."

    Examples:
        FAIL                    → [no supporting fact for question_fol]
        rule1 <- FAIL           → rule1_fol <- [no base fact]
        rule1 <- rule2 <- FAIL  → rule1_fol <- rule2_fol <- [no base fact]
    """
    m = re.search(r'Example of deepest failure = \(([^)]*)\)', proofs_str)
    if not m:
        return f"{question_fol} <- [no supporting fact or rule] <extra_id_5> Cannot be determined from given premises."

    chain_str = m.group(1).strip()  # e.g. "rule1 <- rule2 <- FAIL"
    parts = [p.strip() for p in chain_str.split("<-")]
    rule_ids = [p for p in parts if p.startswith("rule")]

    if not rule_ids:
        # Pure FAIL: no rules attempted
        return f"{question_fol} <- [no supporting fact or rule] <extra_id_5> Cannot be determined from given premises."

    # Translate each rule id to FOL
    rule_fols = []
    for rid in rule_ids:
        if rid in rules:
            rule_fols.append(rule_to_fol(rules[rid]["representation"], subst=subst))
        else:
            rule_fols.append(rid)

    # Chain: outermost rule first, then the rules it depended on, ending at FAIL
    chain = " <- ".join(rule_fols) + " <- [no base fact]"
    return f"{chain} <extra_id_5> Cannot be determined from given premises."


# ── Record → Training Example ────────────────────────────────────────────────

def process_question(record: dict, question: dict, rng: random.Random) -> dict | None:
    """
    Convert one (record, question) pair into a training example.

    A fresh random substitution map is generated per question so entity and
    predicate names vary across every training example, forcing the model to
    learn structural NL→FOL mapping rather than memorising ProofWriter names.
    """
    answer = question.get("answer")
    question_text = question.get("question", "")
    qdep = question.get("QDep", 0)

    # ── Per-question substitution map ──────────────────────────────────────
    subst = make_subst_map(record, rng)

    # ── NL premises (encoder input) with substitution applied ──────────────
    nl_premises = apply_nl_substitution(
        record["theory"] + " <extra_id_0> " + question_text, subst
    )

    # ── FOL premises block ──────────────────────────────────────────────────
    fol_lines = []
    for t in record["triples"].values():
        fol_lines.append(triple_to_fol(t["representation"], subst=subst))
    for r in record["rules"].values():
        fol_lines.append(rule_to_fol(r["representation"], subst=subst))
    premises_fol = "\n".join(fol_lines)

    # ── FOL question ────────────────────────────────────────────────────────
    q_rep = question.get("representation", "")
    question_fol = triple_to_fol(q_rep, subst=subst) if q_rep else question_text

    # ── Proof / derivation block ────────────────────────────────────────────
    pwi_list = question.get("proofsWithIntermediates") or []

    if answer == "Unknown":
        proof_section = parse_unknown_proof(
            question.get("proofs", ""),
            record.get("rules", {}),
            question_fol,
            subst=subst,
        )
        answer_label = "Unknown"
    elif not pwi_list:
        proof_section = "Cannot be determined from given premises."
        answer_label = "Unknown"
    else:
        pwi = pwi_list[0]
        steps = extract_steps(pwi, triples=record.get("triples"),
                              rules=record.get("rules"), subst=subst)

        if steps:
            proof_lines = []
            for input_fols, rule_fol, output_fol in steps:
                if input_fols and rule_fol:
                    inputs_str = " and ".join(input_fols + [rule_fol])
                    proof_lines.append(f"{inputs_str} -> therefore {output_fol}")
                else:
                    proof_lines.append(f"therefore {output_fol}")
            proof_section = " <extra_id_5> ".join(proof_lines)
        else:
            proof_rep = pwi.get("representation", "").strip()
            triples = record.get("triples", {})
            rules   = record.get("rules", {})
            if proof_rep in triples:
                fact_fol = triple_to_fol(triples[proof_rep]["representation"], subst=subst)
            elif proof_rep in rules:
                fact_fol = rule_to_fol(rules[proof_rep]["representation"], subst=subst)
            else:
                fact_fol = question_fol
            proof_section = f"therefore {fact_fol}"

        answer_label = "True" if answer is True else "False"

    logic = (
        f"<extra_id_1>\n{premises_fol}\n"
        f"<extra_id_2>\n{question_fol}\n"
        f"<extra_id_3>\n{proof_section}\n"
        f"<extra_id_4>\n{answer_label}"
    )

    return {
        "premises": nl_premises,   # substituted NL (encoder input)
        "logic": logic,
        "qdep": qdep,
        "answer": answer_label,
        "source": record["id"],
    }


def process_file(filepath: Path, min_qdep: int = 0, max_qdep: int = 99,
                 rng: random.Random = None) -> list:
    if rng is None:
        rng = random.Random()
    examples = []
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            for q in record.get("questions", {}).values():
                qdep = q.get("QDep", 0)
                if not (min_qdep <= qdep <= max_qdep):
                    continue
                ex = process_question(record, q, rng)
                if ex:
                    examples.append(ex)
    return examples


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Preprocess ProofWriter → FOL CoT format")
    parser.add_argument("--pilot", action="store_true",
                        help="Output only QDep=1 examples from train split (for sweep piloting)")
    parser.add_argument("--min-qdep", type=int, default=0,
                        help="Minimum QDep to include (default: 0)")
    parser.add_argument("--max-qdep", type=int, default=99,
                        help="Maximum QDep to include (default: unbounded)")
    parser.add_argument("--depths", nargs="+", default=DEPTHS,
                        help=f"Depths to include (default: {DEPTHS})")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for name substitution (default: 42)")
    parser.add_argument("--no-subst", action="store_true",
                        help="Disable random name substitution (original ProofWriter names)")
    args = parser.parse_args()

    if args.pilot:
        args.min_qdep = 1
        args.max_qdep = 1

    rng = None if args.no_subst else random.Random(args.seed)

    base = Path("data/proofwriter")
    out_dir = Path("data/processed")
    out_dir.mkdir(parents=True, exist_ok=True)

    splits_data = {"train": [], "dev": [], "test": []}
    active_splits = ["train"] if args.pilot else ["train", "dev", "test"]

    subst_label = "disabled" if args.no_subst else f"seed={args.seed}"
    print(f"Name substitution: {subst_label}")

    for depth in args.depths:
        for split in active_splits:
            fpath = base / depth / f"meta-{split}.jsonl"
            if not fpath.exists():
                print(f"  [skip] {fpath} not found")
                continue
            examples = process_file(fpath, args.min_qdep, args.max_qdep, rng=rng)
            splits_data[split].extend(examples)
            print(f"  {depth}/meta-{split}: {len(examples):>7,} examples")

    print()
    for split, examples in splits_data.items():
        if not examples:
            continue
        # Pilot mode writes to a separate file to avoid overwriting the full dataset
        if args.pilot:
            out_path = out_dir / f"pilot_{split}.jsonl"
        else:
            out_path = out_dir / f"{split}.jsonl"
        with open(out_path, "w") as f:
            for ex in examples:
                f.write(json.dumps(ex) + "\n")
        # Answer distribution
        from collections import Counter
        dist = Counter(ex["answer"] for ex in examples)
        print(f"  → {out_path}: {len(examples):,} examples  {dict(dist)}")


if __name__ == "__main__":
    main()