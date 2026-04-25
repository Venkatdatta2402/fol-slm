"""Symbolic reasoner: FOL → ASP (clingo) with provenance tracking.

Takes model output (regex-parsed FOL) and runs clingo to get:
  - answer  (True / False / Unknown)
  - proof   (derived_by atoms reconstructed into human-readable steps)

Usage:
    # Test on eval output JSONL
    python scripts/reason.py --eval-file outputs/eval_v13_final_200.jsonl --max-samples 50
    # Full pipeline with model generation (future)
    python scripts/reason.py --input-file data/processed/test.jsonl --checkpoint ...
"""

import re
import sys
import json
import argparse
from pathlib import Path
from typing import Optional

import clingo

sys.path.append(str(Path(__file__).resolve().parents[1]))

# ---------------------------------------------------------------------------
# FOL → ASP conversion
# ---------------------------------------------------------------------------

_FORALL_RE     = re.compile(r'forall\s+x\s+\((.+)\)', re.IGNORECASE)
_IMPL_SPLIT_RE = re.compile(r'\s*->\s*')
_AND_SPLIT_RE  = re.compile(r'\s+and\s+', re.IGNORECASE)
_ATOM_RE       = re.compile(r'^(not\s+)?([A-Z][a-zA-Z]*)\(([^)]+)\)$')

# Matches one FOL statement on a line (for space-separated model output fallback)
_FOL_ATOM_RE = re.compile(
    r'forall\s+x\s+\([^()]*(?:\([^()]*\)[^()]*)*\)'
    r'|(?:not\s+)?[A-Z][a-zA-Z]*\([^)]+\)'
    r'(?:\s*->\s*(?:not\s+)?[A-Z][a-zA-Z]*\([^)]+\))?'
)


def _asp_pred(name: str) -> str:
    return name.lower()


def _asp_arg(arg: str) -> str:
    arg = arg.strip()
    return arg.upper() if arg == 'x' else arg.lower().replace(' ', '_')


def _parse_atom(text: str) -> Optional[tuple[bool, str, list[str]]]:
    """Returns (negated, predicate, args) or None."""
    m = _ATOM_RE.match(text.strip())
    if not m:
        return None
    neg  = bool(m.group(1))
    pred = m.group(2)
    args = [a.strip() for a in m.group(3).split(',')]
    return neg, pred, args


def _atom_to_asp(neg: bool, pred: str, args: list[str]) -> str:
    a = ','.join(_asp_arg(x) for x in args)
    return ('-' if neg else '') + f'{_asp_pred(pred)}({a})'


def _atom_to_human(neg: bool, pred: str, args: list[str]) -> str:
    a = ','.join(a.strip() for a in args)
    prefix = 'not ' if neg else ''
    return f'{prefix}{pred}({a})'


_ASP_ATOM_RE = re.compile(r'^(-?)([a-zA-Z_][a-zA-Z0-9_]*)\(([^)]+)\)$')


def _parse_asp_atom(atom: str) -> tuple[str, list[str]]:
    """Parse 'pred(a,b)' or '-pred(a,b)' → (full_pred_with_sign, [args])."""
    m = _ASP_ATOM_RE.match(atom.strip())
    if not m:
        return atom.strip(), []
    sign, pred, args_str = m.group(1), m.group(2), m.group(3)
    return sign + pred, [a.strip() for a in args_str.split(',')]


def _unify(goal: str, template: str) -> Optional[dict[str, str]]:
    """Try to unify a ground goal atom with a (possibly variadic) template.

    ProofWriter uses a single variable X (uppercase). Returns substitution
    {var: val} on success, None on failure.
    """
    g_pred, g_args = _parse_asp_atom(goal)
    t_pred, t_args = _parse_asp_atom(template)
    if g_pred != t_pred or len(g_args) != len(t_args):
        return None
    bindings: dict[str, str] = {}
    for g_a, t_a in zip(g_args, t_args):
        if t_a.isupper():           # variable
            if t_a in bindings and bindings[t_a] != g_a:
                return None         # conflict
            bindings[t_a] = g_a
        elif t_a != g_a:
            return None             # constant mismatch
    return bindings


def _apply_subst(atom: str, bindings: dict[str, str]) -> str:
    """Replace every variable in atom with its bound value."""
    if not bindings:
        return atom
    pred, args = _parse_asp_atom(atom)
    if not args:
        return atom
    new_args = [bindings.get(a, a) if a.isupper() else a for a in args]
    sign = '-' if pred.startswith('-') else ''
    bare_pred = pred.lstrip('-')
    return f'{sign}{bare_pred}({",".join(new_args)})'


def _is_valid_stmt(s: str) -> bool:
    s = s.strip()
    return bool(s) and (
        _FORALL_RE.match(s)
        or '->' in s
        or bool(_ATOM_RE.match(s))
        or (s.startswith('not ') and bool(_ATOM_RE.match(s[4:].strip())))
    )


def _extract_statements(raw: str) -> list[str]:
    """Split a raw FOL section into individual statements.

    Prefers newline split (gold data); falls back to regex tokenisation
    (model output, space-separated on one line).
    """
    by_line = [ln.strip() for ln in raw.split('\n') if ln.strip()]
    # If multiple valid lines → use them (gold/newline format)
    if sum(1 for l in by_line if _is_valid_stmt(l)) > 1:
        return [ln for ln in by_line if _is_valid_stmt(ln)]
    # Fall back to regex token scan (space-separated model output)
    return [m.strip() for m in _FOL_ATOM_RE.findall(raw) if m.strip()]


def _extract_between(text: str, start: str, end: str) -> str:
    if start not in text or end not in text:
        return ''
    return text.split(start, 1)[1].split(end, 1)[0]


def parse_logic(logic: str) -> dict:
    """Parse a logic string (gold or model output) into premises + question.

    Unlike evaluate.parse_output, this correctly handles ground conjunctive rules
    (A and B -> C) by preferring newline split over regex atom scan.
    """
    prem_raw  = _extract_between(logic, '<extra_id_1>', '<extra_id_2>')
    q_raw     = _extract_between(logic, '<extra_id_2>', '<extra_id_3>')

    premises = _extract_statements(prem_raw)
    question = q_raw.strip().split('\n')[0].strip()  # first line only

    return {'premises_fol': premises, 'question_fol': question}


def fol_to_asp(premises: list[str], question: str):
    """Convert FOL premises + question to ASP program.

    Returns:
        asp_program   — full ASP string
        rule_names    — {rule_id: human-readable FOL string}
        goal_atom     — ASP goal atom (positive form, for querying stable model)
        rule_structs  — list of (head_asp, body_asp_list, rid) for backward tracing
        base_facts    — set of base fact ASP strings (no trailing dot)
    """
    facts_asp    = []
    rules_asp    = []
    prov_asp     = []
    rule_names   = {}
    rule_structs = []  # (head_asp, [body_atoms], rid)
    rule_idx     = 0

    def _add_rule(head_asp, body_asp, rid, stmt):
        rules_asp.append(f'{head_asp} :- {", ".join(body_asp)}.')
        for ba in body_asp:
            prov_asp.append(f'derived_by({head_asp},{ba},{rid}) :- {", ".join(body_asp)}.')
        rule_names[rid] = stmt
        rule_structs.append((head_asp, body_asp, rid))

    for stmt in premises:
        stmt = stmt.strip()

        # Universal rule: forall x (body1 and body2 -> head)
        fm = _FORALL_RE.match(stmt)
        if fm:
            inner = fm.group(1).strip()
            parts = _IMPL_SPLIT_RE.split(inner)
            if len(parts) == 2:
                body_raw, head_raw = parts
                body_parsed = [_parse_atom(b) for b in _AND_SPLIT_RE.split(body_raw)]
                head_parsed = _parse_atom(head_raw)
                if all(body_parsed) and head_parsed:
                    rule_idx += 1
                    rid = f'rule{rule_idx}'
                    body_asp = [_atom_to_asp(*b) for b in body_parsed]  # type: ignore[arg-type]
                    head_asp = _atom_to_asp(*head_parsed)                # type: ignore[arg-type]
                    _add_rule(head_asp, body_asp, rid, stmt)
                    continue

        # Ground rule: A and B -> C  (single or conjunctive antecedent)
        if '->' in stmt:
            parts = _IMPL_SPLIT_RE.split(stmt)
            if len(parts) == 2:
                body_raw, head_raw = parts
                body_parsed = [_parse_atom(b) for b in _AND_SPLIT_RE.split(body_raw)]
                head_parsed = _parse_atom(head_raw)
                if all(body_parsed) and head_parsed:
                    rule_idx += 1
                    rid = f'rule{rule_idx}'
                    body_asp = [_atom_to_asp(*b) for b in body_parsed]  # type: ignore[arg-type]
                    head_asp = _atom_to_asp(*head_parsed)                # type: ignore[arg-type]
                    _add_rule(head_asp, body_asp, rid, stmt)
                    continue

        # Negated fact: not Pred(args)
        if stmt.startswith('not '):
            parsed = _parse_atom(stmt[4:])
            if parsed:
                facts_asp.append(_atom_to_asp(True, *parsed[1:]) + '.')
                continue

        # Simple fact: Pred(args)
        parsed = _parse_atom(stmt)
        if parsed:
            facts_asp.append(_atom_to_asp(*parsed) + '.')  # type: ignore[arg-type]

    # Parse goal atom (positive form only — negation handled by caller)
    goal_atom = ''
    q = question.strip()
    if q.startswith('not '):
        q = q[4:].strip()
    q_parsed = _parse_atom(q)
    if q_parsed:
        goal_atom = _atom_to_asp(False, *q_parsed[1:])

    asp_program = (
        '\n'.join(facts_asp) + '\n'
        + '\n'.join(rules_asp) + '\n'
        + '\n'.join(prov_asp) + '\n'
    )
    base_facts = {f[:-1] for f in facts_asp}  # strip trailing '.'

    return asp_program, rule_names, goal_atom, rule_structs, base_facts


# ---------------------------------------------------------------------------
# Clingo reasoning
# ---------------------------------------------------------------------------

def _run_clingo(asp_program: str, goal_atom: str, negated_question: bool,
                rule_structs=None, base_facts=None) -> tuple[str, list[str]]:
    """Run clingo and return (answer, proof_steps).

    answer: 'True' / 'False' / 'Unknown'
    proof_steps: list of human-readable derivation steps
    """
    ctl = clingo.Control(['--warn=none'])
    ctl.add('base', [], asp_program)
    ctl.ground([('base', [])])

    model_atoms = set()
    with ctl.solve(yield_=True) as handle:
        for model in handle:
            for sym in model.symbols(atoms=True):
                model_atoms.add(str(sym))

    # OWA semantics: True/False require explicit derivation; else Unknown.
    #   Positive question Pred(args):  True if goal_atom derived, False if -goal_atom derived
    #   Negated question  not Pred:    True if -goal_atom derived, False if goal_atom derived
    if goal_atom:
        pos_holds = goal_atom in model_atoms
        neg_holds = ('-' + goal_atom) in model_atoms
        if not negated_question:
            if pos_holds:
                answer = 'True'
            elif neg_holds:
                answer = 'False'
            else:
                answer = 'Unknown'
        else:
            if neg_holds:
                answer = 'True'
            elif pos_holds:
                answer = 'False'
            else:
                answer = 'Unknown'
    else:
        answer = 'Unknown'

    # Build dependency graph from derived_by atoms:
    # deps[conclusion] = [(all_premises_for_this_rule, ruleid), ...]
    deps: dict[str, list[tuple[list[str], str]]] = {}
    by_conc_rule: dict[tuple[str, str], list[str]] = {}
    for pa in model_atoms:
        if not pa.startswith('derived_by('):
            continue
        inner = pa[len('derived_by('):-1]
        parts = _split_derived_by(inner)
        if len(parts) == 3:
            conc, prem, rid = parts
            key = (conc, rid)
            if key not in by_conc_rule:
                by_conc_rule[key] = []
            by_conc_rule[key].append(prem)

    for (conc, rid), prems in by_conc_rule.items():
        deps.setdefault(conc, []).append((prems, rid))

    # Backward chain from goal target to collect only relevant steps
    proof_target = ('-' + goal_atom) if (answer == 'True' and negated_question) or \
                                        (answer == 'False' and not negated_question) else goal_atom
    proof_steps = []

    if answer == 'Unknown':
        # Trace what would be needed and where the chain breaks
        proof_steps = _trace_unknown(goal_atom, model_atoms, rule_structs or [], base_facts or set())
    elif proof_target and proof_target in deps:
        visited: set[str] = set()
        queue = [proof_target]
        while queue:
            atom = queue.pop(0)
            if atom in visited:
                continue
            visited.add(atom)
            for prems, rid in deps.get(atom, []):
                prems_str = ' and '.join(prems)
                proof_steps.append(f'{prems_str} + {rid} -> therefore {atom}')
                queue.extend(prems)
        proof_steps.reverse()   # backward BFS → flip to forward (cause before effect)
    elif answer in ('True', 'False') and proof_target and proof_target not in deps:
        proof_steps = [f'(base fact) {proof_target}']

    return answer, proof_steps


def _trace_unknown(goal: str, model_atoms: set, rule_structs: list, base_facts: set,
                   depth: int = 0, visited: set = None) -> list[str]:
    """Backward-trace a goal that cannot be derived, showing where the chain breaks.

    Uses proper unification so universal rules (with variable X) are only
    selected when their head actually matches the goal's argument pattern.
    """
    if visited is None:
        visited = set()
    if goal in visited or depth > 8:
        return []
    visited.add(goal)

    if goal in model_atoms or goal in base_facts:
        return []   # already satisfied — no failure here

    # Find applicable rules via unification
    applicable = []
    for head_tmpl, body_tmpl, rid in rule_structs:
        bindings = _unify(goal, head_tmpl)
        if bindings is not None:
            bound_body = [_apply_subst(b, bindings) for b in body_tmpl]
            applicable.append((bound_body, rid))

    if not applicable:
        return [f'{goal} <- [no rule or base fact]']

    lines = []
    bound_body, rid = applicable[0]
    body_str = ' and '.join(bound_body)
    lines.append(f'{goal} <- {body_str} + {rid}')
    missing = [b for b in bound_body if b not in model_atoms and b not in base_facts]
    for m in missing:
        lines.extend(_trace_unknown(m, model_atoms, rule_structs, base_facts, depth + 1, visited))
    return lines


def _split_derived_by(inner: str) -> list[str]:
    """Split 'conclusion,premise,ruleid' respecting nested parens."""
    parts = []
    depth = 0
    buf = []
    for ch in inner:
        if ch == '(':
            depth += 1
            buf.append(ch)
        elif ch == ')':
            depth -= 1
            buf.append(ch)
        elif ch == ',' and depth == 0:
            parts.append(''.join(buf))
            buf = []
        else:
            buf.append(ch)
    if buf:
        parts.append(''.join(buf))
    return parts


# ---------------------------------------------------------------------------
# End-to-end pipeline
# ---------------------------------------------------------------------------

def reason(raw_output: str) -> dict:
    """Parse model output and run clingo. Returns result dict."""
    parsed    = parse_logic(raw_output)
    premises  = parsed['premises_fol']
    question  = parsed['question_fol']

    # Detect question negation (question is queried as-is; negation in NL question
    # is encoded in the FOL question atom via 'not Pred(...)')
    negated_question = question.strip().startswith('not ')
    clean_question   = question.strip()[4:].strip() if negated_question else question.strip()

    asp_program, rule_names, goal_atom, rule_structs, base_facts = fol_to_asp(premises, clean_question)

    if not goal_atom:
        return {
            'premises_fol': premises,
            'question_fol': question,
            'answer': 'Unknown',
            'proof': ['[goal could not be parsed]'],
            'asp_program': asp_program,
        }

    answer, proof = _run_clingo(asp_program, goal_atom, negated_question,
                                rule_structs=rule_structs, base_facts=base_facts)

    return {
        'premises_fol': premises,
        'question_fol': question,
        'answer': answer,
        'proof': proof,
        'asp_program': asp_program,
    }


# ---------------------------------------------------------------------------
# Evaluation on existing eval JSONL
# ---------------------------------------------------------------------------

def evaluate_eval_file(eval_file: str, max_samples: int = 0, verbose: bool = False):
    lines = Path(eval_file).read_text().strip().splitlines()
    if max_samples:
        lines = lines[:max_samples]

    correct = 0
    parse_fail = 0
    clingo_fail = 0
    total = len(lines)

    for i, line in enumerate(lines):
        d = json.loads(line)
        raw = d.get('raw_output', '')
        gold = d.get('gold_answer', '').strip()

        try:
            result = reason(raw)
            pred   = result['answer']
        except Exception as e:
            clingo_fail += 1
            pred = 'Unknown'
            if verbose:
                print(f'[{i}] CLINGO ERROR: {e}')

        if not result.get('premises_fol'):
            parse_fail += 1

        ok = pred.lower() == gold.lower()
        if ok:
            correct += 1

        if verbose or (not ok and i < 20):
            print(f'[{i}] gold={gold} pred={pred} {"OK" if ok else "FAIL"}')
            if result.get('proof'):
                for step in result['proof']:
                    print(f'     {step}')

    print(f'\n=== Results ({total} samples) ===')
    print(f'Answer accuracy : {correct}/{total} = {100*correct/total:.1f}%')
    print(f'Parse failures  : {parse_fail}')
    print(f'Clingo errors   : {clingo_fail}')


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval-file', default='outputs/eval_v13_final_200.jsonl')
    parser.add_argument('--max-samples', type=int, default=0)
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()

    evaluate_eval_file(args.eval_file, args.max_samples, args.verbose)


if __name__ == '__main__':
    main()
