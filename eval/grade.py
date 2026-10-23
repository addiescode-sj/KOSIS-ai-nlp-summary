"""Deterministic grading of a StatAnswer against a GoldenItem and the Parquet cells. No LLM involved.

Error types and the verdict come from eval.scorer.RULES / decide_verdict, the same table human review uses,
so automatic and human verdicts are directly comparable.
"""

from __future__ import annotations

import ast
import operator
import re

from pydantic import BaseModel, Field

from eval.golden import GoldenItem
from eval.scorer import decide_verdict
from kosis_ai.schemas import Cell, Fact, StatAnswer, TextPart

FACT_ERRORS = ("wrong_value", "wrong_unit", "wrong_period", "wrong_dims")
# Judgements code cannot make; they stay with a human reviewer.
NEEDS_REVIEW = {"not_answerable": "insufficient_reason", "clarify": "missing_clarification"}
_NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?")
_IDENT = re.compile(r"[A-Za-z_]\w*")
_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv}


class Grade(BaseModel):
    question_id: str
    kind: str
    type: str | None = None
    eda_traps: list[str] = Field(default_factory=list)  # copied from the golden item for per-trap breakdown
    outcome: str
    verdict: str
    error_types: list[str] = Field(default_factory=list)
    number_errors: int = 0  # numbers in the answer with no support in the data (counted as unsupported_claim)
    needs_review: list[str] = Field(default_factory=list)
    details: list[str] = Field(default_factory=list)


def evaluate(expr: str, values: dict[str, float]) -> float:
    """Evaluate a derivation such as '(f2 / f1 - 1) * 100' over fact values; only + - * / and parentheses."""

    def ev(node: ast.AST) -> float:
        if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](ev(node.left), ev(node.right))
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            return -ev(node.operand)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.Name):
            return values[node.id]
        raise ValueError(f"unsupported expression {ast.dump(node)}")

    return ev(ast.parse(expr, mode="eval").body)


def _inputs(fact: Fact, by_id: dict[str, Fact]) -> frozenset[str]:
    """source_refs of the cells a fact is built from (itself for a cell fact)."""
    if fact.source_ref:
        return frozenset([fact.source_ref])
    if not fact.derivation:
        return frozenset()
    return frozenset().union(*(_inputs(by_id[n], by_id) for n in _IDENT.findall(fact.derivation) if n in by_id))


def _strip_period(ref: str) -> str:
    return ref.rsplit("|", 1)[0]


def _close(expected: float | None, actual: float | None, digits: int | None) -> bool:
    if expected is None or actual is None:
        return expected is actual
    tol = 0.5 * 10 ** -digits if digits is not None else 1e-9 * max(1.0, abs(expected))
    return abs(expected - actual) <= tol + 1e-12


def _fact_errors(item: GoldenItem, answer: StatAnswer, details: list[str]) -> set[str]:
    """Compare required facts: cell facts by source_ref, derived facts by their input cells and value within round."""
    errors: set[str] = set()
    golden_by_id = {f.id: f for f in item.required_facts + item.aux_facts}
    got_by_id = {f.id: f for f in answer.result.facts}
    got_cells = {f.source_ref: f for f in answer.result.facts if f.source_ref}
    got_derived = [(f, _inputs(f, got_by_id)) for f in answer.result.facts if f.derivation]
    for want in item.required_facts:
        if want.source_ref:
            got = got_cells.get(want.source_ref)
            if got is None:
                stripped = _strip_period(want.source_ref)
                same_cell = any(_strip_period(r) == stripped for r in got_cells)
                errors.add("wrong_period" if same_cell else "wrong_dims")
                details.append(f"{want.id}: cell {want.source_ref} missing")
            elif not _close(want.value, got.value, None):
                errors.add("wrong_value")
                details.append(f"{want.id}: value {got.value} != {want.value}")
            elif want.unit and got.unit and want.unit != got.unit:
                errors.add("wrong_unit")
                details.append(f"{want.id}: unit {got.unit!r} != {want.unit!r}")
        elif want.derivation:
            key = _inputs(want, golden_by_id)
            # Several golden facts can share inputs (difference and growth rate), so any matching candidate passes.
            candidates = [f for f, inputs in got_derived if inputs == key]
            if not candidates:
                stripped = {_strip_period(r) for r in key}
                same_cells = any({_strip_period(r) for r in inputs} == stripped for _, inputs in got_derived)
                errors.add("wrong_period" if same_cells else "wrong_dims")
                details.append(f"{want.id}: no derived fact over {sorted(key)}")
            elif not any(_close(want.value, f.value, want.round) for f in candidates):
                errors.add("wrong_value")
                details.append(f"{want.id}: derived {[f.value for f in candidates]} != {want.value}")
    for want in item.aux_facts:
        if want.source_ref and want.source_ref not in got_cells:
            errors.add("missing_aux")
            details.append(f"{want.id}: aux cell {want.source_ref} missing")
    return errors


def count_unsupported(answer: StatAnswer, cells: dict[str, Cell], details: list[str]) -> int:
    """Numbers with no support: cell facts that disagree with the Parquet cell, derived facts that do not recompute,
    and digits in connecting text (a period of a fact is allowed, e.g. '2025년')."""
    count = 0
    by_id = {f.id: f for f in answer.result.facts}
    for f in answer.result.facts:
        if f.source_ref:
            cell = cells.get(f.source_ref)
            if cell is None or not _close(cell.value, f.value, None):
                count += 1
                details.append(f"unsupported {f.id}: {f.source_ref} = {f.value}, data {cell.value_raw if cell else 'no cell'}")
        elif f.derivation:
            try:
                expected = round(evaluate(f.derivation, {k: v.value for k, v in by_id.items() if v.value is not None}), f.round)
            except (KeyError, ValueError, ZeroDivisionError, SyntaxError) as exc:
                expected = None
                details.append(f"unsupported {f.id}: cannot recompute {f.derivation!r} ({exc})")
            if not _close(expected, f.value, f.round):
                count += 1
                if expected is not None:
                    details.append(f"unsupported {f.id}: {f.derivation} = {expected}, answer says {f.value}")
    periods = {f.period for f in answer.result.facts if f.period}
    for part in answer.parts:
        if isinstance(part, TextPart):
            # ponytail: any non-period digit in text counts; widen the allow-list if T1-10 permits e.g. ordinals.
            for num in _NUMBER.findall(part.text):
                if num not in periods:
                    count += 1
                    details.append(f"unsupported text number {num!r} in {part.text!r}")
    return count


def grade(item: GoldenItem, answer: StatAnswer, cells: dict[str, Cell]) -> Grade:
    """cells: Parquet cells of the item's table keyed by source_ref (eval.golden.read_cells)."""
    details: list[str] = []
    errors: set[str] = set()
    number_errors = count_unsupported(answer, cells, details)
    if number_errors:
        errors.add("unsupported_claim")

    if item.kind in ("answerable", "default_allowed"):
        if item.kind == "answerable" and answer.outcome != "answered":
            errors.add("false_refusal")
            details.append(f"outcome {answer.outcome} on an answerable question")
        else:
            errors |= _fact_errors(item, answer, details)
        if item.kind == "default_allowed" and answer.outcome == "answered":
            if not answer.default_applied:
                errors.add("unstated_default")
            elif answer.default_applied.keys() != item.default_applied.keys():
                errors.add("disallowed_default")
                details.append(f"default {sorted(answer.default_applied)} != {sorted(item.default_applied)}")
    elif item.kind == "not_answerable":
        if answer.outcome != "refused":
            errors.add("missed_refusal")
        elif answer.refusal.reason not in item.allowed_reasons:
            errors.add("wrong_reason")
            details.append(f"reason {answer.refusal.reason} not in {item.allowed_reasons}")
    elif item.kind == "clarify" and answer.outcome != "clarify":
        errors.add("assumed_condition")
    # no_source is not checked: StatAnswer already requires a source unless the refusal is out_of_scope.

    verdict = decide_verdict(item.kind, errors)
    review = [NEEDS_REVIEW[item.kind]] if item.kind in NEEDS_REVIEW and verdict != "FAIL" else []
    return Grade(
        question_id=item.id,
        kind=item.kind,
        type=item.type,
        eda_traps=item.eda_traps,
        outcome=answer.outcome,
        verdict=verdict,
        error_types=sorted(errors),
        number_errors=number_errors,
        needs_review=review,
        details=details,
    )
