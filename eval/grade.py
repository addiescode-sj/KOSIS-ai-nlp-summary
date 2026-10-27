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
from kosis_chat.judge.question import ITEM_AXIS, is_total, squash
from kosis_ai.schemas import Cell, Fact, StatAnswer, StatQuery, TextPart

FACT_ERRORS = ("wrong_value", "wrong_unit", "wrong_period", "wrong_dims")
# Judgements code cannot make; they stay with a human reviewer.
NEEDS_REVIEW = {"not_answerable": "insufficient_reason", "clarify": "missing_clarification"}
_NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?")
_IDENT = re.compile(r"[A-Za-z_]\w*")
def _power(base: float, exponent: float) -> float:
    if base < 0 and not float(exponent).is_integer():
        raise ValueError(f"{base} ** {exponent} is not a real number")
    return base**exponent


# ** is for an average annual growth rate (v3); the schema only allows a constant exponent.
_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv, ast.Pow: _power}
SLOT_FIELDS = ("type", "items", "dims", "periods", "ranked")
PERIOD_AXIS = "시점"
_UNIT_PAREN = re.compile(r"\s*\([^()]*\)\s*$")
_SINGLE_AXES: dict[int, tuple[dict, set[str]]] = {}


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
    # T1-23 slot checks per field (None = not graded for this question); never changes the verdict
    slots: dict[str, bool | None] = Field(default_factory=dict)
    slot_recording_limit: bool = False  # the gold needs 2+ labels on an axis StatQuery.filters cannot hold


def evaluate(expr: str, values: dict[str, float]) -> float:
    """Evaluate a derivation such as '(f2 / f1 - 1) * 100' over fact values; + - * / ** and parentheses."""

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
    allowed = allowed_text_numbers(answer.result.facts)
    for part in answer.parts:
        if isinstance(part, TextPart):
            for num in _NUMBER.findall(part.text):
                if num not in allowed:
                    count += 1
                    details.append(f"unsupported text number {num!r} in {part.text!r}")
    return count


def allowed_text_numbers(facts: list[Fact]) -> set[str]:
    """Numbers connecting text may contain without being claims: a fact's period as written in Korean ('202503' ->
    '2025년 3월', '1분기') and digits that belong to a fact's labels ('1인 가구', '1985년생'). Same rule as
    kosis_chat.answer.verify.allowed_numbers, kept here so the grader does not depend on the pipeline."""
    allowed: set[str] = set()
    for f in facts:
        if f.period:
            allowed |= {f.period, f.period[:4]}
            if len(f.period) == 6 and f.period.isdigit():
                month = int(f.period[4:])
                allowed |= {str(month), str((month - 1) // 3 + 1)}
        for label in (f.dims or {}).values():
            allowed |= set(_NUMBER.findall(label))
    return allowed


def _norm(label: str) -> str:
    return squash(_UNIT_PAREN.sub("", label))


def _single_label_axes(cells: dict[str, Cell]) -> set[str]:
    """Axes with one label in the whole table (DT_1B8000F's item 'T1'): no slot to extract."""
    cached = _SINGLE_AXES.get(id(cells))
    if cached is None or cached[0] is not cells:
        labels: dict[str, set[str]] = {}
        for c in cells.values():
            labels.setdefault(ITEM_AXIS, set()).add(c.item_name)
            for d in c.dims:
                labels.setdefault(d.name, set()).add(d.label)
        cached = (cells, {a for a, ls in labels.items() if len(ls) == 1})
        _SINGLE_AXES[id(cells)] = cached
    return cached[1]


def _is_item_axis(axis: str) -> bool:
    return axis == ITEM_AXIS or "항목" in axis  # DT_1B8000F keeps its items in '기본항목별'


def _query_empty(q: StatQuery | None) -> bool:
    return q is None or not (q.type or q.item or q.items or q.periods or q.filters or q.targets or q.over)


def slot_checks(item: GoldenItem, answer: StatAnswer, cells: dict[str, Cell]) -> tuple[dict[str, bool | None], bool]:
    """Compare the parsed question (answer.query) with slots taken from the golden required cells.

    Gold: labels per axis and the first/last period of the required cells. Axes with one label in the table are
    dropped on both sides; total labels too, except in compare questions where '전국' can be the named reference.
    In extreme questions the axis whose winning label the question does not name is the ranked axis, checked
    against query.over (to_query leaves over empty when periods are ranked).
    """
    gold_cells = [cells[f.source_ref] for f in item.required_facts if f.source_ref and f.source_ref in cells]
    if not gold_cells:
        return {}, False
    single = _single_label_axes(cells)
    keep_totals = item.type == "compare"
    question = squash(item.question)

    def clean(axes: dict[str, set[str]]) -> dict[str, set[str]]:
        out = {a: {_norm(x) for x in ls if keep_totals or not is_total(x)} for a, ls in axes.items() if a not in single}
        return {a: ls for a, ls in out.items() if ls}

    gold: dict[str, set[str]] = {}
    for c in gold_cells:
        gold.setdefault(ITEM_AXIS, set()).add(c.item_name)
        for d in c.dims:
            gold.setdefault(d.name, set()).add(d.label)
    gold = clean(gold)
    periods = sorted(c.period for c in gold_cells)
    ranked = None
    if item.type == "extreme":
        # ponytail: label-not-in-question heuristic; aliases in filter labels (제주도) can misplace it
        ranked = next((a for a, ls in gold.items() if not any(x in question for x in ls)), None)
        if ranked is None and periods[-1][:4] not in question:
            ranked = PERIOD_AXIS
        gold.pop(ranked, None)
    limit = not keep_totals and any(len(ls) > 1 for a, ls in gold.items() if not _is_item_axis(a))

    q = answer.query
    graded = ["type", "items", "dims"] + ([] if ranked == PERIOD_AXIS else ["periods"]) + (["ranked"] if item.type == "extreme" else [])
    checks: dict[str, bool | None] = dict.fromkeys(SLOT_FIELDS)
    if _query_empty(q):
        return checks | dict.fromkeys(graded, False), limit
    pred: dict[str, set[str]] = {}
    if q.item:
        pred.setdefault(ITEM_AXIS, set()).add(q.item)
    pred.setdefault(ITEM_AXIS, set()).update(q.items)
    for axis, label in q.filters.items():
        pred.setdefault(axis, set()).add(label)
    for side in q.targets:
        for axis, label in side.items():
            pred.setdefault(axis, set()).add(label)
    pred = clean(pred)
    pred.pop(ranked, None)
    checks["type"] = q.type == item.type
    checks["items"] = {a: ls for a, ls in gold.items() if _is_item_axis(a)} == {a: ls for a, ls in pred.items() if _is_item_axis(a)}
    checks["dims"] = {a: ls for a, ls in gold.items() if not _is_item_axis(a)} == {a: ls for a, ls in pred.items() if not _is_item_axis(a)}
    if "periods" in graded:
        checks["periods"] = bool(q.periods) and (min(q.periods), max(q.periods)) == (periods[0], periods[-1])
    if "ranked" in graded:
        checks["ranked"] = (next(iter(q.over)) if q.over else (PERIOD_AXIS if q.pick else None)) == ranked
    return checks, limit


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
    slots, limit = slot_checks(item, answer, cells) if item.kind in ("answerable", "default_allowed") else ({}, False)
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
        slots=slots,
        slot_recording_limit=limit,
    )
