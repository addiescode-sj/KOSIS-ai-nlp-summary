"""The rules' plan over two or three tables (loop engineering, 2026-11-20; rules written in loop 1).

kosis_chat.calc.calculate calls step_plan_multi_rules for a question over several tables and computes the plan it
returns with compute(tables=): no model is called on this path.

What must stay (kosis_chat/calc.py reads them): the names PlanMultiRulesIn, PlanMultiRulesOut and
step_plan_multi_rules, and their fields. Everything else in this file can change.

How a question is planned (loop 1). Every table given is read once: one cell per table (or, for a question that only
lists values, the cells of every named target and period), at the same region and period.

  1. Per table: the one-table rules read the question against the table's index (analyze.rules.rule_plan): the
     categories it names and the periods. The item is the table's only item, else the item the question names. An
     item name found only inside a longer item name of another table ("사망자수" inside "영아사망자수") is that other
     table's, so two metrics never land on one table. A count item the question calls by its stem ("혼인 1건당" for
     혼인건수) is found by the stem.
  2. Periods: the periods the question names, read on each table; none named: the latest period every table holds a
     number for. A named period a table lacks is left open (never replaced by another).
  3. What is computed, from the question's wording (shared/patterns.py MULTI_*), in this order: a value per unit of
     another ("~당": the table named before 당 is the denominator, its number the scale), a share ("비중", "몇
     퍼센트"), a ratio ("몇 배", "나누"), a sum (judge.question.SUM_WORDS), a difference (차이, 많/적, 빼) and
     otherwise the values side by side. Which table is the reference follows the question (보다, 에서, 대비, 중).
"""

from __future__ import annotations

import re
from itertools import product
from typing import Literal

from pydantic import Field, SkipValidation

from kosis_chat.analyze import blocks as build
from kosis_chat.analyze.rules import _fixed, _lookup, rule_plan
from kosis_chat.analyze.schemas import Built, Plan
from kosis_chat.judge.question import ITEM_AXIS, SUM_WORDS, TableIndex, label_forms, match_labels, squash
from kosis_chat.schemas.resources import MultiResources, StatResources
from kosis_chat.shared.boundary import BoundaryModel
from kosis_chat.shared.patterns import (
    COUNT_STEM_SUFFIX,
    MULTI_DIFF,
    MULTI_FEWER,
    MULTI_RATIO,
    MULTI_SHARE,
    MULTI_WHOLE_AFTER,
    PER_UNIT,
    TOTAL_PREFIX,
    per_scale,
)

from kosis_ai.schemas import CombineStep, PlanV3, SelectStep  # isort: skip

NO_RULES = "multi_table_rules"  # kept for callers that name it: the rules could not plan the question

Op = Literal["list", "diff", "sum", "ratio", "per", "share"]


class PlanMultiRulesIn(BoundaryModel):
    """step_plan_multi_rules's input: the question."""

    question: str


class PlanMultiRulesOut(BoundaryModel):
    """step_plan_multi_rules's output. route "built": built is the plan to compute. route "unresolved": built is
    None and unresolved names what the rules could not decide."""

    built: SkipValidation[Built | None] = None
    unresolved: SkipValidation[list[str]] = Field(default_factory=list)
    route: Literal["built", "unresolved"]


class _Open(Exception):
    def __init__(self, what: str) -> None:
        super().__init__(what)
        self.what = what


# ---------------------------------------------------------------- where each table's metric is named


def _spans(form: str, q: str) -> list[tuple[int, int]]:
    out, at = [], q.find(form)
    while form and at >= 0:
        out.append((at, at + len(form)))
        at = q.find(form, at + 1)
    return out


def _stems(label: str) -> set[str]:
    """Shorter names of an item: without a leading 총 ('총인구수' -> '인구수'), and a count item by its stem ('혼인건수' ->
    '혼인', '사망자수' -> '사망', '총인구수' -> '인구'). The stem only for names that end with a count suffix."""
    exact, _, _ = label_forms(label)
    out = set()
    for form in exact:
        bare = TOTAL_PREFIX.sub("", form)
        if bare != form and len(bare) >= 2:
            out.add(bare)
        for name in {form, bare}:
            m = COUNT_STEM_SUFFIX.search(name)
            if m and m.start() >= 2:
                out.add(name[: m.start()])
    return out


def _item_spans(index: TableIndex, q: str, question: str) -> dict[str, tuple[list[tuple[int, int]], bool]]:
    """item code -> (where the question names it, by the whole name (True) or only by its stem (False))."""
    if ITEM_AXIS not in index.axes:
        return {}
    out: dict[str, tuple[list[tuple[int, int]], bool]] = {}
    for m in match_labels(index, question):
        if m.axis == ITEM_AXIS:
            out.setdefault(m.code, ([], True))[0].extend(_spans(m.form, q))
    if out:
        return out
    for code, label in index.axes[ITEM_AXIS].labels.items():
        spans = [s for stem in _stems(label) for s in _spans(stem, q)]
        if spans:
            out[code] = (spans, False)
    return out


def _covered(span: tuple[int, int], others: list[tuple[int, int]]) -> bool:
    a, b = span
    return any(x <= a and b <= y and (y - x) > (b - a) for x, y in others)


class _Side:
    """One table's part of the question: its rules' plan, its item and where the question names it."""

    def __init__(self, res: StatResources, question: str) -> None:
        self.res = res
        self.index = res.index
        self.plan: Plan = rule_plan(question, res.index)
        self.items = _item_spans(res.index, squash(question), question)
        self.item: str | None = None
        self.at: int | None = None

    @property
    def tbl_id(self) -> str:
        return self.res.table.meta.tbl_id


def _assign_items(sides: list[_Side]) -> None:
    """Each table's item: an item named only inside another table's longer item name is not this table's."""
    for side in sides:
        others = [s for o in sides if o is not side for spans, _ in o.items.values() for s in spans]
        kept = {code: [s for s in spans if not _covered(s, others)] for code, (spans, _) in side.items.items()}
        kept = {code: spans for code, spans in kept.items() if spans}
        labels = side.index.axes[ITEM_AXIS].labels if ITEM_AXIS in side.index.axes else {}
        if len(labels) == 1:
            side.item = next(iter(labels))
            spans = kept.get(side.item, [])
        elif len(kept) == 1:
            side.item, spans = next(iter(kept.items()))
        else:
            spans = []
        side.at = min(a for a, _ in spans) if spans else None


# ---------------------------------------------------------------- what is computed


def _ordered(sides: list[_Side]) -> list[_Side]:
    """The tables in the order the question names their metrics (an unnamed one last, in input order)."""
    return sorted(sides, key=lambda s: (s.at is None, s.at if s.at is not None else 0))


def _before(sides: list[_Side], at: int) -> _Side | None:
    """The table whose metric is named last before position at."""
    named = [s for s in sides if s.at is not None and s.at < at]
    return max(named, key=lambda s: s.at) if named else None


def _end(side: _Side) -> int:
    spans = [s for spans, _ in side.items.values() for s in spans if s[0] == side.at]
    return max((b for _, b in spans), default=side.at or 0)


def _masked(question: str, sides: list[_Side]) -> str:
    """The squashed question with every label any table matched blanked out (same length, so positions hold): a word
    inside a label ('국적', '누적') is not an operation word."""
    q = squash(question)
    for side in sides:
        for m in match_labels(side.index, question):
            for a, b in _spans(m.form, q):
                q = q[:a] + "#" * (b - a) + q[b:]
    return q


def _whole(order: list[_Side], q: str) -> _Side:
    """The denominator of a share or ratio: the table named right before 중/가운데/대비/에서/의 몇, else the last named."""
    return next((s for s in order if s.at is not None and MULTI_WHOLE_AFTER.match(q[_end(s):])), order[-1])


def _operation(question: str, sides: list[_Side]) -> tuple[Op, list[_Side], float | None, bool]:
    """(what is computed, the tables in the order the formula reads them, the per scale, whether that order is the
    question's: False for a difference with no reference named, which is larger minus smaller)."""
    q = _masked(question, sides)
    order = _ordered(sides)
    if len(sides) == 2:
        if (m := PER_UNIT.search(question)) and q[len(squash(question[: m.end()])) - 1] != "#":
            at = len(squash(question[: m.start()]))  # "혼인 1건당 출생아 수": the one named before 당 is the denominator
            denominator = _before(sides, at + 1) or order[-1]
            numerator = next(s for s in sides if s is not denominator)
            scale = per_scale(m)
            return ("ratio" if scale == 1 else "per"), [numerator, denominator], scale, True
        if MULTI_SHARE.search(q):
            whole = _whole(order, q)
            return "share", [next(s for s in sides if s is not whole), whole], None, True
        if MULTI_RATIO.search(q):
            whole = _whole(order, q)
            return "ratio", [next(s for s in sides if s is not whole), whole], None, True
    if SUM_WORDS.search(q):
        return "sum", order, None, True
    if len(sides) == 2 and MULTI_DIFF.search(q):
        if (at := q.find("보다")) >= 0 and (reference := _before(sides, at)) is not None:
            subject = next(s for s in sides if s is not reference)
            fewer = bool(MULTI_FEWER.search(q[at:]))
            return "diff", ([reference, subject] if fewer else [subject, reference]), None, True
        if (at := q.find("에서")) >= 0 and (first := _before(sides, at)) is not None and "빼" in q[at:]:
            return "diff", [first, next(s for s in sides if s is not first)], None, True
        return "diff", order, None, False
    return "list", order, None, True


# ---------------------------------------------------------------- the cells


def _asked_periods(side: _Side) -> list[str]:
    p = side.plan
    if p.missing_years:
        raise _Open(f"{side.tbl_id}: 시점 {'·'.join(p.missing_years)} 없음")
    return list(p.periods)


def _fixed_sets(side: _Side, many: bool) -> list[dict[str, str]]:
    """One code per category; with many, every combination of the codes a category names several of."""
    codes = dict(side.plan.codes)
    if side.item is None:
        if ITEM_AXIS in side.index.axes and len(side.index.axes[ITEM_AXIS].labels) > 1:
            raise _Open(f"{side.tbl_id}: 항목")
    else:
        codes[ITEM_AXIS] = [side.item]
    several = {n: v for n, v in codes.items() if len(v) > 1 and n != ITEM_AXIS}
    if several and not many:
        raise _Open(f"{side.tbl_id}: {', '.join(several)} 여러 개")
    problems: list[str] = []
    base = _fixed(side.plan.model_copy(update={"codes": {k: v for k, v in codes.items() if k not in several}}),
                  side.index, set(several), problems)
    if problems:
        raise _Open(f"{side.tbl_id}: {', '.join(problems)}")
    names = list(several)
    return [{**base, **dict(zip(names, combo))} for combo in product(*(several[n] for n in names))] or [base]


def _common_period(sides: list[_Side], fixed: dict[int, list[dict[str, str]]]) -> str:
    """The latest period every table holds a number for, at the cells asked."""
    shared: set[str] | None = None
    for side in sides:
        series = _lookup(side.res.table).numeric_periods_of(fixed[id(side)])
        found = set.intersection(*(set(s) for s in series)) if series else set()
        shared = found if shared is None else shared & found
    if not shared:
        raise _Open("모든 표에 있는 시점 없음")
    return max(shared)


def _cells(side: _Side, sets: list[dict[str, str]], periods: list[str]) -> list[str]:
    wanted = [(f, p) for f in sets for p in periods]
    found = _lookup(side.res.table).cells(wanted)
    missing = [f"{side.tbl_id} {p}" for (f, p), c in zip(wanted, found) if c is None]
    if missing:
        raise _Open(f"칸 없음: {', '.join(missing[:3])}")
    return [c.source_ref for c in found]


def _plan(question: str, res: MultiResources) -> Built:
    sides = [_Side(r, question) for r in res.tables]
    _assign_items(sides)
    op, order, scale, directed = _operation(question, sides)
    many = op == "list"
    fixed = {id(s): _fixed_sets(s, many) for s in sides}
    asked = {id(s): _asked_periods(s) for s in sides}
    if any(asked.values()) and not all(asked.values()):
        raise _Open("질문한 시점이 없는 표")
    if not any(asked.values()):
        latest = _common_period(sides, fixed)
        asked = {id(s): [latest] for s in sides}
    if not many and any(len(p) != 1 for p in asked.values()):
        raise _Open("표마다 시점 하나가 아님")
    steps: list = []
    ids: list[str] = []
    for n, side in enumerate(order):
        refs = _cells(side, fixed[id(side)], asked[id(side)])
        steps.append(SelectStep(id=f"t{n}", cells=refs, table=side.tbl_id))
        ids.append(f"t{n}")
    if op == "list":
        return build.written(PlanV3(steps=steps, show=ids, present="value"))
    if op == "sum":
        last = ids[0]
        for n, other in enumerate(ids[1:], start=1):
            steps.append(CombineStep(id=f"s{n}", a=last, b=other, fn="sum", note="합계"))
            last = f"s{n}"
        return build.written(PlanV3(steps=steps, show=[*ids, last], present="compare"))
    a, b = ids
    if op == "diff":
        orient = "fixed" if directed else "larger_first"
        steps.append(CombineStep(id="d", a=a, b=b, fn="diff", orient=orient, note="차이"))
        return build.written(PlanV3(steps=steps, show=[a, b, "d"], present="compare"))
    if op == "share":
        steps.append(CombineStep(id="r", a=a, b=b, fn="share", note="비중"))
        return build.written(PlanV3(steps=steps, show=[a, b, "r"], present="share"))
    if op == "per":
        steps.append(CombineStep(id="r", a=a, b=b, fn="per", scale=scale, note=f"{b} {scale:g}당"))
    else:
        steps.append(CombineStep(id="r", a=a, b=b, fn="ratio", note="비율"))
    return build.written(PlanV3(steps=steps, show=[a, b, "r"], present="compare"))


def step_plan_multi_rules(inp: PlanMultiRulesIn, res: MultiResources) -> PlanMultiRulesOut:
    """The rules' plan for a question over res.tables (two or three), or what they could not decide.

    In: PlanMultiRulesIn(question)
    Out: PlanMultiRulesOut(built, unresolved, route "built" | "unresolved")
    Side effects: none (reads every table's index and the cells it plans through the cell store)
    """
    try:
        built = _plan(inp.question, res)
    except _Open as exc:
        return PlanMultiRulesOut(unresolved=[NO_RULES, exc.what], route="unresolved")
    return PlanMultiRulesOut(built=built, route="built")
