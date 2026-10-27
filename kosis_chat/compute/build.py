"""The plans the rules build (T0-04 S5): one function per question shape, each returning the v3 blocks (PlanV3) and
the name of the shape (pattern) that templates and charts read.

The rules (analyze.py) pick the cells by source_ref; these functions only arrange them into blocks, so the facts
come out in the same order as before S5 (the v2 engine's order, which the golden set and the templates expect).
Cells are ordered by period where that order matters; a source_ref ends with its period.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from kosis_chat.answer.render import span_text

from kosis_ai.schemas import AggregateStep, CombineStep, PlanV3, QuestionType, RankStep, SelectStep  # isort: skip

# What the plan does, for the sentence and the chart: the five plain types, one number over periods (avg, sum), two
# period groups compared (pavg, psum), a period against an average (vsa), two shares compared (shr), the gap between
# two targets at two periods (gap), and the change of two targets compared (grw).
Pattern = Literal["value", "trend", "compare", "share", "extreme", "avg", "sum", "pavg", "psum", "vsa", "shr", "gap", "grw"]
Agg = Literal["avg", "sum"]
AGG_WORDS = {"avg": "평균", "sum": "합계"}


@dataclass(frozen=True)
class TermSteps:
    """One number a period pattern compares: its cell step (the parts for a share), its value step, its total."""

    cells: str
    value: str
    total: str | None = None
    agg: str | None = None
    periods: tuple[str, str] = ("", "")


@dataclass(frozen=True)
class Built:
    plan: PlanV3
    pattern: Pattern
    terms: list[TermSteps] = field(default_factory=list)
    context: list[str] = field(default_factory=list)  # cells shown beside the answer, never computed (a 전국 value)
    reference: int | None = None  # compare: which of the two cells the question names as the reference


def period_of(ref: str) -> str:
    return ref.rsplit("|", 1)[-1]


def by_period(refs: list[str]) -> list[str]:
    return sorted(refs, key=period_of)


def _plan(steps: list, show: list[str], present: QuestionType, note: str | None) -> PlanV3:
    return PlanV3(steps=steps, show=show, present=present, note=note)


# ---------------------------------------------------------------- the five plain types


def value(cell: str, context: list[str] | None = None, note: str | None = None) -> Built:
    return Built(_plan([SelectStep(id="c", cells=[cell])], ["c"], "value", note), "value", context=list(context or []))


def trend(cells: list[str], note: str | None = None) -> Built:
    if len(cells) < 2:
        raise ValueError(f"trend needs 2+ cells, got {len(cells)}")
    steps = [SelectStep(id="s", cells=by_period(cells)), AggregateStep(id="a", of="s", fn="first"),
             AggregateStep(id="b", of="s", fn="last"), CombineStep(id="d", a="b", b="a", fn="diff"),
             CombineStep(id="g", a="b", b="a", fn="pct_change", optional=True)]
    return Built(_plan(steps, ["a", "b", "d", "g"], "trend", note), "trend")


def compare(cells: list[str], reference: int | None = None, note: str | None = None) -> Built:
    """Two cells: the subject minus the named reference (reference = its index), else larger minus smaller."""
    if len(cells) != 2:
        raise ValueError(f"compare needs 2 cells, got {len(cells)}")
    if reference not in (None, 0, 1):
        raise ValueError("reference is an index (0 or 1) into the two compare cells")
    steps: list = [SelectStep(id="x", cells=[cells[0]]), SelectStep(id="y", cells=[cells[1]])]
    if reference is None:
        a, b, orient = "x", "y", "larger_first"
    else:
        a, b, orient = ("y", "x", "fixed") if reference == 0 else ("x", "y", "fixed")
    steps += [CombineStep(id="d", a=a, b=b, fn="diff", orient=orient),
              CombineStep(id="r", a=a, b=b, fn="ratio", orient=orient, optional=True)]
    return Built(_plan(steps, ["x", "y", "d", "r"], "compare", note), "compare", reference=reference)


def share(parts: list[str], total: str, note: str | None = None) -> Built:
    if not parts:
        raise ValueError("share needs one or more part cells")
    steps = [SelectStep(id="p", cells=parts), SelectStep(id="t", cells=[total]), CombineStep(id="s", a="p", b="t", fn="share")]
    return Built(_plan(steps, ["p", "t", "s"], "share", note), "share")


def extreme(candidates: list[str], pick: Literal["max", "min"], totals: list[str] | None = None,
            context: list[str] | None = None, note: str | None = None) -> Built:
    steps = [RankStep(id="k", candidates=candidates, totals=list(totals or []), pick=pick)]
    return Built(_plan(steps, ["k"], "extreme", note), "extreme", context=list(context or []))


# ---------------------------------------------------------------- numbers over periods


def _group(n: int, cells: list[str], agg: Agg | None) -> tuple[list, TermSteps]:
    cells = by_period(cells)
    periods = (period_of(cells[0]), period_of(cells[-1]))
    steps: list = [SelectStep(id=f"c{n}", cells=cells)]
    if agg is None:
        if len(cells) != 1:
            raise ValueError(f"a single period holds one cell, got {len(cells)}")
        return steps, TermSteps(f"c{n}", f"c{n}", periods=periods)
    if len(cells) < 2:
        raise ValueError(f"{agg} needs two or more cells, got {len(cells)}")
    steps.append(AggregateStep(id=f"v{n}", of=f"c{n}", fn=agg))
    return steps, TermSteps(f"c{n}", f"v{n}", agg=agg, periods=periods)


def over_periods(cells: list[str], agg: Agg, present: QuestionType = "trend", note: str | None = None) -> Built:
    """One number over periods: their average or their sum ("최근 5년 평균", "2024년 상반기 혼인 건수")."""
    if present not in ("value", "trend"):
        raise ValueError(f"an {agg} over periods answers a value or trend question, not {present}")
    steps, term = _group(1, cells, agg)
    show = [term.cells, term.value] if len(cells) <= 2 else [term.value]
    return Built(_plan(steps, show, present, note), agg, [term])


def _compared(first: TermSteps, later: TermSteps, steps: list, pattern: Pattern, note: str | None) -> Built:
    what = f"{span_text(*first.periods)} {AGG_WORDS.get(first.agg or '', '')}".strip()
    steps += [CombineStep(id="d", a=later.value, b=first.value, fn="diff", note=f"{what} 대비 차이"),
              CombineStep(id="g", a=later.value, b=first.value, fn="pct_change", optional=True, note=f"{what} 대비 증감률")]
    if pattern == "vsa":
        steps.append(CombineStep(id="l", a=later.value, b=first.value, fn="index", optional=True, note=f"{what}을 100으로 본 수준"))
    return Built(_plan(steps, [first.value, later.value, "d", "g"], "trend", note), pattern, [first, later])


def groups_compared(early: list[str], late: list[str], agg: Agg, note: str | None = None) -> Built:
    """Two period groups, each averaged (pavg) or summed (psum); the later one against the earlier one."""
    if period_of(min(late, key=period_of)) < period_of(min(early, key=period_of)):
        early, late = late, early
    s1, first = _group(1, early, agg)
    s2, later = _group(2, late, agg)
    return _compared(first, later, s1 + s2, "pavg" if agg == "avg" else "psum", note)


def vs_average(averaged: list[str], cell: str, note: str | None = None) -> Built:
    """One period against the average of a range ("2025년은 2020~2024년 평균보다")."""
    s1, first = _group(1, averaged, "avg")
    s2, later = _group(2, [cell], None)
    return _compared(first, later, s1 + s2, "vsa", note)


def share_change(early: tuple[list[str], str], late: tuple[list[str], str], note: str | None = None) -> Built:
    """The same parts' share of the total at two periods, the later one against the earlier one in %p."""
    if len(early[0]) != len(late[0]):
        raise ValueError("both shares need the same parts")
    if period_of(late[1]) < period_of(early[1]):
        early, late = late, early
    steps: list = []
    terms: list[TermSteps] = []
    for n, (parts, total) in enumerate((early, late), start=1):
        steps += [SelectStep(id=f"p{n}", cells=parts), SelectStep(id=f"t{n}", cells=[total]),
                  CombineStep(id=f"v{n}", a=f"p{n}", b=f"t{n}", fn="share")]
        terms.append(TermSteps(f"p{n}", f"v{n}", total=f"t{n}", periods=(period_of(total), period_of(total))))
    first, later = terms
    steps.append(CombineStep(id="d", a=later.value, b=first.value, fn="pp_diff",
                             note=f"{span_text(*first.periods)} 대비 {span_text(*later.periods)} 비중 변화"))
    return Built(_plan(steps, [first.value, later.value, "d"], "share", note), "shr", terms)


# ---------------------------------------------------------------- two targets at two periods (new in S5)


def gap(early: tuple[str, str], late: tuple[str, str], note: str | None = None) -> Built:
    """How the gap between two targets changed: the gap at each period (larger minus smaller), later minus earlier.

    early / late: the (first target, second target) cells at the earlier and the later period."""
    if period_of(late[0]) < period_of(early[0]):
        early, late = late, early
    steps = [SelectStep(id="x1", cells=[early[0]]), SelectStep(id="y1", cells=[early[1]]),
             SelectStep(id="x2", cells=[late[0]]), SelectStep(id="y2", cells=[late[1]]),
             CombineStep(id="g1", a="x1", b="y1", fn="diff", orient="larger_first", note=f"{period_of(early[0])} 격차"),
             CombineStep(id="g2", a="x2", b="y2", fn="diff", orient="larger_first", note=f"{period_of(late[0])} 격차"),
             CombineStep(id="d", a="g2", b="g1", fn="diff", note="격차 변화")]
    return Built(_plan(steps, ["x1", "y1", "x2", "y2", "g1", "g2", "d"], "compare", note), "gap")


def growth(first: tuple[str, str], second: tuple[str, str], amount: bool = False, decline: bool = False,
           note: str | None = None) -> Built:
    """Which of two targets changed more: each target's growth rate (or change, amount=True), then their gap.

    first / second: each target's (earlier, later) cells. The gap puts the larger change first, or for a question
    about a decline ("감소율이 더 큰") the more negative one."""
    fn = "diff" if amount else "pct_change"
    unit_word = "증감" if amount else "증감률"
    steps: list = []
    for n, (early, late) in enumerate((first, second), start=1):
        early, late = by_period([early, late])
        steps += [SelectStep(id=f"a{n}", cells=[early]), SelectStep(id=f"b{n}", cells=[late]),
                  CombineStep(id=f"v{n}", a=f"b{n}", b=f"a{n}", fn=fn, note=unit_word)]
    steps.append(CombineStep(id="d", a="v1", b="v2", fn="pp_diff" if not amount else "diff",
                             orient="smaller_first" if decline else "larger_first", note=f"{unit_word} 차이"))
    return Built(_plan(steps, ["a1", "b1", "a2", "b2", "v1", "v2", "d"], "compare", note), "grw")
