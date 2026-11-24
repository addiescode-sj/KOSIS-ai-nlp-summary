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

Loop 2: not only one cell per table.
  4. A calculation between the tables is repeated per named target (two regions: one ratio each) and per named
     period (2020년과 2023년: one ratio each), the targets and periods in question order (_repeated).
  5. A range of periods (2019~2021년) is summed per table, or averaged when an average is asked, and the
     calculation is made on those results; asked period by period (연도별, 해마다: shared EACH_PERIOD) it is repeated.
  6. A change asked at two periods or more (shared CHANGE_ASKED, RATE_OF_CHANGE, the words the one-table rules
     read) is each table's change between its first and last period: the amount, or the rate (_changes).
"""

from __future__ import annotations

import re
from itertools import product
from typing import Literal

from pydantic import Field, SkipValidation

from kosis_chat.analyze import blocks as build
from kosis_chat.analyze.rules import _fixed, _lookup, _named_one_by_one, rule_plan
from kosis_chat.analyze.schemas import Built, Plan
from kosis_chat.judge.question import (
    _DESCRIBED,
    _MEASURE_BETWEEN,
    AVG_WORDS,
    ITEM_AXIS,
    SUM_WORDS,
    TableIndex,
    total_stem,
    asked_periods,
    label_forms,
    match_labels,
    parse,
    squash,
)
from kosis_chat.schemas.resources import MultiResources, StatResources
from kosis_chat.shared.boundary import BoundaryModel
from kosis_chat.shared.patterns import (
    ASKS_AMOUNT,
    CHANGE_ASKED,
    COUNT_STEM_SUFFIX,
    EACH_PERIOD,
    EACH_TARGET,
    GAP_CHANGE,
    GAP_WORDS,
    RATE_OF_CHANGE,
    MULTI_DIFF,
    MULTI_FEWER,
    MULTI_RATIO,
    MULTI_SHARE,
    MULTI_WHOLE_AFTER,
    PER_UNIT,
    TOTAL_PREFIX,
    TOTAL_AMOUNT,
    AGAINST,
    asks_relative,
    per_scale,
)

from kosis_ai.schemas import AggregateStep, CombineStep, PlanV3, SelectStep  # isort: skip

NO_RULES = "multi_table_rules"  # kept for callers that name it: the rules could not plan the question

# change / rate (loop 2): the change of each table's value between two periods, as an amount or a rate
Op = Literal["list", "diff", "sum", "ratio", "per", "share", "change", "rate", "relative"]


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


def _match_spans(m, q: str) -> list[tuple[int, int]]:
    """Loop 16: where the squashed question names a match: its form, or, for a name matched by its parts said apart
    (Match.parts: '자녀' and '없는' of '자녀-없음'), from each place of its first part to the nearest place of its last
    part after it."""
    if m.form in q or not m.parts:
        return _spans(m.form, q)
    out = []
    for a, b in _spans(m.parts[0], q):
        end = q.find(m.parts[-1], b)
        if end >= 0:
            out.append((a, end + len(m.parts[-1])))
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


def _item_spans(index: TableIndex, q: str, question: str,
                apart: set[tuple[int, int]] | None = None) -> dict[str, tuple[list[tuple[int, int]], bool]]:
    """item code -> (where the question names it, by the whole name (True) or only by its stem (False)).

    Loop 18: a whole name matched by its stem only, without its total mark ('가구수' of '가구수-계'), is marked False
    too (_measure_noun reads it). apart, when given, collects the places of names matched by their parts said apart
    (Match.parts, the form not in the question: _match_spans), which _closes_other reads."""
    if ITEM_AXIS not in index.axes:
        return {}
    out: dict[str, tuple[list[tuple[int, int]], bool]] = {}
    for m in match_labels(index, question):
        if m.axis == ITEM_AXIS:
            spans = _match_spans(m, q)
            stem = total_stem(m.label)
            out.setdefault(m.code, ([], not (stem and m.form == squash(stem))))[0].extend(spans)
            if apart is not None and m.parts and m.form not in q:
                apart.update(spans)
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
        self.apart: set[tuple[int, int]] = set()  # loop 18: places of names matched by parts said apart
        self.items = _item_spans(res.index, squash(question), question, self.apart)
        self.item: str | None = None
        self.at: int | None = None
        # loop 12: the items a question names one by one and asks for each of ("남자와 여자 각각"): the calculation is
        # repeated per item (_fixed_sets), as per region
        self.each_item: list[str] = []

    @property
    def tbl_id(self) -> str:
        return self.res.table.meta.tbl_id


def _assign_items(sides: list[_Side], question: str) -> None:
    """Each table's item: an item named only inside another table's longer item name is not this table's.

    Loop 12: two items or more of one table, each named by a name of its own (rules._named_one_by_one, so not an
    over-wide match of one word) with a word asking for each (shared EACH_TARGET: "남자와 여자 각각"), are the targets
    the calculation between the tables is repeated for, in question order (_fixed_sets, _repeated), as two regions
    are. The table is placed where the first of them is named.

    Loop 18: a place of a name matched by its parts said apart that runs over another table's place to its end is
    the other table's (_closes_other). A whole matched by its stem only, said at every place as the noun of what is
    counted after a matched name (_measure_noun: "자녀가 없는 가구수, 자동차가 1대인 가구수"), beside one other item of
    its table that is not asked for each, leaves that item the table's."""
    each = bool(EACH_TARGET.search(_masked(question, sides)))
    q = squash(question)
    raw = {id(o): [s for spans, _ in o.items.values() for s in spans] for o in sides}
    cleared = {id(o): {code: [s for s in spans
                              if not _closes_other(s, o.apart, [x for p in sides if p is not o for x in raw[id(p)]])]
                       for code, (spans, _) in o.items.items()} for o in sides}
    wholes = {id(o): {code for code, (_, by_name) in o.items.items()
                      if not by_name and total_stem(o.index.axes[ITEM_AXIS].labels[code])} for o in sides}
    ends = {b for o in sides for code, spans in cleared[id(o)].items() if code not in wholes[id(o)] for _, b in spans}
    for side in sides:
        others = [s for o in sides if o is not side for spans in cleared[id(o)].values() for s in spans]
        kept = {code: [s for s in spans if not _covered(s, others)] for code, spans in cleared[id(side)].items()}
        kept = {code: spans for code, spans in kept.items() if spans}
        labels = side.index.axes[ITEM_AXIS].labels if ITEM_AXIS in side.index.axes else {}
        counted = [c for c in kept if c in wholes[id(side)] and _measure_noun(kept[c], ends, q)]
        if len(labels) == 1:
            side.item = next(iter(labels))
            spans = kept.get(side.item, [])
        elif len(kept) == 1:
            side.item, spans = next(iter(kept.items()))
        elif len(kept) >= 2 and each and _named_one_by_one(parse(side.index, question), ITEM_AXIS, list(kept)):
            side.each_item = sorted(kept, key=lambda c: min(a for a, _ in kept[c]))
            spans = [s for c in kept for s in kept[c]]
        elif len(kept) == 2 and len(counted) == 1:
            side.item, spans = next((c, s) for c, s in kept.items() if c not in counted)
        else:
            spans = []
        side.at = min(a for a, _ in spans) if spans else None


def _closes_other(span: tuple[int, int], apart: set[tuple[int, int]], others: list[tuple[int, int]]) -> bool:
    """Loop 18: a place of a name matched by its parts said apart (Match.parts, its form not in the question) that
    holds another table's shorter place and ends where it ends: the last part is the other table's ("자동차가 1대인
    가구수, 반려동물이 없는 가구수": '없는' closes 반려동물-없음, not 자동차-없음). Only such places: a name said as it is
    keeps its place (_covered reads those)."""
    a, b = span
    return span in apart and any(a <= x and y == b and y - x < b - a for x, y in others)


def _measure_noun(places: list[tuple[int, int]], ends: set[int], q: str) -> bool:
    """Loop 18: every place q says a whole's stem comes right after the end of a matched name of some table (with what
    judge.question._MEASURE_BETWEEN lets stand between) or after a describing word (_DESCRIBED): the stem is the noun
    of what is counted, as one table's judge.question._measured_whole reads it."""
    return all(_DESCRIBED.search(q[:at]) is not None
               or any(e <= at and _MEASURE_BETWEEN.fullmatch(q[e:at]) for e in ends) for at, _ in places)


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


def _in_percent(sides: list[_Side]) -> bool:
    """Some table's item is measured in percent (its unit, as the one-table rules read it: analyze/rules.py)."""
    for side in sides:
        for item in side.each_item or ([side.item] if side.item else []):
            if (side.index.measure_of(ITEM_AXIS, item) or (None,))[0] in ("%", "퍼센트"):
                return True
    return False


def _operation(question: str, sides: list[_Side], several_periods: bool = False,
               aggregated: bool = False) -> tuple[Op, list[_Side], float | None, bool]:
    """(what is computed, the tables in the order the formula reads them, the per scale, whether that order is the
    question's: False for a difference with no reference named, which is larger minus smaller).

    several_periods: some table is asked at two periods or more. A change asked then (shared/patterns.py
    CHANGE_ASKED, RATE_OF_CHANGE: the words the one-table rules read too) is a change per table between the first
    and the last period ("rate" or "change"), unless the question computes between the tables (~당, 몇 배).
    aggregated: the periods are a range summed per table; a sum word then names that sum, not a sum of the tables,
    when the question also compares the two ("합은 ~ 합보다 많아")."""
    q = _masked(question, sides)
    order = _ordered(sides)
    between = (len(sides) == 2 and (bool(PER_UNIT.search(question)) or bool(MULTI_RATIO.search(q))))
    if several_periods and not between and CHANGE_ASKED.search(q):
        return ("rate" if RATE_OF_CHANGE.search(q) else "change"), order, None, True
    if len(sides) == 2:
        if (m := PER_UNIT.search(question)) and q[len(squash(question[: m.end()])) - 1] != "#":
            at = len(squash(question[: m.start()]))  # "혼인 1건당 출생아 수": the one named before 당 is the denominator
            denominator = _before(sides, at + 1) or order[-1]
            numerator = next(s for s in sides if s is not denominator)
            scale = per_scale(m)
            return ("ratio" if scale == 1 else "per"), [numerator, denominator], scale, True
        # loop 12: "출생아는 사망자보다 몇 퍼센트 많아": the difference as a percent of the reference, read as the one-table
        # rules read it (shared asks_relative: a percent and a word of more / less, no difference named, no ratio, no
        # change), before the percent is read as a share. The reference is the table named before the word naming it
        # (보다, 대비, 에 비해). Not of values already in %, where a percent may be a difference in points.
        if asks_relative(q) and not _in_percent(sides):
            if (m := AGAINST.search(q)) and (reference := _before(sides, m.start())) is not None:
                return "relative", [next(s for s in sides if s is not reference), reference], None, True
        if MULTI_SHARE.search(q):
            whole = _whole(order, q)
            return "share", [next(s for s in sides if s is not whole), whole], None, True
        if MULTI_RATIO.search(q):
            whole = _whole(order, q)
            return "ratio", [next(s for s in sides if s is not whole), whole], None, True
    # loop 9: a word of the whole right before 몇 + a counting word ("출생아와 사망자는 모두 몇 명") adds the tables'
    # values as a sum word does (the same words the one-table rules read: shared TOTAL_AMOUNT)
    summed = SUM_WORDS.search(q) or (not several_periods and TOTAL_AMOUNT.search(q))
    if summed and not (aggregated and len(sides) == 2 and MULTI_DIFF.search(SUM_WORDS.sub("", q))):
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


def _fixed_sets(side: _Side) -> list[dict[str, str]]:
    """One code per category, and every combination of the codes a category names several of (in question order):
    the targets a list shows, or the targets a calculation is repeated for (loop 2, _repeated)."""
    codes = dict(side.plan.codes)
    if side.each_item:
        codes[ITEM_AXIS] = list(side.each_item)  # loop 12: the items asked for each (_assign_items)
    elif side.item is None:
        if ITEM_AXIS in side.index.axes and len(side.index.axes[ITEM_AXIS].labels) > 1:
            raise _Open(f"{side.tbl_id}: 항목")
    else:
        codes[ITEM_AXIS] = [side.item]
    several = {n: v for n, v in codes.items() if len(v) > 1 and (n != ITEM_AXIS or side.each_item)}
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


def _aligned(counts: list[int], what: str) -> int:
    """How many times a calculation is repeated: every table names one or the same number n of targets (periods)."""
    n = max(counts)
    if any(c not in (1, n) for c in counts):
        raise _Open(f"표마다 {what} 수가 다름 {counts}")
    return n


def _combined(op: Op, ids: list[str], scale: float | None, directed: bool, tag: str) -> tuple[list, str, str]:
    """(steps, result id, present) of one calculation between the tables' values ids (formula order)."""
    if op == "sum":
        steps, last = [], ids[0]
        for n, other in enumerate(ids[1:], start=1):
            steps.append(CombineStep(id=f"s{n}{tag}", a=last, b=other, fn="sum", note="합계"))
            last = f"s{n}{tag}"
        return steps, last, "compare"
    if len(ids) != 2:
        raise _Open(f"표 {len(ids)}개의 {op}")
    a, b = ids
    if op == "diff":
        orient = "fixed" if directed else "larger_first"
        return [CombineStep(id=f"d{tag}", a=a, b=b, fn="diff", orient=orient, note="차이")], f"d{tag}", "compare"
    if op == "share":
        return [CombineStep(id=f"r{tag}", a=a, b=b, fn="share", note="비중")], f"r{tag}", "share"
    if op == "per":
        return [CombineStep(id=f"r{tag}", a=a, b=b, fn="per", scale=scale, note=f"{b} {scale:g}당")], f"r{tag}", "compare"
    if op == "relative":  # loop 12: (subject - reference) / reference * 100, as one table's build.relative
        return [CombineStep(id=f"q{tag}", a=a, b=b, fn="pct_change", note="기준 대비 상대 차이")], f"q{tag}", "compare"
    return [CombineStep(id=f"r{tag}", a=a, b=b, fn="ratio", note="비율")], f"r{tag}", "compare"


def _changes(op: Op, order: list[_Side], fixed: dict, asked: dict, targets: int) -> Built:
    """A change per table (and per named target) between its first and its last asked period: the late value minus
    the early one, or its rate (pct_change). Shown: the two cells and the change, table by table."""
    if any(len(asked[id(s)]) < 2 for s in order):
        raise _Open("변화를 볼 두 시점이 없는 표")
    steps: list = []
    show: list[str] = []
    for n, side in enumerate(order):
        sets = fixed[id(side)]
        ends = [min(asked[id(side)]), max(asked[id(side)])]
        refs = _cells(side, [sets[k % len(sets)] for k in range(targets)], ends)
        for k in range(targets):
            tag = f"{n}_{k}"
            early, late = refs[2 * k], refs[2 * k + 1]
            steps += [SelectStep(id=f"e{tag}", cells=[early], table=side.tbl_id),
                      SelectStep(id=f"l{tag}", cells=[late], table=side.tbl_id)]
            if op == "rate":
                steps.append(CombineStep(id=f"c{tag}", a=f"l{tag}", b=f"e{tag}", fn="pct_change", note="증감률"))
            else:
                steps.append(CombineStep(id=f"c{tag}", a=f"l{tag}", b=f"e{tag}", fn="diff", note="증감"))
            show += [f"e{tag}", f"l{tag}", f"c{tag}"]
    return build.written(PlanV3(steps=steps, show=show, present="trend"))


def _between_change(question: str, sides: list[_Side], several_periods: bool,
                    aggregated: bool) -> tuple[Op, list[_Side], float | None, bool, str] | None:
    """Loop 18: (what is computed, the tables in formula order, the per scale, whether that order is the question's,
    the change: "diff" or "pct_change") when a calculation between two tables is asked at two periods together with
    how it changed, else None: the calculation at each period and the later result against the earlier one.

    The calculation between the tables is read as _operation reads it: a value per unit of another or a ratio
    (PER_UNIT, MULTI_RATIO) with a change word (CHANGE_ASKED, GAP_CHANGE); a gap named by its noun with a word of how
    it changed (shared GAP_WORDS and GAP_CHANGE, the words one table's gap change reads: analyze/rules.py _two_by_two);
    or a table named right before a comparing word (보다, 대비, 에 비해) and a word of more or less after it, with a
    change word. A comparing word right after a period compares the periods: each table's change, as before. Not over
    a range summed or averaged per table (a sum or an average word)."""
    if not several_periods or len(sides) != 2:
        return None
    q = _masked(question, sides)
    if aggregated and (SUM_WORDS.search(q) or AVG_WORDS.search(q)):
        return None
    if not (CHANGE_ASKED.search(q) or GAP_CHANGE.search(q)):
        return None
    rate = "pct_change" if RATE_OF_CHANGE.search(q) else "diff"
    if PER_UNIT.search(question) or MULTI_RATIO.search(q):
        op, order, scale, directed = _operation(question, sides)
        return (op, order, scale, directed, rate) if op in ("per", "ratio") else None
    for m in AGAINST.finditer(q):
        reference = _before(sides, m.start())
        if reference is None or reference.item is None or not MULTI_DIFF.search(q[m.end():]):
            continue
        ends = [b for _, b in reference.items.get(reference.item, ([], True))[0] if b <= m.start()]
        if any(_JOINED.fullmatch(q[b:m.start()]) for b in ends):
            subject = next(s for s in sides if s is not reference)
            fewer = bool(MULTI_FEWER.search(q[m.end():]))
            return "diff", ([reference, subject] if fewer else [subject, reference]), None, True, "diff"
    if GAP_WORDS.search(q) and GAP_CHANGE.search(q):
        return "diff", _ordered(sides), None, False, "diff"
    return None


# Loop 18: what may stand between a table's name and the comparing word after it: a particle or a counting word.
_JOINED = re.compile(r"[가-힣]{0,2}")


def _repeated(op: Op, order: list[_Side], fixed: dict, asked: dict, scale: float | None, directed: bool,
              aggregate: str | None, change: str | None = None) -> Built:
    """The calculation between the tables once per named target and, unless the periods are aggregated, once per
    named period (loop 2): "서울과 부산의 혼인 1건당 출생아 수", "2020년과 2023년 ~ 차이". With aggregate ("sum" or
    "avg"), each table's cells over the asked range are aggregated first and the calculation is made on the results
    ("2019~2021년 출생아 수 합계를 혼인 건수 합계로 나누면"). One group of one target and one period is the plan of
    loop 1 (ids t0, t1, then d / r / s1).

    Loop 18, change ("diff" or "pct_change", _between_change): asked holds each table's earliest and latest period;
    per target, the later period's result against the earlier one's is added (ids c_k). A difference with no
    reference named is taken in one direction at both periods: the table larger at the later period first."""
    targets = _aligned([len(fixed[id(s)]) for s in order], "대상")
    periods = 1 if aggregate else _aligned([len(asked[id(s)]) for s in order], "시점")
    if aggregate and op == "share":
        raise _Open("기간 합의 비중")  # compute's share takes cells as the whole, not an aggregate
    if change and periods != 2:
        raise _Open("변화를 볼 두 시점이 없는 표")
    groups = [(k, t) for k in range(targets) for t in range(periods)]
    one = len(groups) == 1
    per_side: dict[int, list[list[str]]] = {}
    values: dict[int, list[float | None]] = {}
    for side in order:
        sets, mine = fixed[id(side)], asked[id(side)]
        wanted = [(sets[k % len(sets)], mine if aggregate else [mine[t % len(mine)]]) for k, t in groups]
        pairs = [(f, p) for f, ps in wanted for p in ps]
        found = _lookup(side.res.table).cells(pairs)
        missing = [f"{side.tbl_id} {p}" for (f, p), c in zip(pairs, found) if c is None]
        if missing:
            raise _Open(f"칸 없음: {', '.join(missing[:3])}")
        refs = iter(c.source_ref for c in found)
        per_side[id(side)] = [[next(refs) for _ in ps] for _, ps in wanted]
        values[id(side)] = [c.value for c in found]
    if change and op == "diff" and not directed:
        late = groups.index((0, 1))  # the first target at the later period: its larger table goes first
        a, b = (values[id(s)][late] for s in order)
        if a is not None and b is not None and a < b:
            order = [order[1], order[0]]
        directed = True
    steps: list = []
    show: list[str] = []
    present = "compare"
    results: dict[tuple[int, int], str] = {}
    for g in range(len(groups)):
        tag = "" if one else f"_{g}"
        ids = []
        for n, side in enumerate(order):
            sid = f"t{n}{tag}"
            refs = per_side[id(side)][g]
            steps.append(SelectStep(id=sid, cells=refs, table=side.tbl_id))
            if aggregate and len(refs) >= 2:
                steps.append(AggregateStep(id=f"a{n}{tag}", of=sid, fn=aggregate))
                sid = f"a{n}{tag}"
            elif len(refs) != 1:
                raise _Open(f"{side.tbl_id}: 합할 시점 부족")
            ids.append(sid)
        more, result, present = _combined(op, ids, scale, directed, tag)
        steps += more
        results[groups[g]] = result
        cells = [f"t{n}{tag}" for n in range(len(order))]
        show += [*cells, *[i for i in ids if i not in cells], result]
    if change:
        for k in range(targets):
            note = "증감률" if change == "pct_change" else "변화"
            steps.append(CombineStep(id=f"c_{k}", a=results[(k, 1)], b=results[(k, 0)], fn=change, note=note))
            show.append(f"c_{k}")
    return build.written(PlanV3(steps=steps, show=show, present=present))


def _split_periods(sides: list[_Side]) -> dict[int, list[str]] | None:
    """Loop 6: the asked periods of one metric kept in one table per period (a survey's tables of two survey years):
    each table's own asked periods, when every table lacks a period the question names that another table holds, no
    period is in two tables, and the tables hold the same metric (the same item name and the same units). None
    otherwise (the periods stay as each table reads them: a period a table lacks is left open)."""
    if len(sides) < 2 or not all(s.plan.missing_years and s.plan.periods for s in sides):
        return None
    held = [{p[:4] for p in s.plan.periods} for s in sides]
    if any(a & b for n, a in enumerate(held) for b in held[n + 1:]):
        return None
    for n, side in enumerate(sides):
        others = set().union(*(h for k, h in enumerate(held) if k != n))
        if not set(side.plan.missing_years) <= others:
            return None
    if any(s.item is None for s in sides) or len({tuple(s.index.units) for s in sides}) != 1:
        return None
    forms = [label_forms(s.index.axes[ITEM_AXIS].labels[s.item])[0] for s in sides]
    if not set.intersection(*forms):  # the item names share no form: two metrics
        return None
    return {id(s): list(s.plan.periods) for s in sides}


def _split_plan(question: str, sides: list[_Side], fixed: dict, asked: dict) -> Built:
    """Loop 6: one metric whose periods sit in different tables (_split_periods): the cells in period order, or, when
    a change is asked (a change or comparing word, the one-table words), the latest cell against the earliest one: its
    rate of change when a rate is asked (RATE_OF_CHANGE), its amount when an amount is asked or no rate is."""
    if any(len(fixed[id(s)]) != 1 or len(asked[id(s)]) != 1 for s in sides):
        raise _Open("표마다 나눈 시점의 대상·시점이 여럿")
    order = sorted(sides, key=lambda s: asked[id(s)][0])
    refs = [_cells(s, fixed[id(s)], asked[id(s)])[0] for s in order]
    steps: list = [SelectStep(id=f"t{n}", cells=[r], table=s.tbl_id) for n, (s, r) in enumerate(zip(order, refs))]
    show = [f"t{n}" for n in range(len(order))]
    q = _masked(question, sides)
    if not (CHANGE_ASKED.search(q) or MULTI_DIFF.search(q)):
        return build.written(PlanV3(steps=steps, show=show, present="value"))
    early, late = show[0], show[-1]
    rate = bool(RATE_OF_CHANGE.search(q))
    if not rate or ASKS_AMOUNT.search(q):
        steps.append(CombineStep(id="d", a=late, b=early, fn="diff", note="증감"))
        show.append("d")
    if rate:
        steps.append(CombineStep(id="g", a=late, b=early, fn="pct_change", note="증감률"))
        show.append("g")
    return build.written(PlanV3(steps=steps, show=show, present="trend"))


def _plan(question: str, res: MultiResources) -> Built:
    sides = [_Side(r, question) for r in res.tables]
    _assign_items(sides, question)
    fixed = {id(s): _fixed_sets(s) for s in sides}
    if (split := _split_periods(sides)) is not None:
        return _split_plan(question, sides, fixed, split)
    asked = {id(s): _asked_periods(s) for s in sides}
    if any(asked.values()) and not all(asked.values()):
        raise _Open("질문한 시점이 없는 표")
    if not any(asked.values()):
        latest = _common_period(sides, fixed)
        asked = {id(s): [latest] for s in sides}
    several = any(len(p) > 1 for p in asked.values())
    # a range of periods (2019~2021년, a decade, a half) is summed per table, or averaged when an average is asked;
    # periods named one by one, or a range asked period by period (연도별, 해마다), repeat the calculation instead
    periods = asked_periods(question)
    q = _masked(question, sides)
    aggregate = None
    if several and periods.ranges and not EACH_PERIOD.search(question):
        aggregate = "avg" if AVG_WORDS.search(q) else "sum"
    if all(len(set(p)) >= 2 for p in asked.values()) and (
            between := _between_change(question, sides, several, aggregate is not None)) is not None:
        op, order, scale, directed, change = between  # loop 18: the calculation at two periods and its change
        ends = {id(s): [min(asked[id(s)]), max(asked[id(s)])] for s in sides}
        return _repeated(op, order, fixed, ends, scale, directed, None, change)
    op, order, scale, directed = _operation(question, sides, several, aggregate is not None)
    if op == "list":
        steps: list = []
        ids: list[str] = []
        for n, side in enumerate(order):
            refs = _cells(side, fixed[id(side)], asked[id(side)])
            steps.append(SelectStep(id=f"t{n}", cells=refs, table=side.tbl_id))
            ids.append(f"t{n}")
        return build.written(PlanV3(steps=steps, show=ids, present="value"))
    if op in ("rate", "change"):
        return _changes(op, order, fixed, asked, _aligned([len(fixed[id(s)]) for s in order], "대상"))
    return _repeated(op, order, fixed, asked, scale, directed, aggregate)


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
