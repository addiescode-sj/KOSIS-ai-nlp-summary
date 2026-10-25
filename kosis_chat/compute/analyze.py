"""Question -> what to compute (T1-06), rules first: the judge's parser finds the periods and the labels the question
names; this module adds the question type, fills the categories the question leaves out (a total, or the only
choice) and builds the CalcRequest. Whatever the rules cannot decide is listed in Plan.unresolved for the LLM step.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, replace

from kosis_chat.compute.request import CalcRequest
from kosis_chat.compute.roles import is_province, pick_of, reference_index
from kosis_chat.data.schema import Cell, TableInput
from kosis_chat.judge.answerability import resolve
from kosis_chat.judge.question import CHANGE_WORDS, ITEM_AXIS, TREND_WORDS, Periods, TableIndex, is_total, parse, squash

from kosis_ai.schemas import QuestionType, StatQuery  # isort: skip

PERIOD_AXIS = "시점"  # Plan.over when an extreme question ranks periods ("가장 낮았던 해")
RECENT_POINTS = 5
EXTREME_WORDS = re.compile(r"가장|제일|최고|최저|최대|최소|1위|꼴찌")
MOST_RECENT = re.compile(r"(?:가장|제일)\s*최근")  # "가장 최근 ~" names a period, not a ranking
SHARE_WORDS = re.compile(r"비중|차지|구성비")
COMPARE_WORDS = re.compile(r"비교|차이|몇\s*배|보다|더\s*(?:높|낮|많|적|크|작)")
ASKS_PERIOD = re.compile(r"해는|해가|연도|년도|언제|시점은")
_LATEST = re.compile(r"latest[^(]*\((\d{4,6})")
_RECENT = re.compile(r"recent (\d+) points \((\d{4,6})~(\d{4,6})\)")
# "3인 이하", "5인 이상", "65세 이상": a range over numbered labels of one category.
_NUM_RANGE = re.compile(r"(\d+)\s*(인|세)\s*(이하|이상|미만|초과)")
_NUM_LABEL = re.compile(r"^(\d+)\s*(인|세)\s*(이상)?(?:\s*\(.*\))?$")
_BOTH_SEXES = re.compile(r"남녀|남성과\s*여성|남자와\s*여자|성별")


@dataclass
class Plan:
    question: str
    type: QuestionType | None
    codes: dict[str, list[str]]  # axis -> named codes ("항목" for items), in the order the question names them
    periods: list[str]  # table periods asked (all of a range), [] when the question names none
    over: str | None = None  # extreme: the axis ranked over, or PERIOD_AXIS
    pick: str | None = None
    missing_years: list[str] = field(default_factory=list)  # asked years the table does not have
    default_applied: dict[str, str] = field(default_factory=dict)
    unresolved: list[str] = field(default_factory=list)


class Unresolved(Exception):
    def __init__(self, what: list[str]) -> None:
        super().__init__(", ".join(what))
        self.what = what


# ---------------------------------------------------------------- rules: question -> plan


def _question_type(question: str, periods: Periods, codes: dict[str, list[str]]) -> QuestionType:
    if EXTREME_WORDS.search(MOST_RECENT.sub(" ", question)):
        return "extreme"
    if SHARE_WORDS.search(question):
        return "share"
    if periods.ranges or len(periods.singles) >= 2 or (periods.recent and TREND_WORDS.search(question)):
        return "trend"
    if TREND_WORDS.search(question) or CHANGE_WORDS.search(question):
        return "trend"
    if COMPARE_WORDS.search(question) or any(len(v) >= 2 for v in codes.values()):
        return "compare"
    return "value"


def _number_range(question: str, index: TableIndex) -> tuple[str, list[str]] | None:
    """'3인 이하' -> (axis, [1인, 2인, 3인] codes); '5인 이상' -> 5인 + the smallest 'K인 이상' above (no double count)."""
    m = _NUM_RANGE.search(question)
    if not m:
        return None
    n, unit, how = int(m.group(1)), m.group(2), m.group(3)
    for name, axis in index.axes.items():
        numbered = {}
        for code, label in axis.labels.items():
            if (lm := _NUM_LABEL.match(label.strip())) and lm.group(2) == unit:
                numbered[code] = (int(lm.group(1)), bool(lm.group(3)))
        if len(numbered) < 2:
            continue
        exact = {code: k for code, (k, open_) in numbered.items() if not open_}
        opened = {code: k for code, (k, open_) in numbered.items() if open_}
        if how in ("이하", "미만"):
            top = n if how == "이하" else n - 1
            return name, sorted((c for c, k in exact.items() if k <= top), key=exact.get)
        low = n if how == "이상" else n + 1
        tail = min(((k, c) for c, k in opened.items() if k > low), default=None)
        end = tail[0] if tail else max(exact.values()) + 1
        picked = sorted((c for c, k in exact.items() if low <= k < end), key=exact.get)
        return name, picked + ([tail[1]] if tail else [])
    return None


def _both_sexes(question: str, index: TableIndex, codes: dict[str, list[str]]) -> None:
    if not _BOTH_SEXES.search(question):
        return
    for name, axis in index.axes.items():
        pair = [c for c, label in axis.labels.items() if label.strip() in ("남자", "여자")]
        if len(pair) == 2 and len(codes.get(name, [])) < 2:
            codes[name] = sorted(pair, key=lambda c: axis.labels[c] != "남자")


def _drop_children(index: TableIndex, codes: dict[str, list[str]]) -> None:
    """'기대수명' matches '기대수명(...)' and '기대수명(...) - 남' / '- 여': keep the label the others extend."""
    for name, named in codes.items():
        labels = {c: index.axes[name].labels[c].strip() for c in named}
        roots = [c for c in named if all(labels[o].startswith(labels[c]) for o in named)]
        if len(named) > 1 and len(roots) == 1:
            codes[name] = roots


def _order_in_question(question: str, index: TableIndex, name: str, named: list[str]) -> list[str]:
    q = squash(question)

    def at(code: str) -> int:
        label = index.axes[name].labels[code]
        forms = [squash(label), squash(re.sub(r"\s*\(.*\)\s*$", "", label)), squash(re.sub(r"(특별자치시|특별자치도|특별시|광역시|도)$", "", label))]
        found = [q.find(f) for f in forms if f and q.find(f) >= 0]
        return min(found) if found else len(q)

    return sorted(named, key=at)


def _periods(p: Periods, index: TableIndex, default: dict[str, str]) -> tuple[list[str], list[str]]:
    """(table periods asked, asked years the table lacks)."""
    years = sorted({x[:4] for x in index.periods})
    if p.ranges:
        a, b = p.ranges[0]
        b = years[-1] if b == "9999" else b
        found = [x for x in index.periods if a <= x[:4] <= b]
        # Only the asked ends: a census table has no 2016~2019 by design, which is not worth a note.
        return found, [y for y in dict.fromkeys((a, b)) if y not in years]
    if p.singles:
        found = [x for s in p.singles for x in index.periods if x == s or (len(s) == 4 and x[:4] == s)]
        return list(dict.fromkeys(found)), [s for s in p.singles if not any(x[:4] == s[:4] for x in index.periods)]
    period = default.get("period", "")
    if m := _RECENT.search(period):
        return [x for x in index.periods if m.group(2) <= x <= m.group(3)], []
    if m := _LATEST.search(period):
        return [m.group(1)], []
    return [], []


def rule_plan(question: str, index: TableIndex, default_applied: dict[str, str] | None = None) -> Plan:
    parsed = parse(index, question)
    resolved = resolve(index, parsed)
    codes = {name: _order_in_question(question, index, name, sorted(cs)) for name, cs in resolved.codes.items()}
    if found := _number_range(question, index):
        codes[found[0]] = found[1]
    _both_sexes(question, index, codes)
    _drop_children(index, codes)
    default = dict(default_applied or {})
    periods, missing = _periods(parsed.periods, index, default)
    kind = _question_type(question, parsed.periods, codes)
    plan = Plan(question, kind, codes, periods, missing_years=missing, default_applied=default)
    if kind == "extreme":
        plan.pick = pick_of(question)
        plan.over = PERIOD_AXIS if ASKS_PERIOD.search(question) else _ranked_axis(index, codes)
        if plan.over is None:
            plan.unresolved.append("over")
    return plan


def _ranked_axis(index: TableIndex, codes: dict[str, list[str]]) -> str | None:
    """The category an extreme question ranks: the regional one it does not pin down, else any other open one."""
    open_axes = [n for n, a in index.axes.items() if n != ITEM_AXIS and len(a.labels) > 1 and len(codes.get(n, [])) != 1]
    regional = [n for n in open_axes if index.axes[n].regional]
    return (regional or open_axes or [None])[0]


# ---------------------------------------------------------------- plan -> request


def _total(index: TableIndex, name: str) -> str | None:
    return next((c for c, label in index.axes[name].labels.items() if is_total(label)), None)


class _Lookup:
    def __init__(self, table: TableInput) -> None:
        self.cells: dict[tuple, Cell] = {}
        self.series: dict[tuple, list[Cell]] = {}
        for c in table.cells:
            key = (c.item_code, tuple(sorted(c.dim_codes.items())))
            self.cells[(*key, c.period_raw)] = c
            self.series.setdefault(key, []).append(c)
        for cells in self.series.values():
            cells.sort(key=lambda c: c.period_raw)

    @staticmethod
    def key(fixed: dict[str, str]) -> tuple:
        return fixed.get(ITEM_AXIS, ""), tuple(sorted((k, v) for k, v in fixed.items() if k != ITEM_AXIS))

    def cell(self, fixed: dict[str, str], period: str) -> Cell | None:
        return self.cells.get((*self.key(fixed), period))

    def numeric_periods(self, fixed: dict[str, str]) -> list[str]:
        return [c.period_raw for c in self.series.get(self.key(fixed), []) if c.value_kind == "ok"]

    def all_periods(self, fixed: dict[str, str]) -> list[str]:
        return [c.period_raw for c in self.series.get(self.key(fixed), [])]


_LOOKUPS: dict[int, tuple[TableInput, _Lookup]] = {}


def _lookup(table: TableInput) -> _Lookup:
    cached = _LOOKUPS.get(id(table))
    if cached is None or cached[0] is not table:
        if len(_LOOKUPS) > 8:
            _LOOKUPS.clear()
        cached = (table, _Lookup(table))
        _LOOKUPS[id(table)] = cached
    return cached[1]


def _fixed(plan: Plan, index: TableIndex, free: set[str], problems: list[str]) -> dict[str, str]:
    """One code per category not in `free`: the named one, else the total, else the only one."""
    out: dict[str, str] = {}
    for name, axis in index.axes.items():
        if name in free:
            continue
        named = plan.codes.get(name, [])
        if len(named) == 1:
            out[name] = named[0]
        elif len(named) > 1:
            problems.append(f"{name}: {len(named)}개 중 하나")
        elif (total := _total(index, name)) is not None:
            out[name] = total
        elif len(axis.labels) == 1:
            out[name] = next(iter(axis.labels))
        else:
            problems.append(name)
    return out


def _one_period(plan: Plan, lookup: _Lookup, fixed_sets: list[dict[str, str]]) -> str | None:
    if plan.periods:
        return plan.periods[-1]
    shared = None
    for fixed in fixed_sets:
        found = set(lookup.numeric_periods(fixed))
        shared = found if shared is None else shared & found
    return max(shared) if shared else None


def _with(fixed: dict[str, str], name: str, code: str) -> dict[str, str]:
    return {**fixed, name: code}


def _ref(lookup: _Lookup, fixed: dict[str, str], period: str, problems: list[str]) -> str:
    cell = lookup.cell(fixed, period)
    if cell is None:
        problems.append(f"칸 없음 {fixed} {period}")
        return ""
    return cell.source_ref


def build_request(plan: Plan, index: TableIndex, table: TableInput) -> CalcRequest:
    """CalcRequest for the plan, or Unresolved naming what is missing (for the LLM step)."""
    if plan.unresolved:
        raise Unresolved(plan.unresolved)
    lookup = _lookup(table)
    problems: list[str] = []
    note = f"{'·'.join(plan.missing_years)}년은 표에 없음" if plan.missing_years else None
    kind = plan.type
    if kind in ("value", "trend"):
        fixed = _fixed(plan, index, set(), problems)
        if problems:
            raise Unresolved(problems)
        if kind == "trend":
            # One asked period ("2023년 얼마나 늘었나"): the recent points that end there.
            numeric = [p for p in lookup.numeric_periods(fixed) if not plan.periods or p <= plan.periods[-1]]
            periods = plan.periods if len(plan.periods) >= 2 else numeric[-RECENT_POINTS:]
            refs = [r for p in periods if (r := _ref(lookup, fixed, p, []))]
            if len(refs) < 2:
                raise Unresolved([f"추이 시점 부족 {periods}"])
            return CalcRequest(type="trend", cells=refs, note=note)
        period = _one_period(plan, lookup, [fixed])
        ref = _ref(lookup, fixed, period or "", problems)
        if problems:
            raise Unresolved(problems)
        newer = [p for p in lookup.all_periods(fixed) if period and p > period] if plan.default_applied.get("period") else []
        return CalcRequest(type="value", cells=[ref], context=[_ref(lookup, fixed, p, []) for p in newer], note=note)
    if kind == "compare":
        side = next((n for n, named in plan.codes.items() if len(named) == 2), None)
        if side is None:
            raise Unresolved(["비교할 두 대상"])
        fixed = _fixed(plan, index, {side}, problems)
        if problems:
            raise Unresolved(problems)
        pair = [_with(fixed, side, c) for c in plan.codes[side]]
        if len(plan.periods) >= 2 and all(len(v) <= 1 for n, v in plan.codes.items() if n != side):
            pass  # two places and two years: compare at the later one
        period = _one_period(plan, lookup, pair)
        refs = [_ref(lookup, f, period or "", problems) for f in pair]
        if problems:
            raise Unresolved(problems)
        labels = [index.axes[side].labels[c] for c in plan.codes[side]]
        return CalcRequest(type="compare", cells=refs, reference=reference_index(plan.question, labels), note=note)
    if kind == "share":
        return _share_request(plan, index, lookup, note)
    if kind == "extreme":
        return _extreme_request(plan, index, lookup, note)
    raise Unresolved(["질문 유형"])


def _share_axis(plan: Plan, index: TableIndex) -> tuple[str, list[str], str] | None:
    """(axis, part codes, total code): items first ('1인 가구 비중'), then a category ('동부 가구 비중')."""
    order = [ITEM_AXIS] + [n for n in plan.codes if n != ITEM_AXIS]
    for name in order:
        total = _total(index, name) if name in index.axes else None
        parts = [c for c in plan.codes.get(name, []) if c != total]
        if total and parts:
            return name, parts, total
    return None


def _share_request(plan: Plan, index: TableIndex, lookup: _Lookup, note: str | None) -> CalcRequest:
    found = _share_axis(plan, index)
    if found is None:
        raise Unresolved(["비중의 분자·분모"])
    name, parts, total = found
    problems: list[str] = []
    fixed = _fixed(replace(plan, codes={k: v for k, v in plan.codes.items() if k != name}), index, {name}, problems)
    if problems:
        raise Unresolved(problems)
    sets = [_with(fixed, name, c) for c in [*parts, total]]
    period = _one_period(plan, lookup, sets)
    refs = [_ref(lookup, f, period or "", problems) for f in sets]
    if problems:
        raise Unresolved(problems)
    return CalcRequest(type="share", cells=refs[:-1], total=refs[-1], note=note)


def _extreme_request(plan: Plan, index: TableIndex, lookup: _Lookup, note: str | None) -> CalcRequest:
    if plan.over is None or (plan.over != PERIOD_AXIS and plan.over not in index.axes):
        raise Unresolved(["over"])  # the model fill can leave it open too
    share = bool(SHARE_WORDS.search(plan.question))
    problems: list[str] = []
    item_total = _total(index, ITEM_AXIS) if share else None
    codes = dict(plan.codes)
    if share and item_total:
        codes[ITEM_AXIS] = [c for c in codes.get(ITEM_AXIS, []) if c != item_total]
    if plan.over == PERIOD_AXIS:
        fixed = _fixed(replace(plan, codes=codes), index, set(), problems)
        if problems:
            raise Unresolved(problems)
        periods = plan.periods or lookup.all_periods(fixed)
        groups = [(fixed, p) for p in periods]
        context_sets: list[tuple[dict, str]] = []
    else:
        name = plan.over
        fixed = _fixed(replace(plan, codes={k: v for k, v in codes.items() if k != name}), index, {name}, problems)
        if problems:
            raise Unresolved(problems)
        axis = index.axes[name]
        pool = [c for c, label in axis.labels.items() if axis.parents.get(c) is None and not is_total(label)]
        if axis.regional and any(is_province(axis.labels[c]) for c in pool):
            pool = [c for c in pool if is_province(axis.labels[c])]
        sets = [_with(fixed, name, c) for c in pool]
        period = _one_period(plan, lookup, sets[:1])
        groups = [(f, period or "") for f in sets]
        total = _total(index, name)
        context_sets = [(_with(fixed, name, total), period or "")] if total else []
    candidates, totals = [], []
    for f, p in groups:
        ref = _ref(lookup, f, p, [])
        if not ref:
            continue
        if share and item_total:
            denominator = _ref(lookup, _with(f, ITEM_AXIS, item_total), p, [])
            if not denominator:
                continue
            totals.append(denominator)
        candidates.append(ref)
    if len(candidates) < 2:
        raise Unresolved(["비교 대상 부족"])
    context = [r for f, p in context_sets if (r := _ref(lookup, f, p, []))]
    return CalcRequest(type="extreme", candidates=candidates, candidate_totals=totals, pick=plan.pick, context=context,
                       note=note)


# ---------------------------------------------------------------- plan -> StatQuery (the record in labels)


def to_query(plan: Plan, index: TableIndex, request: CalcRequest | None = None) -> StatQuery:
    def labels(name: str) -> list[str]:
        return [index.axes[name].labels[c].strip() for c in plan.codes.get(name, []) if c in index.axes[name].labels]

    items = labels(ITEM_AXIS) if ITEM_AXIS in index.axes else []
    side = next((n for n, named in plan.codes.items() if len(named) == 2), None) if plan.type == "compare" else None
    filters = {n: labels(n)[0] for n in plan.codes if n not in (ITEM_AXIS, side) and len(labels(n)) == 1}
    over = {}
    if plan.type == "extreme" and plan.over and plan.over != PERIOD_AXIS:
        over = {plan.over: []}
    return StatQuery(
        question=plan.question,
        type=plan.type,
        item=items[0] if items else None,
        items=items if len(items) > 1 else [],
        periods=plan.periods,
        filters=filters,
        targets=[{side: label} for label in labels(side)] if side else [],
        reference=request.reference if request is not None else None,
        over=over,
        pick=plan.pick,
    )
