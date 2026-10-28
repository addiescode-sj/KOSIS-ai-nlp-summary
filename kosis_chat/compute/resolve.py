"""Periods and targets as a plan names them -> table cells (T0-04 S4).

A select step may name its cells by meaning: an item and categories (TargetSpec: codes, children of a code, the
total, or a label path) and periods (PeriodSpec: a list, a range, the last n with a number, an event or period moved
by an offset, a part of a year). resolve_select turns that into source_refs, in period order, with cautions for what
it filled in (a category the plan left out is set to its total, an asked year the table lacks). It never guesses:
a label that names several codes raises NeedsClarify with the candidates, an event missing from the calendar is
asked of lookup (the model, wired in S5) or raises NeedsClarify.
Period strings follow the Parquet files: 'YYYY' a year, 'YYYYMM' a month (a monthly table may also hold years).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from itertools import product

from kosis_chat.compute.engine import CalcRefused
from kosis_chat.compute.semantics import _whole_label, index_of
from kosis_chat.data.events import event_key as _key
from kosis_chat.data.events import load_events
from kosis_chat.data.schema import Cell, TableInput
from kosis_chat.judge import messages
from kosis_chat.judge.question import ITEM_AXIS, Axis

from kosis_ai.schemas import PeriodSpec, SelectStep, TargetSpec  # isort: skip

SPAN_MONTHS = {"year": (1, 12), "H1": (1, 6), "H2": (7, 12), "Q1": (1, 3), "Q2": (4, 6), "Q3": (7, 9), "Q4": (10, 12)}
CLARIFY_LIMIT = 20  # candidates offered when the plan leaves a category open

# An event name -> its start ('YYYY' or 'YYYY-MM'), or None when unknown (the model, from S5 on).
EventLookup = Callable[[str], "str | None"]


class NeedsClarify(Exception):
    """The plan names something several codes could be; options are (code, label) for the answer to offer."""

    def __init__(self, axis: str, options: list[tuple[str, str]], message: str) -> None:
        super().__init__(message)
        self.axis = axis
        self.options = options
        self.message = message


@dataclass
class Resolved:
    cells: list[str]
    cautions: list[str] = field(default_factory=list)


# ---------------------------------------------------------------- events


def event_start(name: str, lookup: EventLookup | None, cautions: list[str]) -> str:
    found = load_events().get(_key(name))
    if found:
        return found
    found = lookup(name) if lookup else None
    if not found:
        raise NeedsClarify("시점", [], f"'{name}'이(가) 언제인지 알 수 없습니다. 연도를 알려 주세요.")
    cautions.append(f"event_source: model ({name} = {found})")
    return found


# ---------------------------------------------------------------- targets


def _label_core(label: str) -> str:
    return label.split("(")[0].strip()


def resolve_target(axis: Axis, spec: TargetSpec) -> list[str]:
    if spec.codes:
        missing = [c for c in spec.codes if c not in axis.labels]
        if missing:
            raise CalcRefused("class_mismatch", messages.refusal_message("class_mismatch", what=f"{axis.name} {missing}"))
        return list(spec.codes)
    if spec.children_of is not None:
        children = [c for c, p in axis.parents.items() if p == spec.children_of]
        if not children:
            raise CalcRefused("class_mismatch", messages.refusal_message("class_mismatch", what=f"{axis.name} {spec.children_of}의 하위"))
        return children
    if spec.total:
        totals = _totals(axis)
        if not totals:
            raise CalcRefused("class_mismatch", messages.refusal_message("class_mismatch", what=f"{axis.name} 합계"))
        return totals[:1]
    return [_by_path(axis, spec.path)]


def _totals(axis: Axis) -> list[str]:
    """Codes whose label names the whole category, top level first; a '총…' label before 계/합계, since a table
    can hold both and then '합계' is a subtotal (TX_11025_A000_A 거주외국인별: 합계 = 외국인주민, 총인구 = everyone)."""
    found = [c for c, label in axis.labels.items() if _whole_label(label)]
    return sorted(found, key=lambda c: (axis.depth(c), not axis.labels[c].strip().startswith("총"), c))


def _by_path(axis: Axis, path: list[str]) -> str:
    last = _label_core(path[-1])
    found = [c for c, label in axis.labels.items() if _label_core(label) == last]
    for parent_label in path[:-1]:
        want = _label_core(parent_label)
        found = [c for c in found if any(_label_core(axis.labels[a]) == want for a in axis.ancestors(c))]
    if not found:
        raise CalcRefused("class_mismatch", messages.refusal_message("class_mismatch", what=" > ".join(path)))
    if len(found) > 1:
        options = [(c, " > ".join([*(axis.labels[a] for a in reversed(axis.ancestors(c))), axis.labels[c]])) for c in found]
        raise NeedsClarify(axis.name, options, f"'{' > '.join(path)}'이(가) 여러 곳입니다. 어느 것인지 골라 주세요.")
    return found[0]


def _default(axis: Axis, cautions: list[str]) -> list[str]:
    """A category the plan left out: its only code, else its total (with a caution), else ask."""
    if len(axis.labels) == 1:
        return list(axis.labels)
    totals = _totals(axis)
    if totals:
        cautions.append(f"default_applied: {axis.name} = {axis.labels[totals[0]]}")
        return totals[:1]
    top = [c for c in axis.labels if axis.parents.get(c) is None][:CLARIFY_LIMIT]
    raise NeedsClarify(axis.name, [(c, axis.labels[c]) for c in top], f"{axis.name}을(를) 골라 주세요.")


# ---------------------------------------------------------------- periods


def _months(year: str, first: int, last: int) -> list[str]:
    return [f"{year}{m:02d}" for m in range(first, last + 1)]


def _shift(base: str, offset: int, unit: str) -> str:
    """'2020' -1 year -> '2019'; '202001' -1 month -> '201912'."""
    if unit == "year":
        return str(int(base[:4]) + offset)
    y, m = int(base[:4]), int(base[4:6]) - 1 + offset
    return f"{y + m // 12}{m % 12 + 1:02d}"


def resolve_periods(spec: PeriodSpec, periods: list[str], period_type: str, lookup: EventLookup | None,
                    cautions: list[str]) -> list[str]:
    """Table periods for spec, ascending; recent is applied by the caller (it needs the cells)."""
    have = set(periods)
    monthly = any(len(p) == 6 for p in periods)
    if spec.kind == "list":
        want: list[str] = []
        for p in spec.periods:
            want += [p] if p in have or not (monthly and len(p) == 4) else _months(p, 1, 12)
    elif spec.kind == "range":
        start, end = spec.start, spec.end
        if monthly and len(start) == 4 and start not in have:
            start = f"{start}01"
        if monthly and len(end) == 4 and end not in have:
            end = f"{end}12"
        want = [p for p in periods if len(p) == len(start) and start <= p <= end]
        if not want:
            raise CalcRefused("no_period", messages.refusal_message("no_period", asked=f"{spec.start}~{spec.end}"))
    elif spec.kind == "recent":
        return sorted(periods)
    elif spec.kind == "anchor":
        unit = spec.unit or ("month" if period_type == "M" else "year")
        base = spec.period or event_start(spec.event, lookup, cautions).replace("-", "")
        if unit == "month" and len(base) == 4:
            base = f"{base}01"
        target = _shift(base, spec.offset, unit)
        size = 4 if unit == "year" else 6
        want = [p for p in periods if len(p) == size and p >= target] if spec.to_latest else [target]
    else:  # span
        if period_type in ("Q", "H"):
            raise CalcRefused("no_period", f"{spec.year}년 {spec.part}: 분기·반기 표의 시점 해석은 아직 지원하지 않습니다.")
        if not monthly:
            if spec.part != "year":
                raise CalcRefused("no_period", f"이 표는 연 단위라 {spec.year}년 {spec.part} 값이 없습니다.")
            want = [spec.year]
        else:
            want = _months(spec.year, *SPAN_MONTHS[spec.part])
    found = sorted(p for p in dict.fromkeys(want) if p in have)
    lacking = [p for p in want if p not in have]
    if not found:
        raise CalcRefused("no_period", messages.refusal_message("no_period", asked=", ".join(lacking[:6])))
    if lacking:
        cautions.append(f"missing_years: {', '.join(lacking)}")
    return found


# ---------------------------------------------------------------- select


_CELLS: dict[int, tuple[TableInput, dict[tuple, Cell]]] = {}


def _cells_by_key(table: TableInput) -> dict[tuple, Cell]:
    cached = _CELLS.get(id(table))
    if cached is None or cached[0] is not table:
        if len(_CELLS) > 8:
            _CELLS.clear()
        cached = (table, {(c.item_code, tuple(sorted(c.dim_codes.items())), c.period_raw): c for c in table.cells})
        _CELLS[id(table)] = cached
    return cached[1]


def resolve_select(table: TableInput, step: SelectStep, lookup: EventLookup | None = None,
                   star: dict[str, str] | None = None) -> Resolved:
    """source_refs of a select named by item / dims / periods; star fills a map's '*' code."""
    if step.table and step.table != table.meta.tbl_id:
        raise CalcRefused("out_of_scope", f"{step.table}: 이 질문의 표({table.meta.tbl_id})가 아닙니다(여러 표 결합은 아직 엔진만 지원).")
    index = index_of(table)
    cautions: list[str] = []
    codes: dict[str, list[str]] = {}
    for name, axis in index.axes.items():
        spec = step.item if name == ITEM_AXIS else step.dims.get(name)
        if spec is not None and spec.codes == ["*"]:
            if not star or name not in star:
                raise CalcRefused("class_mismatch", f"{name}의 '*'는 map 안에서만 쓸 수 있습니다.")
            spec = TargetSpec(codes=[star[name]])
        codes[name] = resolve_target(axis, spec) if spec is not None else _default(axis, cautions)
    unknown = [n for n in step.dims if n not in index.axes]
    if unknown:
        raise CalcRefused("class_mismatch", messages.refusal_message("class_mismatch", what=f"분류 {unknown}"))
    spec = step.periods or PeriodSpec(kind="recent", n=1)
    if step.periods is None:
        cautions.append("default_applied: 시점 = 최신")
    periods = resolve_periods(spec, index.periods, table.meta.period_type, lookup, cautions)
    names = [n for n in index.axes if n != ITEM_AXIS]
    by_key = _cells_by_key(table)
    combos = list(product(codes[ITEM_AXIS], *(codes[n] for n in names)))
    if spec.kind == "recent":
        monthly = any(len(p) == 6 for p in periods)
        usable = [p for p in periods if (len(p) == 6) == monthly and all(
            (c := by_key.get((combo[0], tuple(sorted(zip(names, combo[1:]))), p))) is not None and c.value is not None
            for combo in combos)]
        periods = usable[-spec.n:]
        if not periods:
            raise CalcRefused("missing_value", messages.refusal_message("missing_value", raw="-", meaning="숫자가 있는 시점이 없음"))
        if len(periods) < spec.n:
            cautions.append(f"recent: {spec.n}개 중 {len(periods)}개 시점만 숫자가 있음")
    refs: list[str] = []
    missing: list[str] = []
    for combo in combos:
        dims = tuple(sorted(zip(names, combo[1:])))
        for p in periods:
            cell = by_key.get((combo[0], dims, p))
            if cell is None:
                missing.append(f"{combo}·{p}")
            else:
                refs.append(cell.source_ref)
    if missing:
        raise CalcRefused("class_mismatch", messages.refusal_message("class_mismatch", what=", ".join(missing[:4])))
    return Resolved(refs, cautions)
