"""Numbers by code (T1-07): read cells and turn them into facts the answer may cite.

compute(table, built) runs the plan the rules built (build.py) on the v3 block engine (plan_engine.py) and returns
cell facts with their source_ref and derived facts whose derivation names other fact ids (recomputed by
eval/grade.py). Nothing is guessed: a cell that is missing or holds a symbol, units that differ, or rates that would
be summed raise CalcRefused with a RefusalReason. This module keeps the cell reading and fact making the blocks use.
Rounding follows the golden set: a difference keeps the inputs' decimals, a rate, share or multiple has 1 decimal,
an average has at least 1, a sum keeps the inputs' decimals.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, replace

from kosis_chat.compute.build import Built, Pattern
from kosis_chat.data.preprocessing import clean_cell
from kosis_chat.data.schema import Cell, TableInput
from kosis_chat.data.serializer import display_value
from kosis_chat.judge import messages
from kosis_chat.judge.question import COUNT_KINDS, cell_measure

from kosis_ai.schemas import Fact, RefusalReason, StatResult  # isort: skip  (question.py puts src/ on the path)

PERCENT_DIGITS = 1  # growth rate, share and multiple
_DECIMALS = re.compile(r"\.(\d+)")
_INDEX_CACHE: dict[int, tuple[TableInput, dict[str, Cell], bool]] = {}


class CalcRefused(Exception):
    """The request cannot be computed from the table; reason is a shared RefusalReason."""

    def __init__(self, reason: RefusalReason, message: str) -> None:
        super().__init__(message)
        self.reason = reason
        self.message = message


@dataclass(frozen=True)
class TermResult:
    """What one number of a period pattern became: its cell fact ids and the id of the number it stands for."""

    cell_ids: list[str]
    value_id: str  # the aggregate / share fact, or the single cell
    periods: tuple[str, str]  # first and last period of the term's cells
    digits: int  # decimals the number keeps
    agg: str | None = None  # "avg" / "sum" / None (a cell or a share)


@dataclass(frozen=True)
class Computed:
    result: StatResult
    main: list[str]  # fact ids the answer should cite, in order (the rest are context)
    note: str | None = None
    picked: dict[str, str] = field(default_factory=dict)  # extreme: {"winner": fact id, "runner_up": fact id}
    # extreme: every ranked candidate for the chart, best first: {"dims", "period", "value", "unit"} (facts keep two)
    ranking: list[dict] = field(default_factory=list)
    # value: the same cell in every period, for the chart: {"period", "value"} in period order (facts keep one)
    series: list[dict] = field(default_factory=list)
    pattern: Pattern | None = None  # the shape the rules built (build.py): templates and charts choose by it
    # period patterns: what each compared number became, earlier first (templates and charts read them)
    terms: list[TermResult] = field(default_factory=list)
    plan: object | None = None  # the PlanV3 that made these facts
    cautions: list[str] = field(default_factory=list)  # allowed numbers that need a word of care (S6 shows them)


def _indexed(table: TableInput) -> tuple[TableInput, dict[str, Cell], bool]:
    """(table, cells by source_ref, whether facts name the item)."""
    cached = _INDEX_CACHE.get(id(table))
    if cached is None or cached[0] is not table:
        if len(_INDEX_CACHE) > 8:
            _INDEX_CACHE.clear()
        cached = (table, {c.source_ref: c for c in table.cells}, _item_named(table))
        _INDEX_CACHE[id(table)] = cached
    return cached


def _item_named(table: TableInput) -> bool:
    """False when one item only wraps a '기본항목별' category that holds the real items (DT_1B8000F)."""
    if len({c.item_code for c in table.cells}) > 1 or not table.cells:
        return True
    return not any("항목" in name for name in table.cells[0].dims)


def _by_ref(table: TableInput) -> dict[str, Cell]:
    return _indexed(table)[1]


@dataclass(frozen=True)
class _Read:
    cell: Cell
    value: float | None
    unit: str | None
    kind: str | None
    digits: int
    show_item: bool  # False when the item label says nothing (see _item_named)


def _read(table: TableInput, ref: str) -> _Read:
    cell = _by_ref(table).get(ref)
    if cell is None:
        raise CalcRefused("class_mismatch", messages.refusal_message("class_mismatch", what=ref))
    if cell.value_flag is None:
        cell = clean_cell(cell)
    unit, kind = cell_measure(table, cell)
    unit = unit or (cell.unit if cell.unit_source != "unknown" else None) or None
    found = _DECIMALS.search(display_value(cell.value_raw))
    return _Read(cell, cell.value, unit, kind, len(found.group(1)) if found else 0, _indexed(table)[2])


def _numeric(read: _Read) -> float:
    if read.value is None:
        raw = read.cell.value_raw.strip()
        meaning = {"missing": "자료 없음", "suppressed": "비공개"}.get(read.cell.value_flag or "", "수치 아님")
        raise CalcRefused("missing_value", messages.refusal_message("missing_value", raw=raw or "빈칸", meaning=meaning))
    return read.value


def _cell_fact(fact_id: str, read: _Read, note: str | None = None) -> Fact:
    c = read.cell
    return Fact(
        id=fact_id,
        value=read.value,
        unit=read.unit,
        period=c.period_raw,
        dims={**c.dims, "항목": c.item} if read.show_item or not c.dims else dict(c.dims),
        source_ref=c.source_ref,
        raw=None if read.value is not None else c.value_raw.strip(),
        measure_kind=read.kind if read.kind in ("count_stock", "count_flow", "ratio", "average", "index") else None,
        note=note,
    )


def _rounded(value: float, digits: int) -> int | float:
    out = round(value, digits)
    return int(out) if digits == 0 else out


def _derived(fact_id: str, expr: str, value: float, digits: int, unit: str | None, note: str | None = None) -> Fact:
    return Fact(id=fact_id, value=_rounded(value, digits), unit=unit, derivation=expr, round=digits, note=note)


def _diff_unit(unit: str | None) -> str | None:
    return "%p" if unit == "%" else unit


def _check_units(reads: list[_Read]) -> None:
    units = {r.unit for r in reads if r.unit}
    if len(units) > 1:
        what = "와(과) ".join(dict.fromkeys(r.cell.item for r in reads))
        raise CalcRefused("unit_mismatch", messages.refusal_message("unit_mismatch", what=what, units=" / ".join(sorted(units))))


class _Ids:
    def __init__(self) -> None:
        self.n = 0

    def __call__(self) -> str:
        self.n += 1
        return f"f{self.n}"


_SERIES_CACHE: dict[int, tuple[TableInput, dict[tuple, list[Cell]]]] = {}


def _series(table: TableInput, cell: Cell) -> list[dict]:
    cached = _SERIES_CACHE.get(id(table))
    if cached is None or cached[0] is not table:
        if len(_SERIES_CACHE) > 8:
            _SERIES_CACHE.clear()
        groups: dict[tuple, list[Cell]] = {}
        for c in table.cells:
            groups.setdefault((c.item_code, tuple(sorted(c.dim_codes.items()))), []).append(c)
        cached = (table, groups)
        _SERIES_CACHE[id(table)] = cached
    same = cached[1].get((cell.item_code, tuple(sorted(cell.dim_codes.items()))), [])
    reads = sorted((_read(table, c.source_ref) for c in same), key=lambda r: r.cell.period_raw)
    return [{"period": r.cell.period_raw, "value": r.value} for r in reads if r.value is not None]


def _share_parts(table: TableInput, parts: list[str], total: str, ids: _Ids) -> tuple[list[Fact], Fact, float]:
    reads = [_read(table, ref) for ref in parts]
    total_read = _read(table, total)
    _check_units([*reads, total_read])
    if len(reads) > 1 and any(r.kind not in COUNT_KINDS for r in reads if r.kind):
        what = "·".join(r.cell.item for r in reads)
        raise CalcRefused("not_aggregatable", messages.refusal_message("not_aggregatable", what=what))
    values = [_numeric(r) for r in reads]
    whole = _numeric(total_read)
    if not whole:
        raise CalcRefused("missing_value", messages.refusal_message("missing_value", raw="0", meaning="합계가 0"))
    facts = [_cell_fact(ids(), r) for r in reads]
    total_fact = _cell_fact(ids(), total_read)
    top = " + ".join(f.id for f in facts)
    expr = f"({top}) / {total_fact.id} * 100" if len(facts) > 1 else f"{top} / {total_fact.id} * 100"
    share = sum(values) / whole * 100
    return [*facts, total_fact], _derived(ids(), expr, share, PERCENT_DIGITS, "%"), share


def _with_context(table: TableInput, computed: Computed, refs: list[str]) -> Computed:
    """Context cells (a 전국 value, a newer period shown as '-') join the facts, symbols included; never cited first."""
    facts = list(computed.result.facts)
    for ref in refs:
        if ref in _by_ref(table) and all(f.source_ref != ref for f in facts):
            facts.append(_cell_fact(f"f{len(facts) + 1}", _read(table, ref)))
    return replace(computed, result=StatResult(facts=facts))


def compute(table: TableInput, built: Built) -> Computed:
    """Facts of the built plan; Computed also keeps what templates and charts read beside the facts."""
    from kosis_chat.compute.plan_engine import run_plan

    run = run_plan(table, built.plan)
    by_id = {f.id: f for f in run.facts}
    extra: dict = {}
    if built.pattern == "value":
        read = _read(table, built.plan.steps[0].cells[0])
        _numeric(read)
        extra["series"] = _series(table, read.cell)
    if built.pattern == "extreme":
        ranked = run.results["k"]
        extra.update(picked=ranked.picked, ranking=ranked.ranking)
    if built.terms:
        terms = []
        for t in built.terms:
            value = run.ids_of(t.value)[0]
            cell_ids = run.ids_of(t.cells)
            digits = by_id[value].round if by_id[value].derivation else run.results[t.cells].reads[0].digits
            terms.append(TermResult(cell_ids, value, t.periods, digits, t.agg))
        extra["terms"] = terms
    computed = Computed(StatResult(facts=run.facts), run.main, built.plan.note, pattern=built.pattern,
                        cautions=run.cautions, **extra)
    return replace(_with_context(table, computed, built.context), plan=built.plan)
