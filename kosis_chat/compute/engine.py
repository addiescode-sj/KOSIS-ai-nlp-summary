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
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

from kosis_chat.compute.build import Built, Pattern
from kosis_chat.data.preprocessing import clean_cell
from kosis_chat.data.schema import Cell, TableInput
from kosis_chat.prepare.index import index_of
from kosis_chat.prepare.store import MemoryStore, dim_key, store_of
from kosis_chat.shared.numbers import display_value  # T1-28 S2b: was data.serializer (B0); re-exported there
from kosis_chat.judge import messages
from kosis_chat.judge.question import COUNT_KINDS, cell_measure

from kosis_ai.schemas import Fact, RefusalReason, StatResult  # isort: skip  (question.py puts src/ on the path)
from kosis_chat.compute.schemas import Computed, TermResult  # isort: skip  # re-export: moved in T1-28 S1b

if TYPE_CHECKING:  # resolve.py imports this module
    from kosis_chat.compute.resolve import EventLookup

PERCENT_DIGITS = 2  # growth rate, share and multiple (loop 1: was 1; a computed value is read to two decimals)
_DECIMALS = re.compile(r"\.(\d+)")


class CalcRefused(Exception):
    """The request cannot be computed from the table; reason is a shared RefusalReason."""

    def __init__(self, reason: RefusalReason, message: str) -> None:
        super().__init__(message)
        self.reason = reason
        self.message = message


def _indexed(table: TableInput) -> tuple[TableInput, dict[str, Cell], bool]:
    """(table, cells by source_ref, whether facts name the item)."""
    # For in-memory tables only (tests): the whole ref dict of the memory store, one per TableInput.
    return table, MemoryStore.of(table).ref_map, _item_named(table)


def _item_named(table: TableInput) -> bool:
    """False when one item only wraps a '기본항목별' category that holds the real items (DT_1B8000F)."""
    return index_of(table).item_named  # a fact of the whole table: the index has it (T1-31 S2)


@dataclass(frozen=True)
class _Read:
    cell: Cell
    value: float | None
    unit: str | None
    kind: str | None
    digits: int
    show_item: bool  # False when the item label says nothing (see _item_named)


def _read(table: TableInput, ref: str, found: Mapping[str, Cell] | None = None) -> _Read:
    """The cell of ref as a fact reads it. found: cells by ref the caller already asked the store for, so a step that
    reads many cells makes one request (_cells); without it, this asks the store for the one cell."""
    cell = found.get(ref) if found is not None else store_of(table).by_ref(ref)
    if cell is None:
        raise CalcRefused("class_mismatch", messages.refusal_message("class_mismatch", what=ref))
    if cell.value_flag is None:
        cell = clean_cell(cell)
    unit, kind = cell_measure(table, cell)
    unit = unit or (cell.unit if cell.unit_source != "unknown" else None) or None
    decimals = _DECIMALS.search(display_value(cell.value_raw))
    return _Read(cell, cell.value, unit, kind, len(decimals.group(1)) if decimals else 0, _item_named(table))


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


def _cells(table: TableInput, refs: list[str]) -> dict[str, Cell]:
    """The cells of refs in one store request, for _read(found=)."""
    return store_of(table).by_refs(refs)


def _series(table: TableInput, cell: Cell) -> list[dict]:
    same = store_of(table).series(cell.item_code, dim_key(cell))
    found = {c.source_ref: c for c in same}  # the series' own cells: no second request (refs are unique)
    reads = sorted((_read(table, c.source_ref, found) for c in same), key=lambda r: r.cell.period_raw)
    return [{"period": r.cell.period_raw, "value": r.value} for r in reads if r.value is not None]


def _share_parts(table: TableInput, parts: list[str], total: str, ids: _Ids,
                 found: Mapping[str, Cell]) -> tuple[list[Fact], Fact, float]:
    """found: the cells of parts and total, asked for by the caller (rank asks for every candidate at once)."""
    reads = [_read(table, ref, found) for ref in parts]
    total_read = _read(table, total, found)
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
    found = _cells(table, refs) if refs else {}  # most questions have no context cells: no request
    for ref in refs:
        if ref in found and all(f.source_ref != ref for f in facts):
            facts.append(_cell_fact(f"f{len(facts) + 1}", _read(table, ref, found)))
    return computed.model_copy(update={"result": StatResult(facts=facts)})


def compute(table: TableInput, built: Built, lookup: EventLookup | None = None,
            tables: dict[str, TableInput] | None = None) -> Computed:
    """Facts of the built plan; Computed also keeps what templates and charts read beside the facts.

    lookup answers an event the calendar does not have (a plan the model wrote may name one). tables (T1-30): every
    table of the question by tbl_id, table among them (a select without `table` reads table)."""
    from kosis_chat.compute.plan_engine import run_plan

    run = run_plan(table, built.plan, lookup, tables)
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
    return _with_context(table, computed, built.context).model_copy(update={"plan": built.plan})
