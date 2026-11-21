"""Run a v3 plan (schemas.PlanV3): each step is one block, and every number it makes is a Fact (T0-04).

Blocks: select reads cells (their facts, symbols included) · aggregate turns cells into one number (an average or a
sum as a derived fact; first / last / min / max point at a cell fact) · combine makes one derived fact from two
numbers · rank orders candidates and keeps the first ones · find points at the cells that meet a condition.
Every derived fact carries its derivation, so eval/grade.py recomputes it. Refusals are CalcRefused with the shared
reasons, as in engine.py, whose rounding rules this follows: a difference keeps the inputs' decimals, a rate, share,
multiple or level has 1 decimal, an average at least 1.
"""

from __future__ import annotations

import operator
import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from functools import partial

from kosis_chat.compute import scale, semantics
from kosis_chat.compute.resolve import EventLookup, resolve_select_cells
from kosis_chat.compute.engine import (
    PERCENT_DIGITS,
    CalcRefused,
    _cell_fact,
    _cells,
    _check_units,
    _derived,
    _diff_unit,
    _Ids,
    _numeric,
    _read,
    _Read,
    _share_parts,
)
from kosis_chat.data.schema import Cell, TableInput
from kosis_chat.judge import messages
from kosis_chat.prepare.store import ITEM_AXIS, CellKey, store_of
from kosis_chat.shared.patterns import FACT_ID
from kosis_chat.shared.periods import span_text  # T1-28 S2b: was answer.render (compute must not import the answer)

from kosis_ai.schemas import AggregateStep, CombineStep, Fact, FindStep, MapStep, PlanV3, RankStep, SelectStep  # isort: skip

# A whole fact id inside a derivation (renaming f3 must not touch f30): the shared FACT_ID between word boundaries.
_FACT_ID_WORD = re.compile(rf"\b{FACT_ID.pattern}\b")
_CMP = {">": operator.gt, ">=": operator.ge, "<": operator.lt, "<=": operator.le, "==": operator.eq}
AGG_NOTES = {"avg": "평균", "sum": "합계"}


class PlanUnsupported(Exception):
    """A block this version cannot run yet (a sum or an average over a map's values)."""


@dataclass
class Cells:
    ids: list[str]
    reads: list[_Read]


@dataclass
class Scalar:
    id: str
    value: float
    unit: str | None
    digits: int
    periods: tuple[str, str]
    series: list[_Read] = field(default_factory=list)  # first / last: the cells it was picked from
    exact: float | None = None  # the value before rounding; a ranking orders by it (rounded values can tie)

    @property
    def key(self) -> float:
        return self.value if self.exact is None else self.exact


@dataclass
class Ranking:
    ids: list[str]  # facts of the first candidate (the answer cites them)
    picked: dict[str, str]
    ranking: list[dict]


@dataclass
class Found:
    ids: list[str]


@dataclass
class Mapped:
    """One value per code of a map: codes, their labels and the scalar each became."""

    axis: str
    codes: list[str]
    labels: list[str]
    values: list[Scalar]

    @property
    def ids(self) -> list[str]:
        return [v.id for v in self.values]


@dataclass
class PlanRun:
    facts: list[Fact]
    results: dict[str, object]
    main: list[str]
    cautions: list[str] = field(default_factory=list)  # allowed numbers that need a word of care (semantics.py)

    def ids_of(self, step_id: str) -> list[str]:
        out = self.results.get(step_id)
        if out is None:  # an optional combine that was skipped
            return []
        if isinstance(out, Scalar):
            return [out.id]
        return list(out.ids)


class _MapCache:
    """The cells a map's bodies read, kept for the whole map, so that a map makes the same store requests for any
    number of codes. A key not seen yet is asked for under every code of the map at once (the key with the map's
    axis set to each code: what the same select asks for at the other codes); refs are kept as read (a body's refs
    do not change with the code). A key or ref the store did not have is remembered too. Only what the store
    returned is given back, so the bodies read the cells they read without the cache."""

    def __init__(self, axis: str, codes: list[str]) -> None:
        self.axis = axis
        self.codes = codes
        self.keys: dict[int, dict[CellKey, Cell | None]] = {}  # id(table) -> key -> its cell
        self.refs: dict[int, dict[str, Cell | None]] = {}  # id(table) -> ref -> its cell

    def _at_codes(self, key: CellKey, code: str) -> list[CellKey]:
        """key at every code of the map, when key is at `code` on the map's axis; else none."""
        item, dims, period = key
        if self.axis == ITEM_AXIS:
            return [(c, dims, period) for c in self.codes] if item == code else []
        if (self.axis, code) not in dims:
            return []
        return [(item, tuple((n, c if n == self.axis else v) for n, v in dims), period) for c in self.codes]

    def cells_at(self, table: TableInput, keys: Iterable[CellKey], code: str) -> dict[CellKey, Cell]:
        keys = list(keys)
        known = self.keys.setdefault(id(table), {})
        new = [k for k in keys if k not in known]
        if new:
            wanted = list(dict.fromkeys([*new, *(k2 for k in new for k2 in self._at_codes(k, code))]))
            found = store_of(table).cells_at(wanted)
            known.update((k, found.get(k)) for k in wanted)
        return {k: cell for k in keys if (cell := known[k]) is not None}

    def by_refs(self, table: TableInput, refs: list[str]) -> dict[str, Cell]:
        known = self.refs.setdefault(id(table), {})
        new = list(dict.fromkeys(r for r in refs if r not in known))
        if new:
            found = _cells(table, new)
            known.update((r, found.get(r)) for r in new)
        return {r: cell for r in refs if (cell := known[r]) is not None}


class _Run:
    def __init__(self, table: TableInput, lookup: EventLookup | None = None,
                 tables: dict[str, TableInput] | None = None) -> None:
        self.table = table  # the first table: a select without `table` reads it
        # T1-30: every table of the question by tbl_id; one table is the only entry (the behaviour before T1-30)
        self.tables = dict(tables) if tables else {table.meta.tbl_id: table}
        self.lookup = lookup
        self.star: dict[str, str] = {}
        self.cache: _MapCache | None = None  # inside a map: the cells its bodies read (one request for all codes)
        self.facts: list[Fact] = []
        self.ids = _Ids()
        self.results: dict[str, object] = {}
        self.cautions: list[str] = []
        self.raw: dict[str, float] = {}  # derived fact id -> its value before rounding
        self.origin: dict[str, frozenset[str]] = {}  # step id -> the tbl_ids its numbers come from (T1-30)

    def add(self, fact: Fact) -> Fact:
        self.facts.append(fact)
        return fact

    @property
    def multi(self) -> bool:
        return len(self.tables) > 1

    def _of(self, ref: str) -> TableInput:
        """The table a source_ref names (its first part is the tbl_id); the first table when no table has it."""
        return self.tables.get(ref.split("|", 1)[0], self.table)

    def _for(self, step: SelectStep) -> TableInput:
        """The table a select reads: its `table`, else the first table. A table the question was not given is left to
        resolve_select, which refuses it as before (out_of_scope)."""
        return self.tables.get(step.table, self.table) if step.table else self.table

    def origin_of(self, step) -> frozenset[str]:
        """The tbl_ids a step's numbers come from (T1-30): a select's table, the union of its inputs' otherwise."""
        if isinstance(step, SelectStep):
            if step.cells:
                return frozenset(ref.split("|", 1)[0] for ref in step.cells)
            return frozenset({step.table if step.table and step.table not in self.tables else self._for(step).meta.tbl_id})
        if isinstance(step, MapStep):
            for body in step.body:
                self.origin[body.id] = self.origin_of(body)
            out = frozenset().union(*(self.origin.pop(b.id) for b in step.body))
            return out
        return frozenset().union(*(self.origin.get(i, frozenset()) for i in step.inputs()))

    def _fetched(self, wanted: list[tuple[TableInput, str]]) -> list[dict[str, Cell]]:
        """For each (table, ref), the cells found in that table, for _read(found=): one store request per table (a
        question has one to three), however many refs."""
        tables: list[TableInput] = []
        refs: list[list[str]] = []
        at: list[int] = []
        for table, ref in wanted:
            n = next((n for n, t in enumerate(tables) if t is table), len(tables))
            if n == len(tables):
                tables.append(table)
                refs.append([])
            refs[n].append(ref)
            at.append(n)
        found = [self.cache.by_refs(t, r) if self.cache is not None else _cells(t, r) for t, r in zip(tables, refs)]
        return [found[n] for n in at]

    def _cross(self, *step_ids: str) -> bool:
        """The steps' numbers come from different tables (T1-30); never true for one table."""
        return self.multi and len({self.origin.get(i, frozenset()) for i in step_ids}) > 1

    # ------------------------------------------------------------ blocks

    def select(self, step: SelectStep) -> Cells:
        cells = step.cells
        if cells:
            wanted = [(self._of(ref), ref) for ref in cells]
            found = self._fetched(wanted)
        else:
            table = self._for(step)
            # in a map: the keys of this select at every code of the map are asked for at once
            cache = self.cache
            cells_at = None if cache is None else partial(cache.cells_at, table, code=self.star[cache.axis])
            resolved, known = resolve_select_cells(table, step, self.lookup, self.star, cells_at)
            cells = resolved.cells
            self.cautions += [c for c in resolved.cautions if c not in self.cautions]
            # the cells resolve found are read as they are; a ref whose tbl_id names another table is read there
            wanted = [(self._of(ref), ref) for ref in cells]
            other = iter(self._fetched([(t, ref) for t, ref in wanted if t is not table]))
            found = [known if t is table else next(other) for t, _ in wanted]
        reads = [_read(t, ref, f) for (t, ref), f in zip(wanted, found)]
        _check_units(reads)
        return Cells([self.add(_cell_fact(self.ids(), r)).id for r in reads], reads)

    def aggregate(self, step: AggregateStep) -> Scalar:
        src = self.results[step.of]
        if isinstance(src, Mapped):
            if step.fn not in ("min", "max"):
                raise PlanUnsupported(f"aggregate {step.id}: {step.fn} over a map")
            pick = (min if step.fn == "min" else max)(src.values, key=lambda v: v.key)
            return pick
        reads, ids = src.reads, src.ids
        periods = (reads[0].cell.period_raw, reads[-1].cell.period_raw)
        if step.fn in ("first", "last", "min", "max"):
            numeric = [(i, r) for i, r in zip(ids, reads) if r.value is not None]
            if not numeric:
                _numeric(reads[0])
            pick = {"first": lambda: numeric[0], "last": lambda: numeric[-1],
                    "min": lambda: min(numeric, key=lambda x: x[1].value),
                    "max": lambda: max(numeric, key=lambda x: x[1].value)}[step.fn]()
            r = pick[1]
            return Scalar(pick[0], r.value, r.unit, r.digits, (r.cell.period_raw,) * 2, series=reads)
        values = [_numeric(r) for r in reads]
        if len(values) < 2:
            raise CalcRefused("missing_value", messages.refusal_message("missing_value", raw="-", meaning="비교할 값이 부족함"))
        if step.fn == "sum":
            semantics.check_sum(reads)
        elif (caution := semantics.average_caution(reads)) and caution not in self.cautions:
            self.cautions.append(caution)
        digits = max(r.digits for r in reads)
        span = span_text(*periods)
        note = f"{span} {len(values)}개 시점 {AGG_NOTES[step.fn]}"
        if step.fn == "avg":
            digits = max(2, digits)  # loop 1: an average is read to two decimals (was 1)
            fact = _derived(self.ids(), f"({' + '.join(ids)}) / {len(ids)}", sum(values) / len(values), digits, reads[0].unit, note=note)
        else:
            fact = _derived(self.ids(), " + ".join(ids), sum(values), digits, reads[0].unit, note=note)
        self.add(fact)
        exact = sum(values) / len(values) if step.fn == "avg" else sum(values)
        return Scalar(fact.id, fact.value, fact.unit, digits, periods, exact=exact)

    def _scalar(self, step_id: str) -> Scalar:
        src = self.results[step_id]
        if isinstance(src, Scalar):
            return src
        if isinstance(src, Cells) and len(src.ids) == 1:
            r = src.reads[0]
            return Scalar(src.ids[0], _numeric(r), r.unit, r.digits, (r.cell.period_raw,) * 2)
        raise CalcRefused("class_mismatch", messages.refusal_message("class_mismatch", what=step_id))

    def combine(self, step: CombineStep) -> Scalar | None:
        if step.fn == "share":
            return self._share(step)
        a_src, b_src = self.results[step.a], self.results[step.b]
        cross = self._cross(step.a, step.b)  # T1-30: two tables; their units are aligned below, not compared here
        for src in (a_src, b_src):
            if isinstance(src, Cells):
                _check_units([*src.reads])
        reads = [s.reads[0] for s in (a_src, b_src) if isinstance(s, Cells)]
        if len(reads) == 2 and not cross:
            _check_units(reads)
        a, b = self._scalar(step.a), self._scalar(step.b)
        if a.id == b.id:  # first and last of a series with one number, or one cell against itself
            symbol = next((r for r in a.series if r.value is None), None)
            if symbol is not None:
                _numeric(symbol)
            raise CalcRefused("missing_value", messages.refusal_message("missing_value", raw="-", meaning="비교할 값이 부족함"))
        if cross:
            a, b = self._aligned(step.fn, a, b)
        if (step.orient == "larger_first" and a.value < b.value) or (step.orient == "smaller_first" and a.value > b.value):
            a, b = b, a
        if step.fn in ("diff", "pp_diff", "pct_change", "cagr", "sum"):
            semantics.check_same_unit(step.fn, a.unit, b.unit, f"{a.id}·{b.id}")
        unit = b.unit if b.unit is not None else a.unit
        if step.optional and self._meaningless(step.fn, a, b, unit):
            return None
        fn = step.fn
        if fn == "sum":
            fact = self._make(f"{a.id} + {b.id}", a.value + b.value, max(a.digits, b.digits), unit, step.note)
        elif fn == "diff":
            fact = self._make(f"{a.id} - {b.id}", a.value - b.value, max(a.digits, b.digits), _diff_unit(unit), step.note)
        elif fn == "pp_diff":
            fact = self._make(f"{a.id} - {b.id}", a.value - b.value, PERCENT_DIGITS, "%p", step.note)
        else:
            if not b.value:
                raise CalcRefused("missing_value", messages.refusal_message("missing_value", raw="0", meaning="기준값이 0"))
            if fn == "ratio":
                fact = self._make(f"{a.id} / {b.id}", a.value / b.value, PERCENT_DIGITS, "배", step.note)
            elif fn == "pct_change":
                fact = self._make(f"({a.id} / {b.id} - 1) * 100", (a.value / b.value - 1) * 100, PERCENT_DIGITS, "%", step.note)
            elif fn == "index":
                fact = self._make(f"{a.id} / {b.id} * 100", a.value / b.value * 100, PERCENT_DIGITS, "%", step.note)
            elif fn == "per":
                scale = int(step.scale) if float(step.scale).is_integer() else step.scale
                fact = self._make(f"{a.id} / {b.id} * {scale}", a.value / b.value * step.scale, PERCENT_DIGITS, None, step.note)
            else:  # cagr
                if a.value <= 0 or b.value <= 0:
                    raise CalcRefused("missing_value", messages.refusal_message("missing_value", raw="0", meaning="연평균 증가율은 양수끼리만"))
                value = ((a.value / b.value) ** (1 / step.years) - 1) * 100
                fact = self._make(f"(({a.id} / {b.id}) ** (1 / {step.years}) - 1) * 100", value, PERCENT_DIGITS, "%", step.note)
        self.add(fact)
        return Scalar(fact.id, fact.value, fact.unit, fact.round, (b.periods[0], a.periods[1]), exact=self.raw[fact.id])

    def _make(self, expr: str, value: float, digits: int, unit: str | None, note: str | None) -> Fact:
        fact = _derived(self.ids(), expr, value, digits, unit, note)
        self.raw[fact.id] = value
        return fact

    def _converted(self, x: Scalar, unit: str) -> Scalar:
        """x in another unit of the same base (T1-30, '천명' -> '명'): a derived fact whose derivation names the factor."""
        op, factor = scale.conversion(x.unit, unit)
        zeros = len(str(factor)) - 1
        if op == "*":
            value, digits = x.value * factor, max(0, x.digits - zeros)
        else:
            value, digits = x.value / factor, x.digits + zeros
        fact = self._make(f"{x.id} {op} {factor}", value, digits, unit, f"{x.unit}→{unit} 환산")
        self.add(fact)
        return Scalar(fact.id, fact.value, unit, digits, x.periods, exact=value)

    def _aligned(self, fn: str, a: Scalar, b: Scalar) -> tuple[Scalar, Scalar]:
        """Two numbers from different tables in one unit (T1-30, D5): a unit that differs only by a prefix is
        converted to the finer one; other different units are refused, except for per, ratio and index, whose
        result does not keep the inputs' unit (births per 1,000 people)."""
        if not a.unit or not b.unit or a.unit == b.unit:
            return a, b
        target = scale.common_unit(a.unit, b.unit)
        if target is None:
            if fn in ("per", "ratio", "index"):
                return a, b
            raise CalcRefused("unit_mismatch", messages.refusal_message("unit_mismatch", what=f"{a.id}·{b.id}",
                                                                        units=f"{a.unit} / {b.unit}"))
        return (a if a.unit == target else self._converted(a, target)), (b if b.unit == target else self._converted(b, target))

    @staticmethod
    def _meaningless(fn: str, a: Scalar, b: Scalar, unit: str | None) -> bool:
        if fn == "ratio":
            return not b.value or a.value / b.value <= 0
        if fn in ("pct_change", "index"):
            return unit == "%" or not b.value
        if fn == "cagr":
            return a.value <= 0 or b.value <= 0
        return fn == "per" and not b.value

    def _share(self, step: CombineStep) -> Scalar:
        """a: the part cells (summed), b: the total cell; the facts were added by their select steps."""
        parts, total = self.results[step.a], self.results[step.b]
        if not (isinstance(parts, Cells) and isinstance(total, Cells) and len(total.ids) == 1):
            raise CalcRefused("class_mismatch", messages.refusal_message("class_mismatch", what=step.id))
        total_id = total.ids[0]
        if self._cross(step.a, step.b):  # T1-30: the part and the whole are in two tables
            _check_units(parts.reads)
            semantics.check_share_parts(parts.reads)
            semantics.check_share_periods(parts.reads, total.reads[0])
            values = [_numeric(r) for r in parts.reads]
            whole = _numeric(total.reads[0])
            unit, total_unit = parts.reads[0].unit, total.reads[0].unit
            if unit and total_unit and unit != total_unit:
                if scale.conversion(total_unit, unit) is None:
                    raise CalcRefused("unit_mismatch", messages.refusal_message(
                        "unit_mismatch", what=f"{step.a}·{step.b}", units=f"{unit} / {total_unit}"))
                r = total.reads[0]
                converted = self._converted(Scalar(total_id, whole, total_unit, r.digits, (r.cell.period_raw,) * 2), unit)
                total_id, whole = converted.id, converted.value
        else:
            _check_units([*parts.reads, *total.reads])
            semantics.check_share_parts(parts.reads)
            semantics.check_share_scope(self.table, parts.reads, total.reads[0])
            values = [_numeric(r) for r in parts.reads]  # the parts first, then the whole (the order before T1-30)
            whole = _numeric(total.reads[0])
        if not whole:
            raise CalcRefused("missing_value", messages.refusal_message("missing_value", raw="0", meaning="합계가 0"))
        top = " + ".join(parts.ids)
        expr = f"({top}) / {total_id} * 100" if len(parts.ids) > 1 else f"{top} / {total_id} * 100"
        semantics.check_share_bound(sum(values) / whole * 100, "·".join(r.cell.item for r in parts.reads))
        fact = self.add(_derived(self.ids(), expr, sum(values) / whole * 100, PERCENT_DIGITS, "%", step.note))
        period = total.reads[0].cell.period_raw
        return Scalar(fact.id, fact.value, "%", PERCENT_DIGITS, (period, period))

    def rank(self, step: RankStep) -> Ranking:
        if step.of:
            return self._rank_mapped(step)
        word = "가장 높음" if step.pick == "max" else "가장 낮음"
        ranked: list[tuple[float, list[Fact], str]] = []
        # every candidate and its total (read in the candidate's table) in one request per table, not one per cell
        fetched = self._fetched([(self._of(ref), ref) for ref in step.candidates]
                                + [(self._of(ref), total) for ref, total in zip(step.candidates, step.totals)])
        for i, ref in enumerate(step.candidates):
            ids = _Ids()
            found = fetched[i]
            try:
                if step.totals:
                    table = self._of(ref)
                    part, whole = _read(table, ref, found), _read(table, step.totals[i], found)
                    facts, share, score = _share_parts(table, [ref], step.totals[i], ids, found)
                    semantics.check_share_scope(table, [part], whole)
                    semantics.check_share_bound(score, part.cell.item)
                    ranked.append((score, [*facts, share], share.id))
                else:
                    read = _read(self._of(ref), ref, found)
                    ranked.append((_numeric(read), [_cell_fact(ids(), read)], "f1"))
            except CalcRefused as exc:
                if exc.reason != "missing_value":
                    raise
        if len(ranked) < 2:
            raise CalcRefused("missing_value", messages.refusal_message("missing_value", raw="-", meaning="비교할 값이 부족함"))
        semantics.check_rank_units([next(f.unit for f in group if f.id == scored) for _, group, scored in ranked])
        ranked.sort(key=lambda x: x[0], reverse=step.pick == "max")
        picked: dict[str, str] = {}
        first: list[str] = []
        places = ("winner", "runner_up") + tuple(f"place_{n}" for n in range(3, step.keep + 1))
        for place, (_, group, scored) in zip(places, ranked[: step.keep]):
            offset = len(self.facts)
            renamed = {f.id: f"f{offset + n}" for n, f in enumerate(group, start=1)}
            for f in group:
                note = (f"{len(ranked)}개 중 {word}" if place == "winner" else "두 번째" if place == "runner_up"
                        else f"{place[6:]}번째") if f.id == scored else None
                expr = _FACT_ID_WORD.sub(lambda m: renamed[m.group(0)], f.derivation) if f.derivation else None
                self.add(f.model_copy(update={"id": renamed[f.id], "derivation": expr, "note": note or f.note}))
            self.ids.n = len(self.facts)
            picked[place] = renamed[scored]
            if place == "winner":
                first = list(renamed.values())
        ranking = []
        for score, group, scored in ranked:
            cell = next(f for f in group if f.source_ref)
            unit = next(f.unit for f in group if f.id == scored)
            ranking.append({"dims": dict(cell.dims or {}), "period": cell.period, "value": score, "unit": unit})
        return Ranking(first, picked, ranking)

    def _rank_mapped(self, step: RankStep) -> Ranking:
        """Rank a map's values; their facts are already made, the first keep get their place as a note."""
        src: Mapped = self.results[step.of]
        semantics.check_rank_units([v.unit for v in src.values])
        order = sorted(range(len(src.values)), key=lambda n: src.values[n].key, reverse=step.pick == "max")
        word = "가장 높음" if step.pick == "max" else "가장 낮음"
        places = ("winner", "runner_up") + tuple(f"place_{n}" for n in range(3, step.keep + 1))
        picked: dict[str, str] = {}
        for place, n in zip(places, order[: step.keep]):
            fid = src.values[n].id
            note = f"{len(order)}개 중 {word}" if place == "winner" else "두 번째" if place == "runner_up" else f"{place[6:]}번째"
            at = next(i for i, f in enumerate(self.facts) if f.id == fid)
            old = self.facts[at].note
            self.facts[at] = self.facts[at].model_copy(update={"note": f"{src.labels[n]}: {note}" + (f" ({old})" if old else "")})
            picked[place] = fid
        ranking = [{"dims": {src.axis: src.labels[n]}, "period": None, "value": src.values[n].value, "unit": src.values[n].unit}
                   for n in order]
        return Ranking([picked[p] for p in places[: min(step.keep, len(order))]], picked, ranking)

    def map(self, step: MapStep) -> Mapped:
        """Run the body once per code, with the mapped axis's '*' set to that code; keep result per code."""
        starred = next(b for b in step.body if isinstance(b, SelectStep) and step.axis in b.dims)
        axis = semantics.index_of(self._for(starred)).axes.get(step.axis)
        if axis is None:
            raise CalcRefused("class_mismatch", messages.refusal_message("class_mismatch", what=f"분류 {step.axis}"))
        values: list[Scalar] = []
        self.cache = _MapCache(step.axis, list(step.codes))
        try:  # the cache and '*' are the map's only: a select after the map (or after a refused map) reads the store
            for code in step.codes:
                if code not in axis.labels:
                    raise CalcRefused("class_mismatch",
                                      messages.refusal_message("class_mismatch", what=f"{step.axis} {code}"))
                self.star = {step.axis: code}
                for body in step.body:
                    self.origin[body.id] = self.origin_of(body)
                    self.results[body.id] = getattr(self, body.op)(body)
                values.append(self._scalar(step.result))
                for body in step.body:
                    del self.results[body.id]
                    self.origin.pop(body.id, None)
        finally:
            self.cache = None
            self.star = {}
        return Mapped(step.axis, list(step.codes), [axis.labels[c] for c in step.codes], values)

    def find(self, step: FindStep) -> Found:
        src = self.results[step.of]
        if step.ref is not None:
            ref = self._scalar(step.ref)
            semantics.check_same_unit("find", src.reads[0].unit if src.reads else None, ref.unit, f"{step.of}·{step.ref}")
        threshold = step.value if step.value is not None else ref.value
        hits = [i for i, r in zip(src.ids, src.reads) if r.value is not None and _CMP[step.cmp](r.value, threshold)]
        if not hits:
            raise CalcRefused("missing_value", messages.refusal_message("missing_value", raw="-", meaning="조건에 맞는 값이 없음"))
        return Found({"first": hits[:1], "last": hits[-1:], "all": hits}[step.which])


def run_plan(table: TableInput, plan: PlanV3, lookup: EventLookup | None = None,
             tables: dict[str, TableInput] | None = None) -> PlanRun:
    """Facts of plan over table; lookup answers events missing from the calendar (resolve.py).

    tables (T1-30): every table of a question by tbl_id, table among them (the one a select without `table` reads)."""
    run = _Run(table, lookup, tables)
    for step in plan.steps:
        run.origin[step.id] = run.origin_of(step)
        run.results[step.id] = getattr(run, step.op)(step)
    out = PlanRun(run.facts, run.results, [], run.cautions)
    out.main = list(dict.fromkeys(i for s in plan.show for i in out.ids_of(s)))
    return out
