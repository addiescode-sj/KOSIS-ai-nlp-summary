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
from dataclasses import dataclass, field

from kosis_chat.answer.render import span_text
from kosis_chat.compute import semantics
from kosis_chat.compute.resolve import EventLookup, resolve_select
from kosis_chat.compute.engine import (
    PERCENT_DIGITS,
    CalcRefused,
    _cell_fact,
    _check_units,
    _derived,
    _diff_unit,
    _Ids,
    _numeric,
    _read,
    _Read,
    _share_parts,
)
from kosis_chat.data.schema import TableInput
from kosis_chat.judge import messages

from kosis_ai.schemas import AggregateStep, CombineStep, Fact, FindStep, MapStep, PlanV3, RankStep, SelectStep  # isort: skip

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


class _Run:
    def __init__(self, table: TableInput, lookup: EventLookup | None = None) -> None:
        self.table = table
        self.lookup = lookup
        self.star: dict[str, str] = {}
        self.facts: list[Fact] = []
        self.ids = _Ids()
        self.results: dict[str, object] = {}
        self.cautions: list[str] = []
        self.raw: dict[str, float] = {}  # derived fact id -> its value before rounding

    def add(self, fact: Fact) -> Fact:
        self.facts.append(fact)
        return fact

    # ------------------------------------------------------------ blocks

    def select(self, step: SelectStep) -> Cells:
        cells = step.cells
        if not cells:
            resolved = resolve_select(self.table, step, self.lookup, self.star)
            cells = resolved.cells
            self.cautions += [c for c in resolved.cautions if c not in self.cautions]
        reads = [_read(self.table, ref) for ref in cells]
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
            digits = max(1, digits)
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
        for src in (a_src, b_src):
            if isinstance(src, Cells):
                _check_units([*src.reads])
        reads = [s.reads[0] for s in (a_src, b_src) if isinstance(s, Cells)]
        if len(reads) == 2:
            _check_units(reads)
        a, b = self._scalar(step.a), self._scalar(step.b)
        if a.id == b.id:  # first and last of a series with one number, or one cell against itself
            symbol = next((r for r in a.series if r.value is None), None)
            if symbol is not None:
                _numeric(symbol)
            raise CalcRefused("missing_value", messages.refusal_message("missing_value", raw="-", meaning="비교할 값이 부족함"))
        if (step.orient == "larger_first" and a.value < b.value) or (step.orient == "smaller_first" and a.value > b.value):
            a, b = b, a
        if step.fn in ("diff", "pp_diff", "pct_change", "cagr"):
            semantics.check_same_unit(step.fn, a.unit, b.unit, f"{a.id}·{b.id}")
        unit = b.unit if b.unit is not None else a.unit
        if step.optional and self._meaningless(step.fn, a, b, unit):
            return None
        fn = step.fn
        if fn == "diff":
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
        _check_units([*parts.reads, *total.reads])
        semantics.check_share_parts(parts.reads)
        semantics.check_share_scope(self.table, parts.reads, total.reads[0])
        values = [_numeric(r) for r in parts.reads]
        whole = _numeric(total.reads[0])
        if not whole:
            raise CalcRefused("missing_value", messages.refusal_message("missing_value", raw="0", meaning="합계가 0"))
        top = " + ".join(parts.ids)
        expr = f"({top}) / {total.ids[0]} * 100" if len(parts.ids) > 1 else f"{top} / {total.ids[0]} * 100"
        semantics.check_share_bound(sum(values) / whole * 100, "·".join(r.cell.item for r in parts.reads))
        fact = self.add(_derived(self.ids(), expr, sum(values) / whole * 100, PERCENT_DIGITS, "%", step.note))
        period = total.reads[0].cell.period_raw
        return Scalar(fact.id, fact.value, "%", PERCENT_DIGITS, (period, period))

    def rank(self, step: RankStep) -> Ranking:
        if step.of:
            return self._rank_mapped(step)
        word = "가장 높음" if step.pick == "max" else "가장 낮음"
        ranked: list[tuple[float, list[Fact], str]] = []
        for i, ref in enumerate(step.candidates):
            ids = _Ids()
            try:
                if step.totals:
                    part, whole = _read(self.table, ref), _read(self.table, step.totals[i])
                    facts, share, score = _share_parts(self.table, [ref], step.totals[i], ids)
                    semantics.check_share_scope(self.table, [part], whole)
                    semantics.check_share_bound(score, part.cell.item)
                    ranked.append((score, [*facts, share], share.id))
                else:
                    read = _read(self.table, ref)
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
                expr = re.sub(r"\bf\d+\b", lambda m: renamed[m.group(0)], f.derivation) if f.derivation else None
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
        axis = semantics.index_of(self.table).axes.get(step.axis)
        if axis is None:
            raise CalcRefused("class_mismatch", messages.refusal_message("class_mismatch", what=f"분류 {step.axis}"))
        values: list[Scalar] = []
        for code in step.codes:
            if code not in axis.labels:
                raise CalcRefused("class_mismatch", messages.refusal_message("class_mismatch", what=f"{step.axis} {code}"))
            self.star = {step.axis: code}
            for body in step.body:
                self.results[body.id] = getattr(self, body.op)(body)
            values.append(self._scalar(step.result))
            for body in step.body:
                del self.results[body.id]
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


def run_plan(table: TableInput, plan: PlanV3, lookup: EventLookup | None = None) -> PlanRun:
    """Facts of plan over table; lookup answers events missing from the calendar (resolve.py)."""
    run = _Run(table, lookup)
    for step in plan.steps:
        run.results[step.id] = getattr(run, step.op)(step)
    out = PlanRun(run.facts, run.results, [], run.cautions)
    out.main = list(dict.fromkeys(i for s in plan.show for i in out.ids_of(s)))
    return out
