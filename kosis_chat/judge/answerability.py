"""Answer, refuse or ask back: rules that look at the question and the whole table before the LLM is called (T1-08).

judge(question, table) runs the rules in order and the first one that fires decides. A final verdict (refused or
clarify) is used as is and the LLM is not called; an answered verdict may carry defaults the answer must state.
The reasons are the shared RefusalReason values (src/kosis_ai/schemas.py) that eval/grade.py checks.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from kosis_chat.data.schema import Cell, TableInput
from kosis_chat.judge import messages
from kosis_chat.judge.question import (
    ARITH_WORDS,
    AVG_WORDS,
    CAUSAL_WORDS,
    CHANGE_WORDS,
    COUNT_KINDS,
    ITEM_AXIS,
    OUT_OF_DOMAIN,
    SCENARIO_AXIS,
    SUM_WORDS,
    TOTAL_WORDS,
    TREND_WORDS,
    VAGUE_WORDS,
    VALUE_ASKS,
    Parsed,
    TableIndex,
    build_index,
    in_range,
    is_total,
    normalize,
    parse,
    squash,
)

from kosis_ai.schemas import Outcome, Refusal  # isort: skip  (question.py puts src/ on the path)

_UNIT_SUFFIX = re.compile(r"\s*\([^()]*\)\s*$")
RECENT_POINTS = 5  # "최근 추이" with no number of years, as in scope.py
# "가장 많았던 해는?": the period is the answer, not something to assume.
_ASKS_PERIOD = re.compile(r"(?:해|연도|년도|시점)(?:는|가|은)?\s*(?:언제|어디|\?|$)|언제")
_DEFAULT_SCENARIO = "중위"
_COUNTRY_WORD = "전국"


@dataclass(frozen=True)
class Verdict:
    outcome: Outcome
    rule: str | None = None  # the rule that decided; None when no rule fired
    refusal: Refusal | None = None
    clarify_question: str = ""
    clarify_options: tuple[str, ...] = ()
    default_applied: dict[str, str] = field(default_factory=dict)
    final: bool = False  # True: use this outcome and skip the LLM

    @classmethod
    def refuse(cls, rule: str, reason: str, available: str | None = None, **fields: str) -> Verdict:
        message = messages.refusal_message(reason, **fields)
        return cls("refused", rule, Refusal(reason=reason, message=message, available=available), final=True)


# ---------------------------------------------------------------- resolving labels to codes


@dataclass
class Resolved:
    codes: dict[str, set[str]]  # axis -> codes the question names
    ambiguous: list[tuple[str, list[str]]]  # (label, options) for a label several parents share


def resolve(index: TableIndex, parsed: Parsed) -> Resolved:
    """Codes per axis. A label several places share ('중구') takes the parent the question names, else the
    shallowest one ('동부' under 전국), else it is ambiguous. A named parent is dropped when its child is kept."""
    codes: dict[str, set[str]] = {}
    ambiguous: list[tuple[str, list[str]]] = []
    for name, matches in parsed.by_axis().items():
        axis = index.axes[name]
        named = {m.code for m in matches}
        by_label: dict[str, list[str]] = {}
        for m in matches:
            by_label.setdefault(squash(m.label), []).append(m.code)
        kept: set[str] = set()
        for group in by_label.values():
            if len(group) == 1:
                kept |= set(group)
                continue
            under_named = [c for c in group if named & set(axis.ancestors(c))]
            depths = {c: axis.depth(c) for c in group}
            top = [c for c in group if depths[c] == min(depths.values())]
            if under_named:
                kept |= set(under_named)
            elif len(top) == 1:
                kept |= set(top)
            else:
                label = axis.labels[group[0]]
                options = [f"{axis.labels.get(axis.parents[c] or '', '')} {label}".strip() for c in group]
                ambiguous.append((label, options))
                kept |= set(group)
        kept -= {a for c in kept for a in axis.ancestors(c)}
        codes[name] = kept
    return Resolved(codes=codes, ambiguous=ambiguous)


def targets(index: TableIndex, parsed: Parsed, resolved: Resolved) -> list[Cell]:
    """Cells the question points at: named codes on every axis it names, inside the periods it names."""
    return [
        c
        for c in index.table.cells
        if parsed.periods.contains(c.period_raw)
        and all(index.axis_of(c, name) in codes for name, codes in resolved.codes.items())
    ]


# ---------------------------------------------------------------- measures


def _kind_class(kind: str | None) -> str | None:
    return None if kind is None else ("count" if kind in COUNT_KINDS else "rate")


def _incompatible(a: tuple[str | None, str | None], b: tuple[str | None, str | None]) -> bool:
    ca, cb = _kind_class(a[1]), _kind_class(b[1])
    if ca and cb and ca != cb:
        return True
    return bool(a[0] and b[0] and a[0] != b[0])


def named_measures(index: TableIndex, resolved: Resolved) -> dict[str, tuple[str | None, str | None]]:
    """Label -> (unit, measure_kind) for named labels that hold exactly one measure (items like '조출생률')."""
    out = {}
    for name, codes in resolved.codes.items():
        for code in codes:
            measure = index.measure_of(name, code)
            if measure is not None and len(_distinct_measures(index, name)) > 1:
                out[index.axes[name].labels[code]] = measure
    return out


def _distinct_measures(index: TableIndex, axis: str) -> set:
    return {m for code in index.axes[axis].labels if (m := index.measure_of(axis, code)) is not None}


def _show(label: str) -> str:
    """'조이혼율(천명당)' -> "'조이혼율'": the label without its unit suffix, quoted."""
    return f"'{_UNIT_SUFFIX.sub('', label.strip().lstrip('-').strip())}'"


def _measure_text(measure: tuple[str | None, str | None]) -> str:
    unit, kind = measure
    return unit if _kind_class(kind) != "rate" else f"{unit or '단위 미상'}, 비율·평균"


def _named_text(index: TableIndex, resolved: Resolved, periods: tuple[str, ...]) -> str:
    labels = [_show(index.axes[n].labels[c]) for n, codes in resolved.codes.items() for c in sorted(codes)]
    return " ".join([*(_period_text(s) for s in periods), ", ".join(labels)]).strip() or "질문하신 조건"


# ---------------------------------------------------------------- rules


def _causal(p: Parsed, **_) -> Verdict | None:
    if CAUSAL_WORDS.search(p.question) and not VALUE_ASKS.search(p.question):
        return Verdict.refuse("causal", "causal")
    return None


def _out_of_scope(p: Parsed, **_) -> Verdict | None:
    if not p.matches and OUT_OF_DOMAIN.search(p.question):
        return Verdict.refuse("out_of_scope", "out_of_scope")
    return None


def _available(index: TableIndex) -> str:
    years = sorted({y[:4] for y in index.periods})
    return f"이 표의 시점: {years[0]}~{years[-1]}년"


def _no_period(p: Parsed, index: TableIndex, **_) -> Verdict | None:
    missing = [_period_text(s) for s in p.periods.singles if not index.period_ok(s)]
    missing += [_range_text(a, b) for a, b in p.periods.ranges if not any(in_range(y, a, b) for y in index.periods)]
    if not missing:
        return None
    return Verdict.refuse("no_period", "no_period", available=_available(index), asked=", ".join(missing))


def _sums_or_averages(p: Parsed) -> bool:
    """The question aggregates several periods into one number (v2 ops avg / sum / pavg / psum / vsa)."""
    return bool(AVG_WORDS.search(p.rest) or SUM_WORDS.search(p.rest) or TOTAL_WORDS.search(p.rest))


def _partial_range(p: Parsed, index: TableIndex, **_) -> Verdict | None:
    """An average or sum over "2010년부터 2024년까지" when the table starts in 2015: the number would be wrong, not
    just incomplete, so it is refused (a plain trend over the same range is answered with a note)."""
    if not _sums_or_averages(p):
        return None
    years = {y[:4] for y in index.periods}
    missing = [y for a, b in p.periods.ranges if b != "9999" and len(a) == 4 for y in (a, b) if y not in years]
    if not missing:
        return None
    asked = ", ".join(_period_text(y) for y in dict.fromkeys(missing))
    return Verdict.refuse("partial_range", "no_period", available=_available(index), asked=asked)


def _period_text(period: str) -> str:
    return f"{period[:4]}년 {int(period[4:])}월" if len(period) == 6 and period.isdigit() else f"{period}년"


def _range_text(start: str, end: str) -> str:
    return f"{_period_text(start)}~{_period_text(end)}" if end != "9999" else f"{_period_text(start)} 이후"


def _same_name(p: Parsed, resolved: Resolved, **_) -> Verdict | None:
    if not resolved.ambiguous:
        return None
    label, options = resolved.ambiguous[0]
    question = messages.CLARIFY_LABEL.format(label=label)
    return Verdict("clarify", "same_name", clarify_question=question, clarify_options=tuple(options), final=True)


def _vague_item(p: Parsed, index: TableIndex, resolved: Resolved, **_) -> Verdict | None:
    """'부산 출생 통계 알려줘': a vague ask whose word is in several measures with different units."""
    if not VAGUE_WORDS.search(p.question) or named_measures(index, resolved):
        return None
    words = {w for w in p.words if not VAGUE_WORDS.search(w)}
    for name, axis in index.axes.items():
        if len(_distinct_measures(index, name)) < 2:
            continue
        hits = {
            code: label
            for code, label in axis.labels.items()
            if index.measure_of(name, code) and any(w in label for w in words)
        }
        if len(hits) >= 2 and len({index.measure_of(name, code) for code in hits}) >= 2:
            options = tuple(label.strip() for label in hits.values())
            return Verdict("clarify", "vague_item", clarify_question=messages.CLARIFY_ITEM, clarify_options=options,
                           final=True)
    return None


def _unit_mismatch(p: Parsed, index: TableIndex, resolved: Resolved, **_) -> Verdict | None:
    if not ARITH_WORDS.search(p.rest):
        return None
    named = named_measures(index, resolved)
    if len(named) == 1:
        named.update(_total_twin(p, index, resolved))
    labels = list(named)
    for i, a in enumerate(labels):
        for b in labels[i + 1 :]:
            if _incompatible(named[a], named[b]):
                units = " / ".join(f"{_show(x)} {_measure_text(named[x])}" for x in (a, b))
                return Verdict.refuse("unit_mismatch", "unit_mismatch", what=f"{_show(a)}와(과) {_show(b)}", units=units)
    return None


def _total_twin(p: Parsed, index: TableIndex, resolved: Resolved) -> dict:
    """'외국인주민 수와 외국인주민 세대수를 합하면': the label's head word asked twice, once for the named label
    ('외국인주민 세대수') and once for the category total ('합계', 명). Returns {total label: measure} or {}."""
    for name, codes in resolved.codes.items():
        axis = index.axes[name]
        for code in codes:
            head = axis.labels[code].split()[0]
            total = next((c for c, label in axis.labels.items() if is_total(label)), None)
            if total and total != code and len(head) >= 2 and p.question.count(head) >= 2:
                measure = index.measure_of(name, total)
                return {axis.labels[total]: measure} if measure else {}
    return {}


def _not_aggregatable(p: Parsed, index: TableIndex, resolved: Resolved, **_) -> Verdict | None:
    if not (SUM_WORDS.search(p.rest) or (TOTAL_WORDS.search(p.rest) and p.periods.ranges)):
        return None
    for name, codes in resolved.codes.items():
        axis = index.axes[name]
        scenario = SCENARIO_AXIS.search(name) or any(SCENARIO_AXIS.search(axis.labels[c]) for c in codes)
        if len(codes) >= 2 and not axis.has_total and scenario:
            what = "·".join(_show(axis.labels[c].split("(")[0]) for c in sorted(codes))
            return Verdict.refuse("not_aggregatable", "not_aggregatable", what=what)
    named = named_measures(index, resolved)
    kinds = {kind for _, kind in named.values()}
    if not named:  # the question names no measure: the table's only measure, if it has one
        kinds = {kind for found in index.measures.values() for _, kind in found}
    if kinds and all(_kind_class(k) == "rate" for k in kinds):
        what = "·".join(_show(label) for label in named) or "이 표의 값"
        return Verdict.refuse("not_aggregatable", "not_aggregatable", what=what)
    # A stock (가구 수, 인구) summed over several periods counts the same households again (v2, CX-REF-03).
    several = bool(p.periods.ranges) or len(p.periods.singles) >= 2
    if several and kinds and all(k == "count_stock" for k in kinds):
        what = "·".join(_show(label) for label in named) or "이 표의 값"
        return Verdict.refuse("stock_over_periods", "not_aggregatable", what=what)
    return None


def _no_cell(p: Parsed, index: TableIndex, resolved: Resolved, cells: list[Cell], **_) -> Verdict | None:
    """Labels and periods each exist but not together (세종 before 2012), or '전국' in a table without it."""
    if _COUNTRY_WORD in p.question and not any(m.form == _COUNTRY_WORD for m in p.matches):
        if any(axis.regional for axis in index.axes.values()):
            return Verdict.refuse("no_cell", "class_mismatch", what="전국")
    if p.periods.explicit and not cells:
        return Verdict.refuse("no_cell", "class_mismatch", what=_named_text(index, resolved, p.periods.singles))
    return None


def _symbol(p: Parsed, index: TableIndex, resolved: Resolved, cells: list[Cell], **_) -> Verdict | None:
    if not p.periods.explicit or not cells or any(c.value_kind == "ok" for c in cells):
        return None
    kinds = {c.value_kind for c in cells}
    if kinds == {"blank"}:  # the cell exists but nothing was collected for that period
        return Verdict.refuse("symbol", "class_mismatch", what=_named_text(index, resolved, p.periods.singles))
    cell = next(c for c in cells if c.value_kind != "blank")
    meaning = normalize.VALUE_LABELS.get(cell.value_kind) or "수치 아님"
    return Verdict.refuse("symbol", "missing_value", raw=cell.value_raw.strip(), meaning=meaning)


# Order matters: the first rule that fires decides (a period that is not there beats a label that is shared).
RULES = (
    _causal, _out_of_scope, _no_period, _partial_range, _same_name, _vague_item, _unit_mismatch, _not_aggregatable,
    _no_cell, _symbol,
)  # fmt: skip


# ---------------------------------------------------------------- defaults


def defaults(p: Parsed, index: TableIndex, resolved: Resolved, cells: list[Cell]) -> dict[str, str]:
    """What the answer assumes when the question leaves it open: the period, and the scenario of a projection."""
    out: dict[str, str] = {}
    for name, axis in index.axes.items():
        if name == ITEM_AXIS or name in resolved.codes or axis.has_total or not SCENARIO_AXIS.search(name):
            continue
        base = next((label for label in axis.labels.values() if label.startswith(_DEFAULT_SCENARIO)), None)
        if base:
            out[name] = base.split("(")[0].strip()
    if p.periods.explicit or p.periods.recent_n or _ASKS_PERIOD.search(p.question):
        return out
    numeric = sorted({c.period_raw for c in cells if c.value_kind == "ok"})
    if not numeric:
        return out
    if p.periods.previous:  # "작년 대비": the latest period is this year, the one before it is last year
        latest = numeric[-1]
        before = [x for x in numeric if x < latest and len(x) == len(latest)]
        if before:
            out["period"] = f"latest ({latest}) vs previous ({before[-1]})"
            return out
    if (p.periods.recent and TREND_WORDS.search(p.question)) or CHANGE_WORDS.search(p.question):
        points = numeric[-RECENT_POINTS:]
        out["period"] = f"recent {len(points)} points ({points[0]}~{points[-1]})"
    else:
        out["period"] = f"latest ({numeric[-1]})"
    return out


# ---------------------------------------------------------------- entry point


def judge(question: str, table: TableInput, index: TableIndex | None = None) -> Verdict:
    """Run the rules on the whole table (before any scope cut). Pass an index to reuse it across questions."""
    index = index or build_index(table)
    parsed = parse(index, question)
    resolved = resolve(index, parsed)
    cells = targets(index, parsed, resolved)
    for rule in RULES:
        verdict = rule(parsed, index=index, resolved=resolved, cells=cells)
        if verdict is not None:
            return verdict
    applied = defaults(parsed, index, resolved, cells)
    return Verdict("answered", "default" if applied else None, default_applied=applied)
