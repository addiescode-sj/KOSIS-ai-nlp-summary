"""A Vega-Lite chart for each answer, built by code from the computed facts (the model never touches it).

value: one bar · trend: a line over every period · compare: two bars · share: a pie of the parts and the rest ·
extreme: every candidate as a sorted bar · avg / sum: the periods as bars with the aggregate in the title ·
pavg / psum / vsa / shr: the two numbers compared as bars · gap: the gap at each period as bars · grw: each target's
change as bars · plan (written by the model): the cells it shows, as bars. (S6) Whenever the main cells form a grid of
two or more targets by two or more periods (answer/table.py), the chart is that grid: bars of the targets side by side
at each period (grouped_bar, up to 3 periods) or one line per target (multi_line). The asked value is highlighted; numbers are the facts' own values.
T1-28 S3a: moved here from answer/chart.py (a re-export shim now); the Vega-Lite pieces are chart/vega.py and the grid
chart/grid.py. T1-28 S5: chart_spec returns a ChartSpec (kind, title, unit, the drawn points with the fact each one is,
the facts drawn, bar layout); chart/vega.py to_vega(spec) draws it as the same Vega-Lite spec the builders made before.
BUILDERS (old name _CHARTS) is the pattern -> builder table (E6), register_builder adds one.
Render with any Vega-Lite v5 viewer (vega-embed, Streamlit st.vega_lite_chart, Altair); eval/view_run.py shows them.
"""

from __future__ import annotations

from collections.abc import Callable

from kosis_chat.chart.grid import grid
from kosis_chat.chart.vega import HIGHLIGHT, OTHER, SCHEMA, WIDTH, _bars, _base, _color, _row, _timeline, _tooltip, _unit_title  # noqa: F401
from kosis_chat.compute import Computed
from kosis_chat.shared.labels import fact_label, item_label
from kosis_chat.shared.numbers import format_fact, format_number  # noqa: F401
from kosis_chat.shared.periods import period_text, span_text

from kosis_ai.schemas import ChartPoint, ChartSeries, ChartSpec, Fact, QuestionType  # isort: skip


def _point(name: str, fact: Fact, highlight: bool, *, link: bool = True, display: str | None = None,
           period: str | None = None, share_pct: float | None = None) -> ChartPoint:
    """One drawn value from a fact (its value and how it is written). link: the point is that fact (False for a row
    made from a fact with another value: another period of the series, the rest of a pie)."""
    return ChartPoint(label=name, value=fact.value, display=format_fact(fact) if display is None else display,
                      fact_id=fact.id if link else None, period=period, highlight=highlight, share_pct=share_pct)


def _spec(kind: str, title: str, unit: str | None, points: list[ChartPoint], fact_ids: list[str], computed: Computed,
          **layout) -> ChartSpec:
    """A one-series chart (every kind but the grid ones)."""
    return ChartSpec(kind=kind, title=title, unit=unit, series=[ChartSeries(name=title, points=points)], fact_ids=fact_ids,
                     pattern=computed.pattern, **layout)


def _differing(rows: list[tuple[dict, str | None]]) -> list[str]:
    """A short label per row: the categories that differ between rows, else the period."""
    keys = [k for k in dict.fromkeys(k for dims, _ in rows for k in dims) if len({dims.get(k) for dims, _ in rows}) > 1]
    if keys:
        return [" ".join(item_label(dims.get(k, "")) for k in keys).strip() for dims, _ in rows]
    return [f"{period}" for _, period in rows]


def _common(rows: list[tuple[dict, str | None]]) -> str:
    """What every row shares ('2023년 1인'), for the title."""
    dims = rows[0][0]
    shared = {k: v for k, v in dims.items() if all(d.get(k) == v for d, _ in rows)}
    periods = {p for _, p in rows}
    label = fact_label(Fact(id="x", dims=shared, note="title"))
    year = f"{rows[0][1]}년 " if len(periods) == 1 and rows[0][1] else ""
    return f"{year}{label}".strip()


def _value(computed: Computed, by_id: dict[str, Fact]) -> ChartSpec | None:
    f = by_id[computed.main[0]]
    if len(computed.series) < 2:
        return _spec("bar", _common([(f.dims or {}, f.period)]) or "값", f.unit, [_point(fact_label(f) or f"{f.period}", f, True)],
                     [f.id], computed, orientation="horizontal", sort="-x")
    points = [_point(p["period"], f.model_copy(update={"value": p["value"]}), p["period"] == f.period,
                     link=p["period"] == f.period and p["value"] == f.value, period=p["period"]) for p in computed.series]
    return _spec("timeline", f"{fact_label(f)} ({f.period}년 강조)".strip(), f.unit, points, [f.id], computed)


def _trend(computed: Computed, by_id: dict[str, Fact]) -> ChartSpec | None:
    points = sorted((f for f in computed.result.facts if f.source_ref and f.value is not None), key=lambda f: f.period or "")
    series = [f for f in points if f.dims == by_id[computed.main[0]].dims]
    ends = {computed.main[0], computed.main[1]}
    drawn = [_point(f.period or "", f, f.id in ends, period=f.period) for f in series]
    return _spec("line", f"{fact_label(series[0])} 추이", series[0].unit, drawn, [f.id for f in series], computed)


def _compare(computed: Computed, by_id: dict[str, Fact]) -> ChartSpec | None:
    a, b, diff = (by_id[i] for i in computed.main[:3])
    subject = diff.derivation.split("-")[0].strip()
    names = _differing([(a.dims or {}, a.period), (b.dims or {}, b.period)])
    points = [_point(names[0], a, a.id == subject), _point(names[1], b, b.id == subject)]
    return _spec("bar", f"{_common([(a.dims or {}, a.period), (b.dims or {}, b.period)])} 비교".strip(), a.unit, points,
                 [a.id, b.id], computed, orientation="vertical", sort=names)


def _share(computed: Computed, by_id: dict[str, Fact]) -> ChartSpec | None:
    *parts, total, share = (by_id[i] for i in computed.main)
    if any(f.value is None for f in [*parts, total]):
        return None
    names = _differing([(f.dims or {}, f.period) for f in [*parts, total]])[:-1]
    points = [_point(n, p, True, share_pct=round(p.value / total.value * 100, 1)) for n, p in zip(names, parts)]
    rest = total.value - sum(p.value for p in parts)
    if rest > 0:
        rest_fact = total.model_copy(update={"value": rest})
        points.append(_point("그 외", rest_fact, False, link=False, share_pct=round(rest / total.value * 100, 1)))
    title = f"{_common([(total.dims or {}, total.period)])} 가운데 {'·'.join(names)} 비중 {format_fact(share)}"
    return _spec("pie", title, total.unit, points, [f.id for f in [*parts, total, share]], computed)


def _extreme(computed: Computed, by_id: dict[str, Fact]) -> ChartSpec | None:
    ranking = computed.ranking
    if not ranking:
        return None
    names = _differing([(r["dims"], r["period"]) for r in ranking])
    digits = by_id[computed.picked["winner"]].round  # a share ranking shows 1 decimal, like its facts
    fact_ids = list(computed.picked.values())  # the winner and the runner-up are facts; the other candidates are rows only
    picked = [by_id[i] for i in fact_ids]
    points = []
    for n, (name, r) in enumerate(zip(names, ranking)):
        value = r["value"] if digits is None else round(r["value"], digits)
        # The ranking is in rank order and picked in place order (winner, runner_up, place_3, ...): the n-th point is
        # the n-th picked fact when it shows that fact's value (a share ranking's fact is the derived share); else a
        # fact of the same cell and value, else none (a candidate shown for context).
        placed = picked[n] if n < len(picked) and picked[n].value == value else None
        same = [f.id for f in picked if f.dims == r["dims"] and f.period == r["period"] and f.value == value]
        fact_id = placed.id if placed is not None else same[0] if len(same) == 1 else None
        points.append(ChartPoint(label=name, value=value, display=format_number(value, r["unit"], digits),
                                 fact_id=fact_id, period=r["period"], highlight=n == 0))
    if all(r["dims"] == ranking[0]["dims"] for r in ranking):  # ranked over periods: keep time order
        order = sorted(range(len(points)), key=lambda n: ranking[n]["period"] or "")
        title = f"{fact_label(Fact(id='x', dims=ranking[0]['dims'], note='title'))} 시점별"
        return _spec("timeline", title, ranking[0]["unit"], [points[n] for n in order], fact_ids, computed)
    title = _common([(r["dims"], r["period"]) for r in ranking])
    word = "비중" if by_id[computed.picked["winner"]].derivation else ""
    return _spec("bar", f"{title} {word} 순위".replace("  ", " ").strip(), ranking[0]["unit"], points, fact_ids, computed,
                 orientation="horizontal", sort=[p.label for p in points])


# ---------------------------------------------------------------- patterns over periods


def _span_name(term) -> str:
    return span_text(*term.periods)


def _one_term(computed: Computed, by_id: dict[str, Fact]) -> ChartSpec | None:
    """avg / sum: the periods as bars, the aggregate in the title (every bar is a fact of the answer)."""
    term = computed.terms[0]
    cells = [by_id[i] for i in term.cell_ids]
    value = by_id[term.value_id]
    word = {"avg": "평균", "sum": "합계"}.get(term.agg or "", "")
    points = [_point(f.period or "", f, True, period=f.period) for f in cells]
    title = f"{fact_label(cells[0])} {_span_name(term)} {word} {format_fact(value)}".replace("  ", " ").strip()
    return _spec("timeline", title, cells[0].unit, points, [f.id for f in cells] + [value.id], computed)


def _two_terms(computed: Computed, by_id: dict[str, Fact]) -> ChartSpec | None:
    """pavg / psum / vsa / shr: the two numbers compared as bars, the later one highlighted."""
    earlier, later = computed.terms
    a, b = by_id[earlier.value_id], by_id[later.value_id]
    word = {"avg": " 평균", "sum": " 합계"}
    names = [f"{_span_name(earlier)}{word.get(earlier.agg or '', '')}", f"{_span_name(later)}{word.get(later.agg or '', '')}"]
    if computed.pattern == "shr":
        names = [_span_name(earlier), _span_name(later)]
    label = fact_label(by_id[earlier.cell_ids[0]])
    what = "비중" if computed.pattern == "shr" else ""
    title = f"{label} {what} 비교".replace("  ", " ").strip()
    return _spec("bar", title, a.unit, [_point(names[0], a, False), _point(names[1], b, True)], [a.id, b.id], computed,
                 orientation="vertical", sort=names)


def _gap(computed: Computed, by_id: dict[str, Fact]) -> ChartSpec | None:
    """gap: the gap at the earlier and the later period as bars, the later one highlighted."""
    x1, y1, x2, _, g1, g2, _ = (by_id[i] for i in computed.main)
    names = [period_text(x1.period or ""), period_text(x2.period or "")]
    names_pair = _differing([(x1.dims or {}, x1.period), (y1.dims or {}, y1.period)])
    title = f"{names_pair[0]}·{names_pair[1]} {_common([(x1.dims or {}, None), (y1.dims or {}, None)])} 격차"
    return _spec("bar", title.replace("  ", " ").strip(), g1.unit, [_point(names[0], g1, False), _point(names[1], g2, True)],
                 [g1.id, g2.id], computed, orientation="vertical", sort=names)


def _grw(computed: Computed, by_id: dict[str, Fact]) -> ChartSpec | None:
    """grw: each target's change (growth rate or amount) as bars, the one the answer names first highlighted."""
    a1, _, a2, _, v1, v2, d = (by_id[i] for i in computed.main)
    names = _differing([(a1.dims or {}, a1.period), (a2.dims or {}, a2.period)])
    subject = d.derivation.split("-")[0].strip()
    word = "증감률" if v1.unit == "%" else "증감"
    points = [_point(names[0], v1, v1.id == subject, display=format_number(v1.value, v1.unit, v1.round)),  # keep the sign
              _point(names[1], v2, v2.id == subject, display=format_number(v2.value, v2.unit, v2.round))]
    return _spec("bar", f"{_common([(a1.dims or {}, None), (a2.dims or {}, None)])} {word} 비교".replace("  ", " ").strip(),
                 v1.unit, points, [v1.id, v2.id], computed, orientation="vertical", sort=names)


def _listed(computed: Computed, by_id: dict[str, Fact]) -> ChartSpec | None:
    """plan (a plan the model wrote): the cells it shows as bars when there are two or more in one unit; a chart of
    mixed numbers (values and their difference) would mislead, so the computed ones are left out."""
    cells = [by_id[i] for i in computed.main if not by_id[i].derivation and by_id[i].value is not None]
    if len(cells) < 2 or len({f.unit for f in cells}) != 1:
        return None
    names = _differing([(f.dims or {}, f.period) for f in cells])
    if len(set(names)) != len(names):
        names = [f"{period_text(f.period or '')} {name}".strip() for f, name in zip(cells, names)]
    title = _common([(f.dims or {}, f.period) for f in cells]) or "값"
    return _spec("bar", title, cells[0].unit, [_point(name, f, False) for name, f in zip(names, cells)],
                 [f.id for f in cells], computed, orientation="vertical", sort=names)


# Patterns whose answer compares targets over periods: drawn as a grid when the cells form one.
GRID_PATTERNS = ("gap", "grw", "plan", "trend", "compare")
BUILDERS: dict[str, Callable[[Computed, dict[str, Fact]], ChartSpec | None]] = {"value": _value, "trend": _trend, "compare": _compare, "share": _share, "extreme": _extreme,
           "avg": _one_term, "sum": _one_term, "pavg": _two_terms, "psum": _two_terms, "vsa": _two_terms, "shr": _two_terms,
           "gap": _gap, "grw": _grw, "plan": _listed}
_CHARTS = BUILDERS  # the old name (T1-28 S3a), the same dict


def register_builder(pattern: str, builder: Callable[[Computed, dict[str, Fact]], ChartSpec | None]) -> None:
    """Add or replace the builder of a pattern (E6): (computed, facts by id) -> ChartSpec or None."""
    BUILDERS[pattern] = builder


GROUPED_MAX_PERIODS = 3


def _grid(computed: Computed) -> ChartSpec | None:
    g = grid(computed)
    if g is None:
        return None
    series = [ChartSeries(name=name, points=points) for name, facts in g.rows
              if (points := [_point(period_text(p), f, False, period=p) for p, f in zip(g.periods, facts)
                             if f is not None and f.value is not None])]
    if not series:
        return None
    first = next(f for _, facts in g.rows for f in facts if f is not None)
    title = _common([(f.dims or {}, None) for _, facts in g.rows for f in facts if f is not None]) or fact_label(first)
    kind = "grouped_bar" if len(g.periods) <= GROUPED_MAX_PERIODS else "multi_line"
    ids = [f.id for _, facts in g.rows for f in facts if f is not None]
    return ChartSpec(kind=kind, title=title, unit=g.unit, series=series, fact_ids=ids, pattern=computed.pattern)


def chart_spec(kind: QuestionType, computed: Computed) -> ChartSpec | None:
    """The chart for the answer (kind, the drawn points, the facts drawn; no Vega-Lite: chart/vega.py to_vega), or None
    when there is nothing to draw."""
    by_id = {f.id: f for f in computed.result.facts}
    if computed.pattern in GRID_PATTERNS and (found := _grid(computed)) is not None:
        return found
    builder = BUILDERS[computed.pattern or kind]
    try:
        return builder(computed, by_id)
    except (KeyError, IndexError, StopIteration):  # an unexpected fact layout: answer without a chart
        return None
