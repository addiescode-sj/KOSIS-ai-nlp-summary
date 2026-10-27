"""A Vega-Lite chart for each answer, built by code from the computed facts (the model never touches it).

value: one bar · trend: a line over every period · compare: two bars · share: a pie of the parts and the rest ·
extreme: every candidate as a sorted bar · avg / sum: the periods as bars with the aggregate in the title ·
pavg / psum / vsa / shr: the two numbers compared as bars · gap: the gap at each period as bars · grw: each target's
change as bars. The asked value is highlighted; numbers are the facts' own values. chart_spec returns the shared Chart (kind, Vega-Lite spec, the facts it draws) that StatAnswer carries.
Render with any Vega-Lite v5 viewer (vega-embed, Streamlit st.vega_lite_chart, Altair); eval/view_run.py shows them.
"""

from __future__ import annotations

from kosis_chat.answer.render import fact_label, format_fact, format_number, item_label, period_text, span_text
from kosis_chat.compute import Computed

from kosis_ai.schemas import Chart, Fact, QuestionType  # isort: skip

SCHEMA = "https://vega.github.io/schema/vega-lite/v5.json"
HIGHLIGHT, OTHER = "#2f6db5", "#b8c4d6"
WIDTH = 360


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


def _unit_title(unit: str | None) -> str:
    return f"값 ({unit})" if unit else "값"


def _base(title: str, values: list[dict]) -> dict:
    return {"$schema": SCHEMA, "title": title, "width": WIDTH, "data": {"values": values},
            "config": {"font": "sans-serif", "axis": {"labelFontSize": 11, "titleFontSize": 11}}}


def _color(field: str = "강조") -> dict:
    return {"field": field, "type": "nominal", "legend": None,
            "scale": {"domain": ["예", "아니오"], "range": [HIGHLIGHT, OTHER]}}


def _tooltip() -> list[dict]:
    return [{"field": "이름", "type": "nominal"}, {"field": "표시", "type": "nominal", "title": "값"}]


def _bars(title: str, rows: list[dict], unit: str | None, sort: str | list = "-x", horizontal: bool = True) -> dict:
    value_axis = {"field": "값", "type": "quantitative", "title": _unit_title(unit)}
    name_axis = {"field": "이름", "type": "nominal", "title": None, "sort": sort, "axis": {"labelAngle": 0, "labelLimit": 160}}
    spec = _base(title, rows)
    spec["layer"] = [
        {"mark": {"type": "bar"}, "encoding": {"y" if horizontal else "x": name_axis, "x" if horizontal else "y": value_axis,
                                                "color": _color(), "tooltip": _tooltip()}},
        {"mark": {"type": "text", "align": "left", "dx": 3} if horizontal else {"type": "text", "dy": -6},
         "encoding": {"y" if horizontal else "x": name_axis, "x" if horizontal else "y": value_axis, "text": {"field": "표시"}}},
    ]
    if horizontal:
        spec["height"] = {"step": 22}
    return spec


def _row(name: str, fact: Fact, highlight: bool) -> dict:
    return {"이름": name, "값": fact.value, "표시": format_fact(fact), "강조": "예" if highlight else "아니오"}


def _timeline(title: str, rows: list[dict], unit: str | None) -> dict:
    """Bars in period order, the asked periods highlighted (a value in its series, a lowest year)."""
    spec = _base(title, rows)
    x = {"field": "이름", "type": "ordinal", "title": None, "sort": None, "axis": {"labelAngle": -45 if len(rows) > 12 else 0}}
    y = {"field": "값", "type": "quantitative", "title": _unit_title(unit)}
    spec["layer"] = [
        {"mark": {"type": "bar"}, "encoding": {"x": x, "y": y, "color": _color(), "tooltip": _tooltip()}},
        {"transform": [{"filter": "datum['강조'] == '예'"}], "mark": {"type": "text", "dy": -6},
         "encoding": {"x": x, "y": y, "text": {"field": "표시"}}},
    ]
    spec["width"] = max(WIDTH, 14 * len(rows))
    return spec


def _value(computed: Computed, by_id: dict[str, Fact]) -> Chart | None:
    f = by_id[computed.main[0]]
    if len(computed.series) < 2:
        spec = _bars(_common([(f.dims or {}, f.period)]) or "값", [_row(fact_label(f) or f"{f.period}", f, True)], f.unit)
        return Chart(kind="bar", spec=spec, fact_ids=[f.id])
    rows = [_row(p["period"], f.model_copy(update={"value": p["value"]}), p["period"] == f.period) for p in computed.series]
    return Chart(kind="timeline", spec=_timeline(f"{fact_label(f)} ({f.period}년 강조)".strip(), rows, f.unit), fact_ids=[f.id])


def _trend(computed: Computed, by_id: dict[str, Fact]) -> Chart | None:
    points = sorted((f for f in computed.result.facts if f.source_ref and f.value is not None), key=lambda f: f.period or "")
    series = [f for f in points if f.dims == by_id[computed.main[0]].dims]
    ends = {computed.main[0], computed.main[1]}
    rows = [{"시점": f.period, **_row(f.period or "", f, f.id in ends)} for f in series]
    unit = series[0].unit
    spec = _base(f"{fact_label(series[0])} 추이", rows)
    x = {"field": "시점", "type": "ordinal", "title": None, "axis": {"labelAngle": 0}}
    y = {"field": "값", "type": "quantitative", "title": _unit_title(unit), "scale": {"zero": False}}
    spec["layer"] = [
        {"mark": {"type": "line", "color": OTHER}, "encoding": {"x": x, "y": y}},
        {"mark": {"type": "point", "filled": True, "size": 60}, "encoding": {"x": x, "y": y, "color": _color(), "tooltip": _tooltip()}},
        {"transform": [{"filter": "datum['강조'] == '예'"}], "mark": {"type": "text", "dy": -10},
         "encoding": {"x": x, "y": y, "text": {"field": "표시"}}},
    ]
    return Chart(kind="line", spec=spec, fact_ids=[f.id for f in series])


def _compare(computed: Computed, by_id: dict[str, Fact]) -> Chart | None:
    a, b, diff = (by_id[i] for i in computed.main[:3])
    subject = diff.derivation.split("-")[0].strip()
    names = _differing([(a.dims or {}, a.period), (b.dims or {}, b.period)])
    rows = [_row(names[0], a, a.id == subject), _row(names[1], b, b.id == subject)]
    spec = _bars(f"{_common([(a.dims or {}, a.period), (b.dims or {}, b.period)])} 비교".strip(), rows, a.unit,
                 sort=names, horizontal=False)
    return Chart(kind="bar", spec=spec, fact_ids=[a.id, b.id])


def _share(computed: Computed, by_id: dict[str, Fact]) -> Chart | None:
    *parts, total, share = (by_id[i] for i in computed.main)
    if any(f.value is None for f in [*parts, total]):
        return None
    names = _differing([(f.dims or {}, f.period) for f in [*parts, total]])[:-1]
    rows = [{**_row(n, p, True), "비중": round(p.value / total.value * 100, 1)} for n, p in zip(names, parts)]
    rest = total.value - sum(p.value for p in parts)
    if rest > 0:
        rest_fact = total.model_copy(update={"value": rest})
        rows.append({**_row("그 외", rest_fact, False), "비중": round(rest / total.value * 100, 1)})
    spec = _base(f"{_common([(total.dims or {}, total.period)])} 가운데 {'·'.join(names)} 비중 {format_fact(share)}", rows)
    spec["layer"] = [
        {"mark": {"type": "arc", "innerRadius": 50, "stroke": "white"},
         "encoding": {"theta": {"field": "값", "type": "quantitative", "stack": True},
                      "color": {"field": "이름", "type": "nominal", "title": None, "sort": [r["이름"] for r in rows],
                                "scale": {"range": [HIGHLIGHT, "#5b8fd1", "#86acd9", "#a9c3e3"][: len(rows) - (rest > 0)] + ([OTHER] if rest > 0 else [])}},
                      "order": {"field": "순서", "type": "quantitative"},
                      "tooltip": [*_tooltip(), {"field": "비중", "type": "quantitative", "title": "비중(%)"}]}},
    ]
    for n, r in enumerate(rows):
        r["순서"] = n
    spec.pop("width")
    return Chart(kind="pie", spec=spec, fact_ids=[f.id for f in [*parts, total, share]])


def _extreme(computed: Computed, by_id: dict[str, Fact]) -> Chart | None:
    ranking = computed.ranking
    if not ranking:
        return None
    names = _differing([(r["dims"], r["period"]) for r in ranking])
    digits = by_id[computed.picked["winner"]].round  # a share ranking shows 1 decimal, like its facts
    rows = []
    for n, (name, r) in enumerate(zip(names, ranking)):
        value = r["value"] if digits is None else round(r["value"], digits)
        rows.append({"이름": name, "값": value, "표시": format_number(value, r["unit"], digits), "강조": "예" if n == 0 else "아니오"})
    fact_ids = list(computed.picked.values())  # the winner and the runner-up are facts; the other candidates are rows only
    if all(r["dims"] == ranking[0]["dims"] for r in ranking):  # ranked over periods: keep time order
        order = sorted(range(len(rows)), key=lambda n: ranking[n]["period"] or "")
        spec = _timeline(f"{fact_label(Fact(id='x', dims=ranking[0]['dims'], note='title'))} 시점별", [rows[n] for n in order],
                         ranking[0]["unit"])
        return Chart(kind="timeline", spec=spec, fact_ids=fact_ids)
    title = _common([(r["dims"], r["period"]) for r in ranking])
    word = "비중" if by_id[computed.picked["winner"]].derivation else ""
    spec = _bars(f"{title} {word} 순위".replace("  ", " ").strip(), rows, ranking[0]["unit"], sort=[r["이름"] for r in rows])
    return Chart(kind="bar", spec=spec, fact_ids=fact_ids)


# ---------------------------------------------------------------- patterns over periods


def _span_name(term) -> str:
    return span_text(*term.periods)


def _one_term(computed: Computed, by_id: dict[str, Fact]) -> Chart | None:
    """avg / sum: the periods as bars, the aggregate in the title (every bar is a fact of the answer)."""
    term = computed.terms[0]
    cells = [by_id[i] for i in term.cell_ids]
    value = by_id[term.value_id]
    word = {"avg": "평균", "sum": "합계"}.get(term.agg or "", "")
    rows = [_row(f.period or "", f, True) for f in cells]
    title = f"{fact_label(cells[0])} {_span_name(term)} {word} {format_fact(value)}".replace("  ", " ").strip()
    return Chart(kind="timeline", spec=_timeline(title, rows, cells[0].unit), fact_ids=[f.id for f in cells] + [value.id])


def _two_terms(computed: Computed, by_id: dict[str, Fact]) -> Chart | None:
    """pavg / psum / vsa / shr: the two numbers compared as bars, the later one highlighted."""
    earlier, later = computed.terms
    a, b = by_id[earlier.value_id], by_id[later.value_id]
    word = {"avg": " 평균", "sum": " 합계"}
    names = [f"{_span_name(earlier)}{word.get(earlier.agg or '', '')}", f"{_span_name(later)}{word.get(later.agg or '', '')}"]
    if computed.pattern == "shr":
        names = [_span_name(earlier), _span_name(later)]
    rows = [_row(names[0], a, False), _row(names[1], b, True)]
    label = fact_label(by_id[earlier.cell_ids[0]])
    what = "비중" if computed.pattern == "shr" else ""
    title = f"{label} {what} 비교".replace("  ", " ").strip()
    spec = _bars(title, rows, a.unit, sort=names, horizontal=False)
    return Chart(kind="bar", spec=spec, fact_ids=[a.id, b.id])


def _gap(computed: Computed, by_id: dict[str, Fact]) -> Chart | None:
    """gap: the gap at the earlier and the later period as bars, the later one highlighted."""
    x1, y1, x2, _, g1, g2, _ = (by_id[i] for i in computed.main)
    names = [period_text(x1.period or ""), period_text(x2.period or "")]
    names_pair = _differing([(x1.dims or {}, x1.period), (y1.dims or {}, y1.period)])
    title = f"{names_pair[0]}·{names_pair[1]} {_common([(x1.dims or {}, None), (y1.dims or {}, None)])} 격차"
    spec = _bars(title.replace("  ", " ").strip(), [_row(names[0], g1, False), _row(names[1], g2, True)], g1.unit, sort=names,
                 horizontal=False)
    return Chart(kind="bar", spec=spec, fact_ids=[g1.id, g2.id])


def _grw(computed: Computed, by_id: dict[str, Fact]) -> Chart | None:
    """grw: each target's change (growth rate or amount) as bars, the one the answer names first highlighted."""
    a1, _, a2, _, v1, v2, d = (by_id[i] for i in computed.main)
    names = _differing([(a1.dims or {}, a1.period), (a2.dims or {}, a2.period)])
    subject = d.derivation.split("-")[0].strip()
    word = "증감률" if v1.unit == "%" else "증감"
    rows = [_row(names[0], v1, v1.id == subject), _row(names[1], v2, v2.id == subject)]
    rows = [{**r, "표시": format_number(f.value, f.unit, f.round)} for r, f in zip(rows, (v1, v2))]  # keep the sign here
    spec = _bars(f"{_common([(a1.dims or {}, None), (a2.dims or {}, None)])} {word} 비교".replace("  ", " ").strip(), rows,
                 v1.unit, sort=names, horizontal=False)
    return Chart(kind="bar", spec=spec, fact_ids=[v1.id, v2.id])


_CHARTS = {"value": _value, "trend": _trend, "compare": _compare, "share": _share, "extreme": _extreme,
           "avg": _one_term, "sum": _one_term, "pavg": _two_terms, "psum": _two_terms, "vsa": _two_terms, "shr": _two_terms,
           "gap": _gap, "grw": _grw}


def chart_spec(kind: QuestionType, computed: Computed) -> Chart | None:
    """The chart for the answer (kind, Vega-Lite v5 spec, the facts drawn), or None when there is nothing to draw."""
    by_id = {f.id: f for f in computed.result.facts}
    builder = _CHARTS[computed.pattern or kind]
    try:
        return builder(computed, by_id)
    except (KeyError, IndexError, StopIteration):  # an unexpected fact layout: answer without a chart
        return None
