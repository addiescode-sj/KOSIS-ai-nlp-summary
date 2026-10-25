"""A Vega-Lite chart for each answer, built by code from the computed facts (the model never touches it).

value: one bar · trend: a line over every period · compare: two bars · share: a pie of the parts and the rest ·
extreme: every candidate as a sorted bar. The asked value is highlighted; numbers are the facts' own values.
Render with any Vega-Lite v5 viewer (vega-embed, Streamlit st.vega_lite_chart, Altair); eval/view_run.py shows them.
"""

from __future__ import annotations

from kosis_chat.answer.render import fact_label, format_fact, item_label
from kosis_chat.compute import Computed

from kosis_ai.schemas import Fact, QuestionType  # isort: skip

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


def _value(computed: Computed, by_id: dict[str, Fact]) -> dict | None:
    f = by_id[computed.main[0]]
    if len(computed.series) < 2:
        return _bars(_common([(f.dims or {}, f.period)]) or "값", [_row(fact_label(f) or f"{f.period}", f, True)], f.unit)
    rows = [_row(p["period"], f.model_copy(update={"value": p["value"]}), p["period"] == f.period) for p in computed.series]
    return _timeline(f"{fact_label(f)} ({f.period}년 강조)".strip(), rows, f.unit)


def _trend(computed: Computed, by_id: dict[str, Fact]) -> dict | None:
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
    return spec


def _compare(computed: Computed, by_id: dict[str, Fact]) -> dict | None:
    a, b, diff = (by_id[i] for i in computed.main[:3])
    subject = diff.derivation.split("-")[0].strip()
    names = _differing([(a.dims or {}, a.period), (b.dims or {}, b.period)])
    rows = [_row(names[0], a, a.id == subject), _row(names[1], b, b.id == subject)]
    return _bars(f"{_common([(a.dims or {}, a.period), (b.dims or {}, b.period)])} 비교".strip(), rows, a.unit,
                 sort=names, horizontal=False)


def _share(computed: Computed, by_id: dict[str, Fact]) -> dict | None:
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
    return spec


def _extreme(computed: Computed, by_id: dict[str, Fact]) -> dict | None:
    ranking = computed.ranking
    if not ranking:
        return None
    names = _differing([(r["dims"], r["period"]) for r in ranking])
    digits = by_id[computed.picked["winner"]].round  # a share ranking shows 1 decimal, like its facts
    rows = []
    for n, (name, r) in enumerate(zip(names, ranking)):
        value = r["value"] if digits is None else round(r["value"], digits)
        shown = Fact(id="x", value=value, unit=r["unit"], source_ref="chart", round=digits)
        rows.append({"이름": name, "값": value, "표시": format_fact(shown), "강조": "예" if n == 0 else "아니오"})
    if all(r["dims"] == ranking[0]["dims"] for r in ranking):  # ranked over periods: keep time order
        order = sorted(range(len(rows)), key=lambda n: ranking[n]["period"] or "")
        return _timeline(f"{fact_label(Fact(id='x', dims=ranking[0]['dims'], note='title'))} 시점별", [rows[n] for n in order],
                         ranking[0]["unit"])
    title = _common([(r["dims"], r["period"]) for r in ranking])
    word = "비중" if by_id[computed.picked["winner"]].derivation else ""
    return _bars(f"{title} {word} 순위".replace("  ", " ").strip(), rows, ranking[0]["unit"], sort=[r["이름"] for r in rows])


_CHARTS = {"value": _value, "trend": _trend, "compare": _compare, "share": _share, "extreme": _extreme}


def chart_spec(kind: QuestionType, computed: Computed) -> dict | None:
    """Vega-Lite v5 spec for the answer, or None when there is nothing to draw."""
    by_id = {f.id: f for f in computed.result.facts}
    try:
        return _CHARTS[kind](computed, by_id)
    except (KeyError, IndexError, StopIteration):  # an unexpected fact layout: answer without a chart
        return None
