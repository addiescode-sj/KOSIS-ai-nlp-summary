"""Vega-Lite pieces of the answer charts (T1-28 S3a: split from answer/chart.py): the spec frame, the colour and
tooltip encodings, bars and a timeline, and one data row per fact. chart/builders.py picks the facts.

T1-28 S5: to_vega(spec) draws a ChartSpec (chart/builders.py) as Vega-Lite v5, the same spec the builders made before
S5 (tests/test_chart_spec.py). DRAW is the kind -> drawing table (E6); a kind it does not know has no Vega-Lite (None).
"""

from __future__ import annotations

from kosis_chat.shared.numbers import format_fact

from collections.abc import Callable

from kosis_ai.schemas import ChartPoint, ChartSpec, Fact  # isort: skip

SCHEMA = "https://vega.github.io/schema/vega-lite/v5.json"
HIGHLIGHT, OTHER = "#2f6db5", "#b8c4d6"
WIDTH = 360


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


# ---------------------------------------------------------------- ChartSpec -> Vega-Lite (T1-28 S5)


def _point_row(p: ChartPoint) -> dict:
    return {"이름": p.label, "값": p.value, "표시": p.display, "강조": "예" if p.highlight else "아니오"}


def _draw_bar(spec: ChartSpec) -> dict:
    rows = [_point_row(p) for p in spec.points()]
    return _bars(spec.title, rows, spec.unit, sort="-x" if spec.sort is None else spec.sort,
                 horizontal=spec.orientation != "vertical")


def _draw_timeline(spec: ChartSpec) -> dict:
    return _timeline(spec.title, [_point_row(p) for p in spec.points()], spec.unit)


def _draw_line(spec: ChartSpec) -> dict:
    rows = [{"시점": p.period, **_point_row(p)} for p in spec.points()]
    out = _base(spec.title, rows)
    x = {"field": "시점", "type": "ordinal", "title": None, "axis": {"labelAngle": 0}}
    y = {"field": "값", "type": "quantitative", "title": _unit_title(spec.unit), "scale": {"zero": False}}
    out["layer"] = [
        {"mark": {"type": "line", "color": OTHER}, "encoding": {"x": x, "y": y}},
        {"mark": {"type": "point", "filled": True, "size": 60}, "encoding": {"x": x, "y": y, "color": _color(), "tooltip": _tooltip()}},
        {"transform": [{"filter": "datum['강조'] == '예'"}], "mark": {"type": "text", "dy": -10},
         "encoding": {"x": x, "y": y, "text": {"field": "표시"}}},
    ]
    return out


def _draw_pie(spec: ChartSpec) -> dict:
    points = spec.points()
    rows = [{**_point_row(p), "비중": p.share_pct} for p in points]
    rest = any(not p.highlight for p in points)  # the rest of the total ("그 외") is the one slice not highlighted
    out = _base(spec.title, rows)
    out["layer"] = [
        {"mark": {"type": "arc", "innerRadius": 50, "stroke": "white"},
         "encoding": {"theta": {"field": "값", "type": "quantitative", "stack": True},
                      "color": {"field": "이름", "type": "nominal", "title": None, "sort": [r["이름"] for r in rows],
                                "scale": {"range": [HIGHLIGHT, "#5b8fd1", "#86acd9", "#a9c3e3"][: len(rows) - rest] + ([OTHER] if rest else [])}},
                      "order": {"field": "순서", "type": "quantitative"},
                      "tooltip": [*_tooltip(), {"field": "비중", "type": "quantitative", "title": "비중(%)"}]}},
    ]
    for n, r in enumerate(rows):
        r["순서"] = n
    out.pop("width")
    return out


def _draw_grid(spec: ChartSpec) -> dict:
    """grouped_bar / multi_line: one series per target, a point per period (the label is the period text)."""
    rows = [{"대상": s.name, "시점": p.label, **_point_row(p), "이름": s.name} for s in spec.series for p in s.points]
    out = _base(spec.title, rows)
    y = {"field": "값", "type": "quantitative", "title": _unit_title(spec.unit)}
    color = {"field": "대상", "type": "nominal", "title": None}
    tooltip = [{"field": "대상", "type": "nominal"}, {"field": "시점", "type": "ordinal"}, {"field": "표시", "type": "nominal", "title": "값"}]
    if spec.kind == "grouped_bar":
        out["mark"] = {"type": "bar"}
        out["encoding"] = {"x": {"field": "시점", "type": "ordinal", "title": None, "axis": {"labelAngle": 0}},
                           "xOffset": {"field": "대상"}, "y": y, "color": color, "tooltip": tooltip}
    else:
        out["mark"] = {"type": "line", "point": True}
        out["encoding"] = {"x": {"field": "시점", "type": "ordinal", "title": None, "sort": None}, "y": {**y, "scale": {"zero": False}},
                           "color": color, "tooltip": tooltip}
    return out


DRAW: dict[str, Callable[[ChartSpec], dict]] = {"bar": _draw_bar, "timeline": _draw_timeline, "line": _draw_line,
                                                "pie": _draw_pie, "grouped_bar": _draw_grid, "multi_line": _draw_grid}


def register_kind(kind: str, draw: Callable[[ChartSpec], dict]) -> None:
    """Add or replace how a chart kind is drawn as Vega-Lite (E6)."""
    DRAW[kind] = draw


def to_vega(spec: ChartSpec) -> dict | None:
    """The Vega-Lite v5 spec of a ChartSpec (its own vega_lite when it has one); None for a kind DRAW does not know."""
    if spec.vega_lite is not None:
        return spec.vega_lite
    draw = DRAW.get(spec.kind)
    return draw(spec) if draw is not None else None


def with_vega(spec: ChartSpec) -> ChartSpec:
    """The ChartSpec with vega_lite filled in (what run_eval records)."""
    return spec if spec.vega_lite is not None else spec.model_copy(update={"vega_lite": to_vega(spec)})
