"""Vega-Lite pieces of the answer charts (T1-28 S3a: split from answer/chart.py): the spec frame, the colour and
tooltip encodings, bars and a timeline, and one data row per fact. chart/builders.py picks the facts."""

from __future__ import annotations

from kosis_chat.shared.numbers import format_fact

from kosis_ai.schemas import Fact  # isort: skip

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
