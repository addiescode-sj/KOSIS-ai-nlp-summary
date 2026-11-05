"""ChartSpec builders and to_vega (T1-28 S5): the chart is data (kind, points, the facts drawn); Vega-Lite is drawn from it.

The drawn Vega-Lite is the spec the builders made before S5: the mock golden run compares it chart by chart with the
S0 run (eval.parity --mode semantic prints "vega_lite ... identical"); here, made-up tables only.
"""

from __future__ import annotations

import json

import pytest

from kosis_chat.chart import builders, vega
from kosis_chat.chart.builders import BUILDERS, _CHARTS, chart_spec, register_builder
from kosis_chat.chart.vega import DRAW, SCHEMA, register_kind, to_vega, with_vega
from kosis_chat.compute import build, compute
from kosis_chat.data.loader import load_table
from kosis_chat.data.preprocessing import clean

from kosis_ai.schemas import ChartPoint, ChartSeries, ChartSpec  # isort: skip


def ref(item, region, period):
    return f"FAKE_TBL_002|{item}|{region}|{period}"


@pytest.fixture
def households(household_path):
    return clean(load_table(household_path))


CASES = [
    ("value", lambda: build.value(ref("T1", "11", "2023")), "timeline"),
    ("trend", lambda: build.trend([ref("T1", "21", p) for p in ("2015", "2020", "2023")]), "line"),
    ("compare", lambda: build.compare([ref("T1", "11", "2023"), ref("T1", "00", "2023")], 1), "bar"),
    ("share", lambda: build.share([ref("T1", "11", "2023")], ref("T0", "11", "2023")), "pie"),
    ("extreme", lambda: build.extreme([ref("T1", r, "2023") for r in ("11", "21", "39")], "max"), "bar"),
    ("compare", lambda: build.gap((ref("T1", "11", "2015"), ref("T1", "21", "2015")),
                                  (ref("T1", "11", "2024"), ref("T1", "21", "2024"))), "grouped_bar"),
]


@pytest.mark.parametrize("kind, built, chart_kind", CASES)
def test_every_kind_is_data_and_draws_its_rows(households, kind, built, chart_kind):
    computed = compute(households, built())
    chart = chart_spec(kind, computed)
    assert isinstance(chart, ChartSpec) and chart.kind == chart_kind and chart.vega_lite is None
    assert ChartSpec.model_validate_json(chart.model_dump_json()) == chart  # plain JSON
    facts = {f.id: f for f in computed.result.facts}
    for point in chart.points():  # a point names a fact only when it shows that fact's own value
        if point.fact_id is not None:
            assert point.fact_id in chart.fact_ids and facts[point.fact_id].value == point.value
    spec = to_vega(chart)
    rows = spec["data"]["values"]
    assert spec["$schema"] == SCHEMA and spec["title"] == chart.title and len(rows) == len(chart.points())
    assert [r["값"] for r in rows] == [p.value for p in chart.points()]
    assert [r["표시"] for r in rows] == [p.display for p in chart.points()]
    assert [r["강조"] == "예" for r in rows] == [p.highlight for p in chart.points()]
    json.dumps(spec)


def test_bars_keep_their_layout(households):
    chart = chart_spec("compare", compute(households, build.compare([ref("T1", "11", "2023"), ref("T1", "00", "2023")], 1)))
    assert chart.orientation == "vertical" and chart.sort == [p.label for p in chart.points()]
    spec = to_vega(chart)
    assert spec["layer"][0]["encoding"]["x"]["sort"] == chart.sort and "height" not in spec
    ranked = chart_spec("extreme", compute(households, build.extreme([ref("T1", r, "2023") for r in ("11", "21", "39")], "max")))
    assert ranked.orientation == "horizontal" and to_vega(ranked)["height"] == {"step": 22}


def test_a_pie_draws_the_rest_as_the_one_plain_slice(households):
    chart = chart_spec("share", compute(households, build.share([ref("T1", "11", "2023")], ref("T0", "11", "2023"))))
    rest = [p for p in chart.points() if not p.highlight]
    assert [p.label for p in rest] == ["그 외"] and rest[0].fact_id is None and rest[0].share_pct is not None
    spec = to_vega(chart)
    assert "width" not in spec and [r["순서"] for r in spec["data"]["values"]] == [0, 1]
    assert spec["layer"][0]["encoding"]["color"]["scale"]["range"][-1] == vega.OTHER


def test_grid_points_are_periods_of_each_target(households):
    computed = compute(households, build.gap((ref("T1", "11", "2015"), ref("T1", "21", "2015")),
                                             (ref("T1", "11", "2024"), ref("T1", "21", "2024"))))
    chart = chart_spec("compare", computed)
    assert [s.name for s in chart.series] == ["서울특별시", "부산광역시"]
    assert [p.label for p in chart.series[0].points] == ["2015년", "2024년"] and chart.series[0].points[0].period == "2015"
    rows = to_vega(chart)["data"]["values"]
    assert list(rows[0]) == ["대상", "시점", "이름", "값", "표시", "강조"] and rows[0]["이름"] == rows[0]["대상"] == "서울특별시"


def test_the_registries(households):
    assert BUILDERS is _CHARTS and set(DRAW) == {"bar", "timeline", "line", "pie", "grouped_bar", "multi_line"}
    point = ChartPoint(label="a", value=1, display="1", fact_id="f1")
    odd = ChartSpec(kind="scatter", title="t", series=[ChartSeries(name="t", points=[point])], fact_ids=["f1"])
    assert to_vega(odd) is None  # a kind nobody draws has no Vega-Lite; the data is still the chart
    saved = dict(DRAW)
    try:
        register_kind("scatter", lambda spec: {"mark": "point", "data": {"values": [p.model_dump() for p in spec.points()]}})
        assert to_vega(odd)["mark"] == "point" and with_vega(odd).vega_lite == to_vega(odd)
    finally:
        DRAW.clear()
        DRAW.update(saved)
    saved_builders = dict(BUILDERS)
    try:
        register_builder("value", lambda computed, by_id: None)
        assert chart_spec("value", compute(households, build.value(ref("T1", "11", "2023")))) is None
    finally:
        BUILDERS.clear()
        BUILDERS.update(saved_builders)
    assert builders.BUILDERS["value"] is builders._value


def test_with_vega_keeps_a_given_spec():
    point = ChartPoint(label="a", value=1, display="1")
    given = ChartSpec(kind="bar", title="t", series=[ChartSeries(name="t", points=[point])], fact_ids=["f1"],
                      vega_lite={"mark": "bar"})
    assert with_vega(given) is given and to_vega(given) == {"mark": "bar"}


def test_a_share_ranking_names_its_share_facts(households):
    """Review 11/05 (S-EXT-04): a share ranking's points are the derived share facts, linked by rank place."""
    regions = ("11", "21", "39")
    computed = compute(households, build.extreme([ref("T1", r, "2023") for r in regions], "max",
                                                 totals=[ref("T0", r, "2023") for r in regions]))
    chart = chart_spec("extreme", computed)
    facts = {f.id: f for f in computed.result.facts}
    picked = list(computed.picked.values())
    linked = [p.fact_id for p in chart.points()]
    assert linked[: len(picked)] == picked and all(facts[i].derivation for i in picked)
    assert all(i is None for i in linked[len(picked):])  # the other candidates are rows only
    assert [p.value for p in chart.points()[: len(picked)]] == [facts[i].value for i in picked]
