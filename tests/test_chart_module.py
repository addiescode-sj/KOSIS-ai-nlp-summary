"""The chart module (T1-32): the question + the computed facts -> whether to draw, which kind, and why.

A made-up table (가상 데이터, not KOSIS values): 전국 + 17 시도, 15 irregular years 1990~2024, household counts, a
share (%) and an item with tied values.
"""

from __future__ import annotations

import json

import pytest

from conftest import write_table
from kosis_chat.chart.check import check
from kosis_chat.chart.intent import read_intent
from kosis_chat.chart.module import chart_data, make_chart, redraw
from kosis_chat.chart.node import step_chart
from kosis_chat.chart.schemas import ChartIn
from kosis_chat.compute import build, compute

REGIONS = [("00", "전국")] + [(f"{10 + i}", f"시도{i:02d}") for i in range(1, 18)]
SIDO = [c for c, _ in REGIONS[1:]]
YEARS = ["1990", "1995", "2000", "2005", "2010", "2015", "2016", "2017", "2018", "2019", "2020", "2021", "2022", "2023", "2024"]
ITEMS = [("T0", "총가구 (가구)", None), ("T1", "1인 (가구)", None), ("T2", "2인 (가구)", None),
         ("P1", "1인가구 비중 (%)", "%"), ("TIE", "동률 (명)", "명")]


def _value(item: str, ri: int, yi: int) -> float:
    base = 100000 * (ri + 1)
    return {"T0": base, "T1": base * 0.3 + yi * 100, "T2": base * 0.25, "P1": 30 + yi * 0.5 + ri * 0.1,
            "TIE": 500 if ri in (3, 7) else 100 + ri}[item]


@pytest.fixture(scope="module")
def table(tmp_path_factory):
    from kosis_chat.data.loader import load_table
    from kosis_chat.data.preprocessing import clean

    rows = [{"period": y, "item_id": it, "item_name": name, "unit": unit, "dim1_name": "행정구역별", "dim1_code": c,
             "dim1_label": label, "value_raw": str(_value(it, ri, yi)), "value": _value(it, ri, yi)}
            for yi, y in enumerate(YEARS) for ri, (c, label) in enumerate(REGIONS) for it, name, unit in ITEMS]
    path = write_table(tmp_path_factory.mktemp("data"), "FAKE", rows,
                       {"survey_name": "가상", "organization": "가상", "table_name": "가상 가구", "period": "년",
                        "url": "https://example.invalid/x"})
    return clean(load_table(path))


def ref(item: str, region: str, period: str) -> str:
    return f"FAKE|{item}|{region}|{period}"


def run(table, question, built, qtype=None, **kw):
    return make_chart(question, compute(table, built), qtype, **kw)


def facts_of(table, built):
    return {f.id: f.value for f in compute(table, built).result.facts}


# ---------------------------------------------------------------- 질문 해석

@pytest.mark.parametrize("q, requested, kind, suppress, cue", [
    ("서울 1인가구 추이 그래프로 보여줘", True, None, False, "time"),
    ("부산이랑 대구 막대그래프로 비교해줘", True, "bar", False, "compare"),
    ("1인가구 비중 원형 차트로", True, "pie", False, "composition"),
    ("비중 변화를 누적 막대로 그려줘", True, "stacked", False, "time"),
    ("가장 높은 시도는? 그래프는 빼고", False, None, True, "ranking"),
    ("2023년 서울 1인가구 수는?", False, None, False, None),
    ("숫자만 알려줘", False, None, True, None),
    ("2023년 누적 확진자 수는?", False, None, False, None),  # 누적 alone is a number, not a chart
    ("누적 막대그래프로", True, "stacked", False, None),
])
def test_intent(q, requested, kind, suppress, cue):
    i = read_intent(q)
    assert (i.requested, i.kind, i.suppress, i.cue) == (requested, kind, suppress, cue)


# ---------------------------------------------------------------- 자동 선택 (결정 표)

AUTO = [
    ("서울 1인가구 2015년부터 2024년까지 추이", lambda: build.trend([ref("T1", "11", y) for y in YEARS[5:]]), "trend", "line"),
    ("서울과 부산 1인가구 비교", lambda: build.compare([ref("T1", "11", "2023"), ref("T1", "12", "2023")]), "compare", "bar"),
    ("서울 1인가구 비중", lambda: build.share([ref("T1", "11", "2023")], ref("T0", "11", "2023")), "share", "pie"),
    ("1인가구가 가장 많은 시도", lambda: build.extreme([ref("T1", g, "2023") for g in SIDO], "max"), "extreme", "bar"),
    ("서울 1인가구가 가장 많았던 해", lambda: build.extreme([ref("T1", "11", y) for y in YEARS], "max"), "extreme", "line"),
    ("최근 5년 평균", lambda: build.over_periods([ref("T1", "11", y) for y in YEARS[-5:]], "avg"), "trend", "line"),
    ("2015년과 2024년 1인가구 비중 변화", lambda: build.share_change(([ref("T1", "11", "2015")], ref("T0", "11", "2015")),
                                                          ([ref("T1", "11", "2024")], ref("T0", "11", "2024"))), "share", "stacked_bar"),
    ("2023년 서울 1인가구", lambda: build.value(ref("T1", "11", "2023")), "value", "timeline"),
    ("2023년 서울 1인가구 연도별로", lambda: build.value(ref("T1", "11", "2023")), "value", "line"),  # 단서 time
]


@pytest.mark.parametrize("q, built, qtype, kind", AUTO)
def test_auto_kind_and_numbers_are_facts_or_table_cells(table, q, built, qtype, kind):
    out = run(table, q, built(), qtype)
    assert out.drawn and out.decision == "auto" and out.kind == kind, out.reasons
    facts = facts_of(table, built())
    for p in out.spec.points():  # 사실 점은 사실 값 그대로
        if p.fact_id:
            assert facts[p.fact_id] == p.value
    json.dumps(chart_data(out.spec))


def test_growth_draws_the_rates_not_the_levels(table):
    built = build.growth((ref("T1", "11", "2015"), ref("T1", "11", "2024")), (ref("T1", "12", "2015"), ref("T1", "12", "2024")))
    out = run(table, "서울과 부산 중 1인가구 증가율이 더 큰 곳", built, "compare")
    assert out.drawn and out.kind == "bar" and out.spec.unit == "%"
    assert len(out.spec.points()) == 2


# ---------------------------------------------------------------- 요청 종류와 가드

def test_requested_line_on_regions_falls_back_with_a_reason(table):
    out = run(table, "서울과 부산 꺾은선으로", build.compare([ref("T1", "11", "2023"), ref("T1", "12", "2023")]), "compare")
    assert out.drawn and out.decision == "fallback" and out.kind == "bar"
    assert "시점이 이어지는" in out.note


def test_requested_pie_on_ranking_falls_back(table):
    out = run(table, "가장 많은 시도 원형 차트로", build.extreme([ref("T1", g, "2023") for g in SIDO], "max"), "extreme")
    assert out.kind == "bar" and "구성비" in out.note


def test_requested_bar_on_trend_is_a_timeline(table):
    out = run(table, "서울 추이 막대로 그려줘", build.trend([ref("T1", "11", y) for y in YEARS[-5:]]), "trend")
    assert out.decision == "requested" and out.kind == "timeline"


def test_requested_line_on_two_periods_of_one_target(table):
    out = run(table, "2015년과 2024년 서울 비교 선 그래프로", build.compare([ref("T1", "11", "2015"), ref("T1", "11", "2024")]), "compare")
    assert out.decision == "requested" and out.kind == "line"


# ---------------------------------------------------------------- 그리지 않음

def test_user_suppressed(table):
    out = run(table, "가장 많은 시도는? 그래프는 빼고", build.extreme([ref("T1", g, "2023") for g in SIDO], "max"), "extreme")
    assert not out.drawn and out.reasons == ["user_suppressed"]


def test_not_requested_and_auto_off(table):
    out = run(table, "서울 추이", build.trend([ref("T1", "11", y) for y in YEARS[-3:]]), "trend", auto_draw=False)
    assert not out.drawn and out.reasons == ["not_requested"]


def test_builder_error_never_breaks_the_answer(table, monkeypatch):
    import kosis_chat.chart.decide as decide

    def boom(*a, **k):
        raise ZeroDivisionError

    monkeypatch.setattr(decide, "chart_spec", boom)
    out = run(table, "서울 추이 그래프", build.trend([ref("T1", "11", y) for y in YEARS[-3:]]), "trend")
    assert not out.drawn and out.reasons == ["builder_error:ZeroDivisionError"] and out.note


def test_check_skips_a_single_point(table):
    spec = run(table, "서울과 부산", build.compare([ref("T1", "11", "2023"), ref("T1", "12", "2023")]), "compare").spec
    one = spec.model_copy(update={"series": [spec.series[0].model_copy(update={"points": spec.series[0].points[:1]})]})
    assert check(one, "bar").action == "skip"


def test_check_skips_a_chart_without_any_fact(table):
    spec = run(table, "서울과 부산", build.compare([ref("T1", "11", "2023"), ref("T1", "12", "2023")]), "compare").spec
    pts = [p.model_copy(update={"fact_id": None}) for p in spec.series[0].points]
    bare = spec.model_copy(update={"series": [spec.series[0].model_copy(update={"points": pts})]})
    assert check(bare, "bar").reason == "no_fact_point"


# ---------------------------------------------------------------- 다듬기

def test_ranking_is_cut_to_top_ten_and_keeps_the_facts(table):
    built = build.extreme([ref("T1", g, "2023") for g in SIDO], "max")
    out = run(table, "1인가구가 가장 많은 시도", built, "extreme")
    pts = out.spec.points()
    assert len(pts) == 10 and pts[0].highlight and "상위 10개" in out.note
    assert out.spec.sort == [p.label for p in pts] or out.spec.sort == "-x"


def test_tied_winners_are_all_highlighted(table):
    out = run(table, "동률이 가장 큰 시도", build.extreme([ref("TIE", g, "2023") for g in SIDO], "max"), "extreme")
    assert sum(p.highlight for p in out.spec.points()) == 2


# ---------------------------------------------------------------- the node, redraw, the screen's data

def test_the_node_passes_the_question(table):
    computed = compute(table, build.extreme([ref("T1", g, "2023") for g in SIDO], "max"))
    out = step_chart(ChartIn(type="extreme", computed=computed, question="가장 많은 시도 원형 차트로"), None)
    assert out.chart.kind == "bar" and "구성비" in out.note and out.chart.vega_lite is None
    assert step_chart(ChartIn(type="extreme", computed=computed, question="그래프는 빼고"), None).chart is None
    assert step_chart(ChartIn(type="extreme", computed=computed), None).chart.kind == "bar"  # no question: the table


def test_redraw_keeps_the_kind_and_draws_the_asked_one(table):
    spec = run(table, "서울 추이", build.trend([ref("T1", "11", y) for y in YEARS[-5:]]), "trend").spec
    assert redraw(spec).kind == spec.kind == "line"
    bars = redraw(spec, "bar")
    assert bars.decision == "requested" and bars.kind == "timeline" and bars.drawn
    pie = redraw(spec, "pie")
    assert pie.kind == "line" and "구성비" in pie.note  # not a whole's parts: kept as drawn
    assert redraw(None, "bar").note and not redraw(None, "bar").drawn


def test_stacked_bar_data_has_the_share_and_the_rest(table):
    built = build.share_change(([ref("T1", "11", "2015")], ref("T0", "11", "2015")),
                               ([ref("T1", "11", "2024")], ref("T0", "11", "2024")))
    out = run(table, "2015년과 2024년 1인가구 비중 변화", built, "share")
    assert out.kind == "stacked_bar"
    data = chart_data(out.spec)
    assert "vega_lite" not in data and [s["name"] for s in data["series"]][1] == "그 외"
    share, rest = data["series"]
    assert all(abs(a["value"] + b["value"] - 100) < 0.11 for a, b in zip(share["points"], rest["points"]))
    facts = facts_of(table, built)
    assert all(facts[p["fact_id"]] == p["value"] for p in share["points"] if p["fact_id"])
    assert all(p["fact_id"] is None for p in rest["points"])  # the rest is not a fact


def test_targets_at_one_period_are_no_time_axis(table):
    out = run(table, "세 지역 꺾은선으로", build.extreme([ref("T2", g, "2020") for g in SIDO[:3]], "min"), "extreme")
    assert out.decision == "fallback" and out.kind == "bar" and "시점" in out.note
