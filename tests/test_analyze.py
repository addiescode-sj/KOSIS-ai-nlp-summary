"""Rule question analysis (T1-06): question -> Plan -> CalcRequest -> StatQuery, on the made-up household table."""

from dataclasses import replace

import pytest

from kosis_chat.compute import compute
from kosis_chat.compute.analyze import PERIOD_AXIS, Unresolved, build_request, rule_plan, to_query
from kosis_chat.data.loader import load_table
from kosis_chat.data.preprocessing import clean
from kosis_chat.judge.answerability import judge
from kosis_chat.judge.question import CHANGE_WORDS, build_index

REGION = "행정구역별(읍면동)"


@pytest.fixture
def table(household_path):
    return clean(load_table(household_path))


@pytest.fixture
def index(table):
    return build_index(table)


def ref(item, region, period):
    return f"FAKE_TBL_002|{item}|{region}|{period}"


def analyze(question, table, index):
    plan = rule_plan(question, index, judge(question, table, index).default_applied)
    return plan, build_request(plan, index, table)


def test_value_names_one_cell(table, index):
    plan, request = analyze("2023년 서울특별시 1인 가구 수는?", table, index)
    assert plan.type == "value" and request.cells == [ref("T1", "11", "2023")]
    query = to_query(plan, index, request)
    assert query.item == "1인 (가구)" and query.filters == {REGION: "서울특별시"} and query.periods == ["2023"]


def test_default_latest_value_shows_newer_blank_period(table, index):
    plan, request = analyze("서울 1인 가구 수는?", table, index)
    assert plan.default_applied["period"].startswith("latest (2024")
    assert request.cells == [ref("T1", "11", "2024")] and request.context == [ref("T1", "11", "2025")]


def test_trend_takes_every_period_in_range_without_gap_note(table, index):
    plan, request = analyze("2015년부터 2023년까지 부산 1인 가구는 어떻게 변했어?", table, index)
    assert plan.type == "trend" and request.cells == [ref("T1", "21", p) for p in ("2015", "2020", "2023")]
    assert request.note is None  # 2016~2019 are not in a census table by design


def test_trend_notes_a_missing_end_year(table, index):
    _, request = analyze("2015년부터 2022년까지 부산 1인 가구 추이", table, index)
    assert request.note == "2022년은 표에 없음"


def test_compare_keeps_named_reference(table, index):
    plan, request = analyze("2023년 서울이 전국보다 1인 가구가 얼마나 많아?", table, index)
    assert request.cells == [ref("T1", "11", "2023"), ref("T1", "00", "2023")] and request.reference == 1
    query = to_query(plan, index, request)
    assert query.targets == [{REGION: "서울특별시"}, {REGION: "전국"}] and query.reference == 1


def test_share_uses_item_total(table, index):
    _, request = analyze("2023년 서울의 1인 가구 비중은?", table, index)
    assert request.cells == [ref("T1", "11", "2023")] and request.total == ref("T0", "11", "2023")


def test_extreme_over_provinces_with_country_as_context(table, index):
    plan, request = analyze("2023년 1인 가구가 가장 많은 시도는?", table, index)
    assert plan.over == REGION and plan.pick == "max"
    assert request.candidates == [ref("T1", r, "2023") for r in ("11", "21", "39")]  # not 동부 or 시군구
    assert request.context == [ref("T1", "00", "2023")]
    winner = compute(table, request).picked["winner"]
    assert winner == "f1" and compute(table, request).result.facts[0].source_ref == ref("T1", "39", "2023")


def test_extreme_over_periods(table, index):
    plan, request = analyze("서울 1인 가구가 가장 적었던 해는?", table, index)
    assert plan.over == PERIOD_AXIS and plan.pick == "min"
    assert ref("T1", "11", "2015") in request.candidates


def test_most_recent_is_a_period_not_a_ranking(table, index):
    plan, request = analyze("가장 최근 서울특별시 1인 가구 수는?", table, index)
    assert plan.type == "value" and request.cells == [ref("T1", "11", "2024")]


def test_extreme_without_ranked_axis_is_unresolved_not_a_crash(table, index):
    plan, _ = analyze("2023년 1인 가구가 가장 많은 시도는?", table, index)
    with pytest.raises(Unresolved) as exc:
        build_request(replace(plan, over=None), index, table)  # what a model fill with over=null leaves
    assert exc.value.what == ["over"]


def test_change_without_period_is_a_recent_trend(table, index):
    plan, request = analyze("서울특별시 1인 가구는 얼마나 늘었나?", table, index)
    assert plan.type == "trend" and plan.default_applied["period"].startswith("recent 4 points")
    assert request.cells == [ref("T1", "11", p) for p in ("2015", "2020", "2023", "2024")]


def test_change_with_one_year_ends_at_that_year(table, index):
    plan, request = analyze("2023년 서울특별시 1인 가구는 얼마나 늘었나?", table, index)
    assert plan.type == "trend" and request.cells == [ref("T1", "11", p) for p in ("2015", "2020", "2023")]


def test_change_words_skip_labels():
    assert CHANGE_WORDS.search("얼마나 늘었어") and CHANGE_WORDS.search("감소했나요")
    assert not CHANGE_WORDS.search("2024년 부산의 자연증가건수는 몇 명이야?")


def test_item_total_fills_an_unnamed_item(table, index):
    _, request = analyze("2023년 서울 가구는 몇이야?", table, index)
    assert request.cells == [ref("T0", "11", "2023")]  # 일반가구_계


def test_unnamed_item_without_total_is_unresolved(pop_path):
    pop = clean(load_table(pop_path))
    plan = rule_plan("2023년 남자는?", build_index(pop))
    with pytest.raises(Unresolved) as exc:
        build_request(plan, build_index(pop), pop)
    assert exc.value.what == ["항목"]


# ---------------------------------------------------------------- T0-03 v2: ops over periods


def test_average_over_a_range(table, index):
    plan, request = analyze("2015년부터 2024년까지 서울특별시 1인 가구의 연평균은?", table, index)
    assert plan.type == "trend" and plan.op == "avg" and plan.groups == [["2015", "2020", "2023", "2024"]]
    assert request.op == "avg" and request.terms[0].cells == [ref("T1", "11", p) for p in ("2015", "2020", "2023", "2024")]
    query = to_query(plan, index, request)
    assert query.op == "avg" and query.period_groups == plan.groups and query.periods == plan.groups[0]


def test_two_ranges_compared_and_value_against_average(table, index):
    plan, request = analyze("서울특별시 1인 가구는 2015~2020년 평균보다 2023~2024년 평균이 얼마나 많아?", table, index)
    assert plan.op == "pavg" and [t.cells for t in request.terms] == [[ref("T1", "11", "2015"), ref("T1", "11", "2020")], [ref("T1", "11", "2023"), ref("T1", "11", "2024")]]
    plan, request = analyze("2024년 서울특별시 1인 가구는 2015~2023년 평균보다 얼마나 많아?", table, index)
    assert plan.op == "vsa" and request.terms[0].agg == "avg" and request.terms[1].cells == [ref("T1", "11", "2024")]
    assert to_query(plan, index, request).periods == ["2015", "2020", "2023", "2024"]


def test_share_at_two_periods(table, index):
    plan, request = analyze("전국 일반가구 중 1인 가구 비중은 2015년보다 2024년에 몇 %p 늘었어?", table, index)
    assert plan.type == "share" and plan.op == "shr" and plan.groups == [["2015"], ["2024"]]
    assert [(t.cells, t.total) for t in request.terms] == [([ref("T1", "00", "2015")], ref("T0", "00", "2015")), ([ref("T1", "00", "2024")], ref("T0", "00", "2024"))]


def test_plain_words_do_not_start_an_op(table, index):
    plan, request = analyze("2023년 서울특별시 1인 가구는 전국 평균보다 많아?", table, index)
    assert plan.op is None and request.op is None  # "평균" names the reference, there is no period range
    plan, request = analyze("2015년부터 2024년까지 서울특별시 1인 가구 추이는?", table, index)
    assert plan.op is None and request.type == "trend" and len(request.cells) == 4


def test_previous_year_compares_the_last_two_periods(table, index):
    plan, request = analyze("작년 대비 서울특별시 1인 가구는 몇 % 늘었어?", table, index)
    assert plan.default_applied == {"period": "latest (2024) vs previous (2023)"}  # 2025 is blank
    assert request.type == "trend" and request.cells == [ref("T1", "11", "2023"), ref("T1", "11", "2024")]
