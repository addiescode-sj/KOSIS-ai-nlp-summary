"""Rule-based answerability (T1-08) on made-up tables: each rule fires on its case and stays quiet on a plain question."""

from __future__ import annotations

import re

import pytest

from kosis_chat.data.events import event_key, load_events
from kosis_chat.data.loader import load_table
from kosis_chat.judge.answerability import judge
from kosis_chat.judge.question import asked_periods, build_index, match_labels
from tests.conftest import write_table


@pytest.fixture
def pop(pop_path):
    return load_table(pop_path)


@pytest.fixture
def households(household_path):
    return load_table(household_path)


@pytest.fixture
def projection(tmp_path):
    """가상 장래추계: scenarios with no total (고위/중위/저위), 총인구 in 명 and 노년부양비 with no unit."""
    rows = []
    for s_code, scenario in (("1", "중위 추계(출산율-중위 / 기대수명-중위)"), ("2", "고위 추계(출산율-고위)"), ("3", "저위 추계(출산율-저위)")):
        for m_code, measure in (("A", "총인구(명)"), ("B", "노년부양비")):
            for year in ("2030", "2050"):
                rows.append({"period": year, "item_id": "T10", "item_name": "인구항목", "unit": None,
                             "dim1_name": "가정별", "dim1_code": s_code, "dim1_label": scenario,
                             "dim2_name": "인구구조부양비별", "dim2_code": m_code, "dim2_label": measure,
                             "value_raw": "100", "value": 100})
    path = write_table(tmp_path / "data", "FAKE_PROJ", rows, {"survey_name": "가상 장래추계", "organization": "가상 기관",
                       "table_name": "가상 장래인구", "period": "년 (2030 ~ 2050)", "url": "https://example.invalid/proj"})
    return load_table(path)


@pytest.fixture
def new_region(tmp_path):
    """가상 시도별 혼인: 세종 has rows from 2012 only, as in DT_1B8000H."""
    rows = [{"period": year, "item_id": "T40", "item_name": "혼인건수 (건)", "unit": None, "dim1_name": "행정구역별",
             "dim1_code": code, "dim1_label": label, "value_raw": "10", "value": 10}
            for year in ("2009", "2012") for code, label in (("00", "전국"), ("29", "세종특별자치시"))
            if not (code == "29" and year == "2009")]
    path = write_table(tmp_path / "data", "FAKE_REGION", rows, {"survey_name": "가상 인구동향", "organization": "가상 기관",
                       "table_name": "가상 혼인", "period": "년 (2009 ~ 2012)", "url": "https://example.invalid/region"})
    return load_table(path)


def reason(verdict):
    return verdict.refusal.reason if verdict.refusal else None


def test_periods_skip_cohorts_and_read_months_and_ranges():
    assert asked_periods("1985년생 여성 중 2020년까지").singles == ("2020",)
    assert asked_periods("2026년 1월 전국 혼인").singles == ("202601",)
    assert asked_periods("2015년부터 2020년까지 추이").ranges == (("2015", "2020"),)
    assert asked_periods("최근 추이").recent and not asked_periods("최근 추이").explicit


def test_plain_questions_are_answered(pop, households):
    assert judge("2023년 남자 가상 인구는?", pop).outcome == "answered"
    assert judge("2023년 서울 1인 가구는 몇 가구야?", households).outcome == "answered"
    assert judge("2023년 부산 중구 1인 가구는?", households).outcome == "answered"  # 부모 지역을 밝힘
    assert judge("2023년 서울과 부산 1인 가구를 더하면?", households).outcome == "answered"  # 가구 수는 더해도 됨


def test_causal_question_is_refused_but_trend_with_cause_is_answered(pop):
    assert reason(judge("가상 인구가 늘어난 이유가 뭐야?", pop)) == "causal"
    assert judge("2022~2023년 가상 인구 추이와 원인을 알려줘", pop).outcome == "answered"


def test_out_of_scope_needs_no_label_and_an_outside_word(pop):
    assert reason(judge("최근 5년 기업 기술개발비는?", pop)) == "out_of_scope"


def test_period_not_in_table_is_refused_with_the_range(pop):
    verdict = judge("2019년 남자 가상 인구는?", pop)
    assert reason(verdict) == "no_period" and verdict.final
    assert "2022~2024" in verdict.refusal.available


def test_shared_label_without_parent_asks_back(households):
    verdict = judge("2023년 중구 1인 가구는?", households)
    assert verdict.outcome == "clarify" and verdict.final
    assert set(verdict.clarify_options) == {"서울특별시 중구", "부산광역시 중구"}


def test_vague_ask_over_measures_with_different_units_asks_back(pop):
    verdict = judge("2023년 가상 통계 알려줘", pop)
    assert verdict.outcome == "clarify" and set(verdict.clarify_options) == {"가상 인구", "가상 성비"}


def test_values_in_different_units_are_not_combined(pop, projection):
    assert reason(judge("2023년 가상 인구와 가상 성비를 더하면?", pop)) == "unit_mismatch"
    assert reason(judge("중위 추계 2050년 총인구와 노년부양비를 더하면?", projection)) == "unit_mismatch"


def test_rates_and_scenarios_are_not_summed(pop, projection):
    assert reason(judge("2023년 가상 성비를 다 더하면?", pop)) == "not_aggregatable"
    assert reason(judge("2050년 고위·중위·저위 추계 총인구를 다 더하면?", projection)) == "not_aggregatable"


def test_label_and_period_without_a_cell_together(new_region):
    assert reason(judge("2009년 세종시 혼인건수는?", new_region)) == "class_mismatch"
    assert judge("2012년 세종시 혼인건수는?", new_region).outcome == "answered"


def test_country_word_in_a_table_without_it(tmp_path):
    rows = [{"period": "2030", "item_id": "T10", "item_name": "순이동률", "unit": None, "dim1_name": "시도별",
             "dim1_code": code, "dim1_label": label, "value_raw": "0.5", "value": 0.5}
            for code, label in (("11", "서울특별시"), ("21", "부산광역시"))]
    path = write_table(tmp_path / "data", "FAKE_SIDO", rows, {"survey_name": "가상 장래추계", "organization": "가상 기관",
                       "table_name": "가상 시도 순이동률", "period": "년 (2030)", "url": "https://example.invalid/sido"})
    table = load_table(path)
    assert reason(judge("2030년 전국 순이동률은?", table)) == "class_mismatch"
    assert judge("2030년 서울 순이동률은?", table).outcome == "answered"


def test_symbol_cells_are_refused_and_blank_is_not_collected(pop, households):
    verdict = judge("2024년 남자 가상 인구는?", pop)
    assert reason(verdict) == "missing_value" and "'x'" in verdict.refusal.message
    assert reason(judge("2024년 계 가상 인구는?", pop)) == "missing_value"  # '-'
    assert reason(judge("2025년 서울 1인 가구는?", households)) == "class_mismatch"  # blank


def test_defaults_name_the_period_and_the_base_scenario(households, projection):
    assert judge("서울 1인 가구는?", households).default_applied == {"period": "latest (2024)"}
    trend = judge("최근 서울 1인 가구 추이는?", households).default_applied
    assert trend == {"period": "recent 4 points (2015~2024)"}  # 2025 is blank: numeric periods only
    assert judge("2050년 총인구는?", projection).default_applied == {"가정별": "중위 추계"}
    assert "period" not in judge("서울 1인 가구가 가장 많았던 해는?", households).default_applied


def test_two_character_lower_level_names_need_a_word_boundary(households):
    index = build_index(households)
    assert {m.code for m in match_labels(index, "서울 중구") if m.label == "중구"} == {"11020", "21010"}
    assert not [m for m in match_labels(index, "한가운데중구") if m.label == "중구"]


# ---------------------------------------------------------------- T0-03 v2: period expressions and op refusals


def test_period_expressions():
    assert asked_periods("2010년대 사망자 수의 연평균은?").ranges == (("2010", "2019"),)
    assert asked_periods("2000년대 초반 3년 동안").ranges == (("2000", "2002"),)
    assert asked_periods("2010년대 후반 5년").ranges == (("2015", "2019"),)
    both = asked_periods("2010년대 전반(2010~2014년)과 후반(2015~2019년)의 평균")
    assert both.ranges == (("2010", "2014"), ("2015", "2019"))  # the decade itself is not a third group
    half = asked_periods("2024년 상반기 혼인 건수는?")
    assert half.ranges == half.spans == (("202401", "202406"),)
    assert asked_periods("2025년 1분기는 2024년 1분기보다?").spans == (("202501", "202503"), ("202401", "202403"))
    assert asked_periods("작년 대비 몇 % 늘었어?").previous and not asked_periods("작년에 태어난").previous
    before = asked_periods("10년 전과 비교해서 2025년 사망자 수는?")
    assert before.singles == ("2015", "2025") and before.before == 10
    assert asked_periods("2025년 전국 혼인 건수는 직전 5년(2020~2024년) 평균보다?").singles == ("2025",)


def test_average_or_sum_over_a_partly_missing_range_is_refused(households):
    verdict = judge("2010년부터 2024년까지 서울 1인 가구의 연평균은?", households)
    assert verdict.rule == "partial_range" and reason(verdict) == "no_period" and "2010년" in verdict.refusal.message
    assert judge("2010년부터 2024년까지 서울 1인 가구 추이는?", households).outcome == "answered"  # a trend only notes the gap
    assert judge("2015년부터 2024년까지 서울 1인 가구의 연평균은?", households).outcome == "answered"
    # "누적" names an item (누적 혼인 비율), it does not ask for a sum: the range is answered with a note
    assert judge("2010년부터 2024년까지 서울 1인 가구 누적 비율은 어떻게 변했어?", households).outcome == "answered"


def test_stock_summed_over_periods_is_refused(households, pop):
    verdict = judge("2015~2024년 서울 1인 가구를 모두 합치면 몇 가구야?", households)
    assert verdict.rule == "stock_over_periods" and reason(verdict) == "not_aggregatable"
    assert judge("2015~2024년 서울 1인 가구를 모두 합치면?", households).outcome == "refused"
    assert judge("2023년 서울 1인 가구와 2인 가구를 합하면?", households).outcome == "answered"  # one period: parts of a whole
    assert judge("2015~2024년 서울 1인 가구 평균은?", households).outcome == "answered"


def test_previous_year_default(households):
    assert judge("작년 대비 서울 1인 가구는 몇 % 늘었어?", households).default_applied == {"period": "latest (2024) vs previous (2023)"}
    assert judge("2024년 서울 1인 가구는 작년보다 늘었어?", households).default_applied == {}  # the period is named


def test_periods_anchored_on_a_calendar_event():
    """S5: a calendar name counts as an event only with a time word after it (직전, 이후, N년 전, 당시...)."""
    before = asked_periods("코로나19 유행 직전 해와 비교하면 2025년 사망자 수는?")
    assert before.singles == ("2019", "2025") and before.events == (("코로나19 유행 직전", "2020-01"),)
    assert asked_periods("IMF 이후 출생아 수 추이").ranges == (("1997", "9999"),)
    assert asked_periods("금융위기 당시 실업률").singles == ("2008",)
    assert asked_periods("코로나 3년 전 사망자 수").singles == ("2017",)
    assert asked_periods("4·19 혁명 직후 인구").singles == ("1961",)
    for plain in ("코로나 환자 수", "세종시 인구밀도는?", "제주특별자치도 인구", "4월 19일 이후 혼인 건수"):
        assert asked_periods(plain).events == (), plain


def test_calendar_has_no_alias_that_reads_as_a_date_or_a_place():
    """Aliases removed on 10/27: bare dates (4·19, 5.18...) read as days of this year, region names and common words
    (세종시, 제주특별자치도, 매미, 루사) read as table labels."""
    names = load_events()
    for alias in ("4·19", "4.19", "5·16", "5.18", "6·25", "10·26", "12·12", "6·29", "세종시", "제주특별자치도", "매미", "루사"):
        assert event_key(alias) not in names, alias
    assert all(not re.fullmatch(r"[\d.·\s]+", name) for name in names)
    assert names[event_key("4·19 혁명")] == "1960-04"
