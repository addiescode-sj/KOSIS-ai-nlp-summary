"""Rule question analysis (T1-06): question -> Plan -> blocks (build.py) -> StatQuery, on the made-up household table."""

from dataclasses import replace

import pytest

from conftest import write_table
from kosis_chat.compute import compute
from kosis_chat.compute.analyze import PERIOD_AXIS, Unresolved, build_plan, rule_plan, to_query
from kosis_chat.data.loader import load_table
from kosis_chat.data.preprocessing import clean
from kosis_chat.judge.answerability import judge
from kosis_chat.judge.question import CHANGE_WORDS, asked_periods, build_index

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
    return plan, build_plan(plan, index, table)


def cells(built, *steps):
    """The source_refs the plan's select steps read, in step order (all selects when no step is named)."""
    picked = [s for s in built.plan.steps if s.op == "select" and (not steps or s.id in steps)]
    return [c for s in picked for c in s.cells]


def test_value_names_one_cell(table, index):
    plan, built = analyze("2023년 서울특별시 1인 가구 수는?", table, index)
    assert plan.type == "value" and built.pattern == "value" and cells(built) == [ref("T1", "11", "2023")]
    query = to_query(plan, index, built)
    assert query.item == "1인 (가구)" and query.filters == {REGION: "서울특별시"} and query.periods == ["2023"]
    assert query.plan == built.plan


def test_default_latest_value_shows_newer_blank_period(table, index):
    plan, built = analyze("서울 1인 가구 수는?", table, index)
    assert plan.default_applied["period"].startswith("latest (2024")
    assert cells(built) == [ref("T1", "11", "2024")] and built.context == [ref("T1", "11", "2025")]


def test_trend_takes_every_period_in_range_without_gap_note(table, index):
    plan, built = analyze("2015년부터 2023년까지 부산 1인 가구는 어떻게 변했어?", table, index)
    assert plan.type == "trend" and cells(built) == [ref("T1", "21", p) for p in ("2015", "2020", "2023")]
    assert built.plan.note is None  # 2016~2019 are not in a census table by design


def test_trend_notes_a_missing_end_year(table, index):
    _, built = analyze("2015년부터 2022년까지 부산 1인 가구 추이", table, index)
    assert built.plan.note == "2022년은 표에 없음"


def test_compare_keeps_named_reference(table, index):
    plan, built = analyze("2023년 서울이 전국보다 1인 가구가 얼마나 많아?", table, index)
    assert cells(built) == [ref("T1", "11", "2023"), ref("T1", "00", "2023")] and built.reference == 1
    query = to_query(plan, index, built)
    assert query.targets == [{REGION: "서울특별시"}, {REGION: "전국"}] and query.reference == 1


def test_share_uses_item_total(table, index):
    _, built = analyze("2023년 서울의 1인 가구 비중은?", table, index)
    assert cells(built, "p") == [ref("T1", "11", "2023")] and cells(built, "t") == [ref("T0", "11", "2023")]


def test_extreme_over_provinces_with_country_as_context(table, index):
    plan, built = analyze("2023년 1인 가구가 가장 많은 시도는?", table, index)
    assert plan.over == REGION and plan.pick == "max"
    assert built.plan.step("k").candidates == [ref("T1", r, "2023") for r in ("11", "21", "39")]  # not 동부 or 시군구
    assert built.context == [ref("T1", "00", "2023")]
    winner = compute(table, built).picked["winner"]
    assert winner == "f1" and compute(table, built).result.facts[0].source_ref == ref("T1", "39", "2023")


def test_extreme_over_periods(table, index):
    plan, built = analyze("서울 1인 가구가 가장 적었던 해는?", table, index)
    assert plan.over == PERIOD_AXIS and plan.pick == "min"
    assert ref("T1", "11", "2015") in built.plan.step("k").candidates


def test_most_recent_is_a_period_not_a_ranking(table, index):
    plan, built = analyze("가장 최근 서울특별시 1인 가구 수는?", table, index)
    assert plan.type == "value" and cells(built) == [ref("T1", "11", "2024")]


def test_extreme_without_ranked_axis_is_unresolved_not_a_crash(table, index):
    plan, _ = analyze("2023년 1인 가구가 가장 많은 시도는?", table, index)
    with pytest.raises(Unresolved) as exc:
        build_plan(replace(plan, over=None), index, table)  # what a model fill with over=null leaves
    assert exc.value.what == ["over"]


def test_change_without_period_is_a_recent_trend(table, index):
    plan, built = analyze("서울특별시 1인 가구는 얼마나 늘었나?", table, index)
    assert plan.type == "trend" and plan.default_applied["period"].startswith("recent 4 points")
    assert cells(built) == [ref("T1", "11", p) for p in ("2015", "2020", "2023", "2024")]


def test_change_with_one_year_ends_at_that_year(table, index):
    plan, built = analyze("2023년 서울특별시 1인 가구는 얼마나 늘었나?", table, index)
    assert plan.type == "trend" and cells(built) == [ref("T1", "11", p) for p in ("2015", "2020", "2023")]


def test_change_words_skip_labels():
    assert CHANGE_WORDS.search("얼마나 늘었어") and CHANGE_WORDS.search("감소했나요")
    assert not CHANGE_WORDS.search("2024년 부산의 자연증가건수는 몇 명이야?")


def test_item_total_fills_an_unnamed_item(table, index):
    _, built = analyze("2023년 서울 가구는 몇이야?", table, index)
    assert cells(built) == [ref("T0", "11", "2023")]  # 일반가구_계


def test_unnamed_item_without_total_is_unresolved(pop_path):
    pop = clean(load_table(pop_path))
    plan = rule_plan("2023년 남자는?", build_index(pop))
    with pytest.raises(Unresolved) as exc:
        build_plan(plan, build_index(pop), pop)
    assert exc.value.what == ["항목"]


# ---------------------------------------------------------------- patterns over periods (T0-03 v2, blocks since T0-04)


def test_average_over_a_range(table, index):
    plan, built = analyze("2015년부터 2024년까지 서울특별시 1인 가구의 연평균은?", table, index)
    assert plan.type == "trend" and plan.pattern == "avg" and plan.groups == [["2015", "2020", "2023", "2024"]]
    assert built.pattern == "avg" and cells(built) == [ref("T1", "11", p) for p in ("2015", "2020", "2023", "2024")]
    assert [(s.op, getattr(s, "fn", None)) for s in built.plan.steps] == [("select", None), ("aggregate", "avg")]
    query = to_query(plan, index, built)
    assert query.plan == built.plan and query.periods == plan.groups[0]


def test_two_ranges_compared_and_value_against_average(table, index):
    plan, built = analyze("서울특별시 1인 가구는 2015~2020년 평균보다 2023~2024년 평균이 얼마나 많아?", table, index)
    assert plan.pattern == "pavg" and [cells(built, t.cells) for t in built.terms] == [[ref("T1", "11", "2015"), ref("T1", "11", "2020")], [ref("T1", "11", "2023"), ref("T1", "11", "2024")]]
    plan, built = analyze("2024년 서울특별시 1인 가구는 2015~2023년 평균보다 얼마나 많아?", table, index)
    assert plan.pattern == "vsa" and built.terms[0].agg == "avg" and cells(built, built.terms[1].cells) == [ref("T1", "11", "2024")]
    assert to_query(plan, index, built).periods == ["2015", "2020", "2023", "2024"]


def test_share_at_two_periods(table, index):
    plan, built = analyze("전국 일반가구 중 1인 가구 비중은 2015년보다 2024년에 몇 %p 늘었어?", table, index)
    assert plan.type == "share" and plan.pattern == "shr" and plan.groups == [["2015"], ["2024"]]
    assert [(cells(built, t.cells), cells(built, t.total)) for t in built.terms] == [([ref("T1", "00", "2015")], [ref("T0", "00", "2015")]), ([ref("T1", "00", "2024")], [ref("T0", "00", "2024")])]


def test_plain_words_do_not_start_an_op(table, index):
    plan, built = analyze("2023년 서울특별시 1인 가구는 전국 평균보다 많아?", table, index)
    assert plan.pattern is None and built.pattern == "compare"  # "평균" names the reference, there is no period range
    plan, built = analyze("2015년부터 2024년까지 서울특별시 1인 가구 추이는?", table, index)
    assert plan.pattern is None and built.pattern == "trend" and len(cells(built)) == 4


def test_previous_year_compares_the_last_two_periods(table, index):
    plan, built = analyze("작년 대비 서울특별시 1인 가구는 몇 % 늘었어?", table, index)
    assert plan.default_applied == {"period": "latest (2024) vs previous (2023)"}  # 2025 is blank
    assert built.pattern == "trend" and cells(built) == [ref("T1", "11", "2023"), ref("T1", "11", "2024")]


# ---------------------------------------------------------------- S5: shapes the v2 requests could not write


def test_gap_between_two_targets_at_two_periods(table, index):
    plan, built = analyze("서울특별시와 부산광역시의 1인 가구 격차는 2015년보다 2024년에 벌어졌어?", table, index)
    assert plan.type == "compare" and plan.pattern == "gap" and plan.groups == [["2015"], ["2024"]]
    assert built.pattern == "gap" and cells(built) == [ref("T1", r, p) for p in ("2015", "2024") for r in ("11", "21")]
    assert [s.fn for s in built.plan.steps if s.op == "combine"] == ["diff", "diff", "diff"]


def test_growth_of_two_targets_compared(table, index):
    plan, built = analyze("2015년 대비 2024년 서울특별시와 부산광역시 중 1인 가구 증가율이 더 큰 곳은?", table, index)
    assert plan.pattern == "grw" and built.pattern == "grw"
    assert cells(built) == [ref("T1", "11", "2015"), ref("T1", "11", "2024"), ref("T1", "21", "2015"), ref("T1", "21", "2024")]
    last = built.plan.steps[-1]
    assert (last.fn, last.orient) == ("pp_diff", "larger_first")
    _, built = analyze("2015년 대비 2024년 서울특별시와 부산광역시 중 1인 가구 감소율이 더 큰 곳은?", table, index)
    assert built.plan.steps[-1].orient == "smaller_first"


def test_two_targets_without_gap_or_growth_words_stay_a_trend(table, index):
    question = "2015년과 2024년 서울특별시와 부산광역시 1인 가구는?"
    plan = rule_plan(question, index, judge(question, table, index).default_applied)
    assert plan.pattern is None


def test_share_of_a_sub_region_uses_its_parent(table, index):
    """Before S5 the denominator was the category total (전국); now it is the part's nearest parent (07 plan 3-3)."""
    _, built = analyze("2023년 서울특별시 일반가구 중 종로구 가구 비중은?", table, index)
    assert cells(built, "p") == [ref("T0", "11010", "2023")] and cells(built, "t") == [ref("T0", "11", "2023")]
    _, built = analyze("2023년 일반가구 중 종로구 가구 비중은?", table, index)
    assert cells(built, "t") == [ref("T0", "11", "2023")]  # the parent even when the question does not name it
    _, built = analyze("2023년 일반가구 중 서울특별시 가구 비중은?", table, index)
    assert cells(built, "t") == [ref("T0", "00", "2023")]  # a province has no parent: the total


# ---------------------------------------------------------------- general rules from the S5b review (10/28)
# Questions worded unlike the golden set, on made-up tables, so the rules are not fitted to the golden questions.

BIRTH_REGIONS = (("00", "전국"), ("11", "서울특별시"), ("21", "부산광역시"))


@pytest.fixture
def births(tmp_path):
    """A monthly table that also holds each year's value (the sum of its months), counted in '출생건수'."""
    rows = []
    for r_no, (code, label) in enumerate(BIRTH_REGIONS):
        for year in (2023, 2024):
            months = [100 * (3 - r_no) + m + (year - 2023) * 50 for m in range(1, 13)]
            values = {f"{year}{m:02d}": v for m, v in enumerate(months, 1)} | {str(year): sum(months)}
            rows += [{"period": p, "prd_se": "M", "item_id": "T10", "item_name": "출생건수 (명)", "unit": "명",
                      "dim1_name": "행정구역별", "dim1_code": code, "dim1_label": label, "value_raw": str(v), "value": float(v)}
                     for p, v in values.items()]
    catalog = {"survey_name": "가상 동향", "organization": "가상 기관", "table_name": "가상 월별 출생", "period": "월",
               "url": "https://example.invalid/births"}
    table = clean(load_table(write_table(tmp_path / "data", "FAKE_BIRTH", rows, catalog)))
    return table, build_index(table)


def birth(region, period):
    return f"FAKE_BIRTH|T10|{region}|{period}"


def test_a_count_is_found_by_the_counted_person(births):
    table, index = births
    plan, built = analyze("2024년 부산 출생아 수는?", table, index)
    assert plan.codes["항목"] == ["T10"] and cells(built) == [birth("21", "2024")]
    plan, _ = analyze("2023년 전국 출생아 중 서울특별시 출생아 비중은?", table, index)
    assert plan.codes["항목"] == ["T10"]


def test_a_year_on_a_monthly_table_is_its_yearly_value_unless_months_are_asked(births):
    table, index = births
    _, built = analyze("2023년 서울과 부산의 출생아 수 차이는?", table, index)
    assert cells(built) == [birth("11", "2023"), birth("21", "2023")]
    plan, _ = analyze("2024년 부산 출생아 수 월별 추이는?", table, index)
    assert plan.periods == ["2024"] + [f"2024{m:02d}" for m in range(1, 13)]


def test_a_month_without_a_year_takes_the_year_before_it(births):
    table, index = births
    plan, built = analyze("부산 출생아 수는 2024년 3월 대비 9월에 몇 퍼센트 늘었어?", table, index)
    assert plan.type == "trend" and cells(built) == [birth("21", "202403"), birth("21", "202409")]
    assert asked_periods("2024년 1~6월 혼인").singles == ("2024",)  # the end of a month range is not a month
    assert asked_periods("3월 출생아 비중").singles == ()  # no year before it: not guessed


def test_months_as_a_part_of_their_year(births):
    table, index = births
    plan, built = analyze("2023년 서울 출생아 중 7월 출생아가 차지하는 비중은?", table, index)
    assert plan.type == "share" and plan.pattern is None
    assert cells(built, "p") == [birth("11", "202307")] and cells(built, "t") == [birth("11", "2023")]
    share = compute(table, built).result.facts[-1]
    assert share.unit == "%" and share.value == pytest.approx(round(207 / sum(200 + m for m in range(1, 13)) * 100, 1))


def test_a_label_before_eui_myeot_percent_asks_a_share(births):
    table, index = births
    plan, built = analyze("2024년 부산 출생아 수는 전국의 몇 퍼센트야?", table, index)
    assert plan.type == "share" and cells(built, "p") == [birth("21", "2024")] and cells(built, "t") == [birth("00", "2024")]


@pytest.fixture
def structure(tmp_path):
    """A yearly table whose categories hold counts and their shares side by side (장래추계 주요 지표 layout)."""
    labels = (("23", "인구(명): 65세 이상", {"2030": 1000, "2040": 1500}), ("33", "- 구성비(%): 65세 이상", {"2030": 20.1, "2040": 30.6}))
    rows = [{"period": p, "item_id": "T1", "item_name": "가상 지표", "unit": None, "dim1_name": "인구구조별", "dim1_code": code,
             "dim1_label": label, "value_raw": str(v), "value": float(v)} for code, label, values in labels for p, v in values.items()]
    catalog = {"survey_name": "가상 추계", "organization": "가상 기관", "table_name": "가상 인구구조", "period": "년",
               "url": "https://example.invalid/structure"}
    table = clean(load_table(write_table(tmp_path / "data", "FAKE_STRUCT", rows, catalog)))
    return table, build_index(table)


def test_a_share_word_inside_a_named_label_is_not_a_share_question(structure):
    table, index = structure
    plan, built = analyze("65세 이상 인구 구성비는 2030년보다 2040년에 몇 %p 높아져?", table, index)
    assert plan.type == "trend" and plan.codes["인구구조별"] == ["33"]  # %p keeps the code measured in %
    diff = next(f for f in compute(table, built).result.facts if f.derivation and "-" in f.derivation)
    assert (diff.value, diff.unit) == (pytest.approx(10.5), "%p")
