"""Loop 4, one table (analyze/rules.py, shared/patterns.py, judge/rules.py):

- A-1: an ordinal or a rank word inside a label the question names ('첫째아', '둘째', '1순위') is the label, not a
  ranking: the ranking words are read on the question without its labels (Parsed.rest);
- B-1: a range of years asked period by period (연도별, 해마다, 매년, "N년 간격으로", "N년마다") lists the cells of the
  periods inside, every one (or every N-th from the first), with no change computed;
- C-1: the largest and the smallest of a ranking asked together for their values, no formula: the two cells only;
- D-3: the rest of a category but one ("A를 제외한", "A 말고", "A 외"): the category's total minus that one.

The tables are made up (가상 데이터). Every test checks the answer's values and the cells they were computed from.
"""

from __future__ import annotations

import pytest

from conftest import write_table
from kosis_chat.adapters.parquet import load_table
from kosis_chat.calc import calculate
from kosis_chat.schemas.resources import load_resources

CATALOG = {"survey_name": "가상 조사", "organization": "가상 기관", "table_name": "가상 표", "period": "년",
           "url": "https://example.invalid/t"}
YEARS = tuple(str(y) for y in range(2014, 2023))
REGIONS = {"00": "전국", "11": "서울특별시", "21": "부산광역시", "22": "대구광역시", "23": "인천광역시"}
# FAKE_ORD: items whose names are ordinals (birth order)
ORD_ITEMS = {"T1": "첫째아", "T2": "둘째아", "T3": "셋째아 이상"}
# FAKE_SEQ: a category whose labels are ordinals, and one item
SEQ_LABELS = {"0": "계", "1": "첫째", "2": "둘째", "3": "셋째 이상"}
# FAKE_JOB: a category whose labels hold 순위
JOB_LABELS = {"0": "전체", "1": "1순위 응답", "2": "2순위 응답"}


def ord_value(item, code, year):
    n = int(year) - 2014
    base = {"11": 4000, "21": 1500, "22": 1100, "23": 1300}
    if code == "00":
        return sum(ord_value(item, c, year) for c in base)
    return base[code] // int(item[1]) - 13 * n * int(item[1]) + int(code)


def seq_value(code, year):
    n = int(year) - 2014
    if code == "0":
        return sum(seq_value(c, year) for c in ("1", "2", "3"))
    return {"1": 9000, "2": 6000, "3": 1500}[code] - 70 * n * int(code)


def job_value(code, year):
    n = int(year) - 2014
    return {"0": 1000, "1": 420, "2": 310}[code] + 5 * n


def ord_rows():
    out = []
    for period in YEARS:
        for item, name in ORD_ITEMS.items():
            for code, label in REGIONS.items():
                v = ord_value(item, code, period)
                out.append({"period": period, "item_id": item, "item_name": name, "unit": "명",
                            "dim1_name": "행정구역별", "dim1_code": code, "dim1_label": label,
                            "value_raw": str(v), "value": v})
    return out


def seq_rows():
    out = []
    for period in YEARS:
        for code, label in SEQ_LABELS.items():
            v = seq_value(code, period)
            out.append({"period": period, "item_id": "T1", "item_name": "출생아수", "unit": "명",
                        "dim1_name": "출산순위별", "dim1_code": code, "dim1_label": label,
                        "value_raw": str(v), "value": v})
    return out


def job_rows():
    out = []
    for period in YEARS:
        for code, label in JOB_LABELS.items():
            v = job_value(code, period)
            out.append({"period": period, "item_id": "T1", "item_name": "응답자수", "unit": "명",
                        "dim1_name": "응답순위별", "dim1_code": code, "dim1_label": label,
                        "value_raw": str(v), "value": v})
    return out


@pytest.fixture(scope="module")
def res(tmp_path_factory):
    folder = tmp_path_factory.mktemp("loop4_one") / "data"
    return {name: load_resources(load_table(write_table(folder, name, rows(), CATALOG)))
            for name, rows in (("FAKE_ORD", ord_rows), ("FAKE_SEQ", seq_rows), ("FAKE_JOB", job_rows))}


def answer(res, question, table):
    got = calculate(question, [res[table]])
    assert got.status == "ok", (got.status, got.reason, got.detail)
    facts = {f.id: f for f in got.facts}
    return [facts[i] for i in got.answer]


def cells_of(facts):
    return [f.source_ref for f in facts if f.source_ref]


def derived_of(facts):
    return [f.value for f in facts if not f.source_ref]


def ord_ref(item, code, year):
    return f"FAKE_ORD|{item}|{code}|{year}"


def seq_ref(code, year):
    return f"FAKE_SEQ|T1|{code}|{year}"


def job_ref(code, year):
    return f"FAKE_JOB|T1|{code}|{year}"


# ---------------------------------------------------------------- A-1: ordinals inside the labels


@pytest.mark.parametrize("question, table, refs", [
    ("2022년 부산의 둘째아 수 알려줘", "FAKE_ORD", [ord_ref("T2", "21", "2022")]),
    ("2018년 인천 첫째아는 몇 명이야?", "FAKE_ORD", [ord_ref("T1", "23", "2018")]),
    ("2021년 서울 첫째아와 둘째아는 각각 몇 명이야?", "FAKE_ORD",
     [ord_ref("T1", "11", "2021"), ord_ref("T2", "11", "2021")]),
    ("2020년 둘째 출생아 수는?", "FAKE_SEQ", [seq_ref("2", "2020")]),
    ("2016년 셋째 이상 출생아 수 알려줘", "FAKE_SEQ", [seq_ref("3", "2016")]),
    ("2019년 1순위 응답 응답자수는?", "FAKE_JOB", [job_ref("1", "2019")]),
])
def test_ordinal_in_a_label_is_no_ranking(res, question, table, refs):
    facts = answer(res, question, table)
    assert sorted(cells_of(facts)) == sorted(refs)
    assert derived_of(facts) == []


@pytest.mark.parametrize("question, table, early, late, change", [
    ("2017년부터 2022년까지 대구 셋째아 이상은 얼마나 줄었어?", "FAKE_ORD",
     ord_ref("T3", "22", "2017"), ord_ref("T3", "22", "2022"), ord_value("T3", "22", "2022") - ord_value("T3", "22", "2017")),
    ("2015년과 2020년 사이 둘째 출생아 수는 얼마나 변했어?", "FAKE_SEQ",
     seq_ref("2", "2015"), seq_ref("2", "2020"), seq_value("2", "2020") - seq_value("2", "2015")),
])
def test_ordinal_label_change_is_a_trend(res, question, table, early, late, change):
    facts = answer(res, question, table)
    assert {early, late} <= set(cells_of(facts))
    assert pytest.approx(change, abs=0.005) in derived_of(facts)


def test_a_ranking_on_an_ordinal_label_still_ranks(res):
    # 2021 둘째아: 서울 2000-..., the most among the provinces
    facts = answer(res, "2021년 둘째아가 가장 많은 시도는?", "FAKE_ORD")
    assert cells_of(facts)[:1] == [ord_ref("T2", "11", "2021")]
    facts = answer(res, "2021년 첫째아 수가 두 번째로 많은 시도는?", "FAKE_ORD")  # 서울, then 부산
    assert cells_of(facts) == [ord_ref("T1", "21", "2021")]


# ---------------------------------------------------------------- B-1: a range asked period by period


@pytest.mark.parametrize("question, refs", [
    ("2016년부터 2020년까지 서울 첫째아 수를 연도별로 알려줘",
     [ord_ref("T1", "11", y) for y in ("2016", "2017", "2018", "2019", "2020")]),
    ("2015~2019년 해마다 부산 둘째아는 몇 명이었어?",
     [ord_ref("T2", "21", y) for y in ("2015", "2016", "2017", "2018", "2019")]),
    ("2014~2022년 둘째 출생아 수를 2년 간격으로 보여줘",
     [seq_ref("2", y) for y in ("2014", "2016", "2018", "2020", "2022")]),
    ("2014년부터 2022년까지 4년마다 1순위 응답 응답자수는?",
     [job_ref("1", y) for y in ("2014", "2018", "2022")]),
    ("2015~2021년 셋째 이상 출생아 수를 격년으로 알려줘",
     [seq_ref("3", y) for y in ("2015", "2017", "2019", "2021")]),
])
def test_range_period_by_period_lists_the_cells(res, question, refs):
    table = refs[0].split("|")[0]
    facts = answer(res, question, table)
    assert sorted(cells_of(facts)) == sorted(refs)
    assert derived_of(facts) == []


def test_range_trend_stays_a_trend(res):
    facts = answer(res, "2016년부터 2020년까지 서울 첫째아 수는 얼마나 줄었어?", "FAKE_ORD")
    assert pytest.approx(ord_value("T1", "11", "2020") - ord_value("T1", "11", "2016")) in derived_of(facts)


# ---------------------------------------------------------------- C-1: both ends of a ranking, values only


@pytest.mark.parametrize("question, refs", [
    # 2021 둘째아: 서울 the most, 대구 the fewest of the provinces
    ("2021년 둘째아가 가장 많은 시도와 가장 적은 시도는?", [ord_ref("T2", "11", "2021"), ord_ref("T2", "22", "2021")]),
    # 2018 첫째아: 서울 and 대구 again, asked with 제일 and 높/낮 words of another form
    ("2018년 첫째아 수가 제일 많은 곳과 제일 적은 곳의 값을 알려줘",
     [ord_ref("T1", "11", "2018"), ord_ref("T1", "22", "2018")]),
    # periods ranked: a falling series, the most at the first year, the fewest at the last
    ("2014~2022년 첫째 출생아 수의 최댓값과 최솟값은?", [seq_ref("1", "2014"), seq_ref("1", "2022")]),
    ("부산 셋째아 이상이 가장 많았던 해와 가장 적었던 해는 언제야?",
     [ord_ref("T3", "21", "2014"), ord_ref("T3", "21", "2022")]),
])
def test_largest_and_smallest_values_only(res, question, refs):
    table = refs[0].split("|")[0]
    facts = answer(res, question, table)
    assert sorted(cells_of(facts)) == sorted(refs)
    assert derived_of(facts) == []


def test_largest_and_smallest_with_a_difference_stays(res):
    facts = answer(res, "2021년 둘째아가 가장 많은 시도와 가장 적은 시도의 차이는?", "FAKE_ORD")
    assert sorted(cells_of(facts)) == sorted([ord_ref("T2", "11", "2021"), ord_ref("T2", "22", "2021")])
    assert derived_of(facts) == [pytest.approx(ord_value("T2", "11", "2021") - ord_value("T2", "22", "2021"))]


# ---------------------------------------------------------------- D-3: the rest of a category but one


@pytest.mark.parametrize("question, whole, part, value", [
    ("2022년 전국 둘째아 중 서울을 제외한 나머지는 몇 명이야?", ord_ref("T2", "00", "2022"), ord_ref("T2", "11", "2022"),
     ord_value("T2", "00", "2022") - ord_value("T2", "11", "2022")),
    ("2019년 부산 말고 나머지 지역의 첫째아 수는?", ord_ref("T1", "00", "2019"), ord_ref("T1", "21", "2019"),
     ord_value("T1", "00", "2019") - ord_value("T1", "21", "2019")),
    ("2017년 인천 외 지역의 셋째아 이상 수 알려줘", ord_ref("T3", "00", "2017"), ord_ref("T3", "23", "2017"),
     ord_value("T3", "00", "2017") - ord_value("T3", "23", "2017")),
    ("2020년 셋째 이상을 제외한 출생아 수는?", seq_ref("0", "2020"), seq_ref("3", "2020"),
     seq_value("0", "2020") - seq_value("3", "2020")),
    ("2016년 첫째 빼고 출생아 수는 얼마야?", seq_ref("0", "2016"), seq_ref("1", "2016"),
     seq_value("0", "2016") - seq_value("1", "2016")),
])
def test_total_minus_the_part_left_out(res, question, whole, part, value):
    table = whole.split("|")[0]
    facts = answer(res, question, table)
    assert sorted(cells_of(facts)) == sorted([whole, part])
    assert derived_of(facts) == [pytest.approx(value, abs=0.005)]


# ---------------------------------------------------------------- B-2: the average change per year (an amount)


@pytest.mark.parametrize("question, early, late, value", [
    ("2016년부터 2021년까지 서울 첫째아는 한 해 평균 몇 명씩 줄었어?", ord_ref("T1", "11", "2016"), ord_ref("T1", "11", "2021"),
     (ord_value("T1", "11", "2021") - ord_value("T1", "11", "2016")) / 5),
    ("2014년과 2022년 사이 부산 둘째아 수는 연평균 얼마나 변했어?", ord_ref("T2", "21", "2014"), ord_ref("T2", "21", "2022"),
     (ord_value("T2", "21", "2022") - ord_value("T2", "21", "2014")) / 8),
    ("최근 4년 둘째 출생아 수는 연평균 몇 명씩 감소했어?", seq_ref("2", "2019"), seq_ref("2", "2022"),
     (seq_value("2", "2022") - seq_value("2", "2019")) / 3),
    ("2015~2020년 1순위 응답 응답자수는 해마다 평균 몇 명씩 늘었어?", job_ref("1", "2015"), job_ref("1", "2020"),
     (job_value("1", "2020") - job_value("1", "2015")) / 5),
])
def test_average_yearly_change_from_the_two_ends(res, question, early, late, value):
    table = early.split("|")[0]
    facts = answer(res, question, table)
    assert sorted(cells_of(facts)) == sorted([early, late])
    assert derived_of(facts) == [pytest.approx(value, abs=0.005)]


def test_an_average_of_values_stays_a_mean(res):
    facts = answer(res, "2015~2017년 서울 첫째아 평균은?", "FAKE_ORD")
    mean = sum(ord_value("T1", "11", y) for y in ("2015", "2016", "2017")) / 3
    assert derived_of(facts) == [pytest.approx(mean, abs=0.005)]
