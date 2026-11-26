"""Loop 11 (analyze/rules.py, analyze/schemas.py):

- C-1: a ranking among targets the question names one by one ranks those targets only (regions, another category,
  or items), for the first place, another place, and both ends.
- C-2: a ranking by the change between two periods among named targets: the picked target's change and its two
  cells, not every target's change (three named) or both changes and their difference (two named).
- C-3: "몇 년" asked of a value measured in years is the value's unit, not a ranked period.
- F-1: an average word beside a per-unit word ("사업체 1개당 평균 종사자 수") describes the per-unit value; it is not
  the average of the two named items.

Every behaviour has neighbour questions that must stay as they were (the whole pool when nothing is named one by
one, each target's change without a ranking word, which of two changed more, a period ranked by a period word, the
average of named targets without a per-unit word). The tables are made up (가상 데이터); every test checks the
answer's values and the cells they were computed from.
"""

from __future__ import annotations

import re

import pytest

from conftest import write_table
from kosis_chat.adapters.parquet import load_table
from kosis_chat.calc import calculate
from kosis_chat.schemas.resources import load_resources

REGIONS = (("00", "전국"), ("11", "서울특별시"), ("21", "부산광역시"), ("22", "대구광역시"), ("23", "인천광역시"),
           ("24", "광주광역시"))
YEARS = ["2019", "2020", "2021", "2022", "2023"]
INDUSTRIES = (("0", "전체"), ("C", "제조업"), ("F", "건설업"), ("K", "금융및보험업"), ("P", "교육서비스업"))


def _row(year, item, name, unit, code, label, v, dim_name="시도별"):
    return {"period": year, "item_id": item, "item_name": name, "unit": unit, "dim1_name": dim_name, "dim1_code": code,
            "dim1_label": label, "value_raw": str(v), "value": v}


# 광주 has the most births and deaths of all; of 서울·부산·대구, 대구 has the most births, 부산 the most deaths.
BIRTHS = {"11": 5100, "21": 4300, "22": 5600, "23": 3900, "24": 7700}
BIRTH_STEP = {"11": 100, "21": -45, "22": 60, "23": 15, "24": 20}  # 서울 grows the most of 서울·부산·대구
DEATHS = {"11": 3300, "21": 6100, "22": 2900, "23": 2500, "24": 8800}
MOVES = {"11": 6800, "21": 2100, "22": 1500, "23": 3100, "24": 900}
MARRIAGES = {"11": 9100, "21": 1200, "22": 800, "23": 700, "24": 400}


def pop_rows():
    """출생아수, 사망자수, 전입자수 (명) and 혼인건수 (건) by region."""
    out = []
    for y, year in enumerate(YEARS):
        vals = {"B": {c: v + BIRTH_STEP[c] * y for c, v in BIRTHS.items()},
                "D": {c: v + 37 * y for c, v in DEATHS.items()},
                "M": {c: v - 11 * y for c, v in MOVES.items()},
                "W": {c: v + 3 * y for c, v in MARRIAGES.items()}}
        for item, (name, unit) in {"B": ("출생아수", "명"), "D": ("사망자수", "명"), "M": ("전입자수", "명"),
                                   "W": ("혼인건수", "건")}.items():
            v = vals[item]
            v["00"] = sum(v.values()) + 1000
            out += [_row(year, item, name, unit, c, l, v[c]) for c, l in REGIONS]
    return out


TENURE = {"0": 7.4, "C": 9.1, "F": 5.2, "K": 12.3, "P": 10.6}


def tenure_rows():
    """One item measured in years (평균근속기간, 년) by industry: 금융및보험업 is the longest of all."""
    return [_row(year, "T", "평균근속기간", "년", c, l, round(TENURE[c] + 0.1 * y * (n % 3), 1), "산업별")
            for y, year in enumerate(YEARS) for n, (c, l) in enumerate(INDUSTRIES)]


def work_rows():
    """사업체수 (개), 종사자수 (명), and a rate whose name holds 당 (인구천명당사업체수, 개) by region."""
    out = []
    for y, year in enumerate(YEARS):
        for r, (code, label) in enumerate(REGIONS):
            firms = 30000 + 111 * y if code == "00" else 9000 - 1300 * r + 37 * y
            workers = 260000 + 1999 * y if code == "00" else 77123 - 9001 * r + 451 * y
            out += [_row(year, "W1", "사업체수", "개", code, label, firms),
                    _row(year, "W2", "종사자수", "명", code, label, workers),
                    _row(year, "W3", "인구천명당사업체수", "개", code, label, round(40 + 1.3 * r + 0.2 * y, 1))]
    return out


def catalog(title):
    return {"survey_name": "가상 조사", "organization": "가상 기관", "table_name": title, "period": "년",
            "url": "https://example.invalid/t"}


SPECS = {
    "FAKE11_POP": (pop_rows, catalog("가상 시도별 인구동태")),
    "FAKE11_TENURE": (tenure_rows, catalog("가상 산업별 평균 근속기간")),
    "FAKE11_WORK": (work_rows, catalog("가상 시도별 사업체와 종사자")),
}
ROWS = {t: make() for t, (make, _) in SPECS.items()}


@pytest.fixture(scope="module")
def res(tmp_path_factory):
    folder = tmp_path_factory.mktemp("loop11") / "data"
    return {t: load_resources(load_table(write_table(folder, t, ROWS[t], cat))) for t, (_, cat) in SPECS.items()}


def ask(res, question, tbl):
    got = calculate(question, [res[tbl]])
    assert got.status == "ok", (got.status, got.reason, got.detail)
    facts = {f.id: f for f in got.facts}
    return facts, [facts[i] for i in got.answer]


def refs_in(facts, fact):
    out = []
    for i in re.findall(r"f\d+", fact.derivation or ""):
        f = facts[i]
        out += refs_in(facts, f) if f.derivation else [f.source_ref]
    return sorted(out)


def value(tbl, item, code, year):
    return next(r["value"] for r in ROWS[tbl] if r["item_id"] == item and r["dim1_code"] == code and r["period"] == year)


def ref(tbl, item, code, year):
    return f"{tbl}|{item}|{code}|{year}"


def cells(answer):
    return sorted(f.source_ref for f in answer if f.source_ref)


def computed(answer):
    return [f for f in answer if not f.source_ref]


# ---------------------------------------------------------------- C-1: candidates named one by one


@pytest.mark.parametrize("question, year, code", [
    ("2023년 서울, 부산, 대구 가운데 출생아 수가 가장 많은 곳은?", "2023", "22"),
    ("서울과 부산, 대구 중에서 2021년 출생아가 제일 많은 지역은 어디야?", "2021", "22"),
    ("2022년 서울, 부산, 대구 중 사망자 수가 가장 많은 곳은?", "2022", "21"),
    ("2020년 서울, 대구, 인천 중 출생아 수가 가장 적은 곳은?", "2020", "23"),
])
def test_named_regions_are_the_candidates(res, question, year, code):
    item = "D" if "사망" in question else "B"
    _, answer = ask(res, question, "FAKE11_POP")
    assert cells(answer) == [ref("FAKE11_POP", item, code, year)]
    assert answer[0].value == value("FAKE11_POP", item, code, year)


def test_named_regions_another_place(res):
    _, answer = ask(res, "2023년 서울, 부산, 대구 중 출생아 수가 두 번째로 많은 곳은?", "FAKE11_POP")
    assert cells(answer) == [ref("FAKE11_POP", "B", "11", "2023")]


@pytest.mark.parametrize("question, year, item", [
    ("2023년 서울의 출생아 수, 사망자 수, 전입자 수 중 가장 많은 것은?", "2023", "M"),
    ("2021년 부산은 출생아 수, 사망자 수, 전입자 수 가운데 가장 적은 것은?", "2021", "M"),
])
def test_named_items_are_the_candidates(res, question, year, item):
    code = "11" if "서울" in question else "21"
    _, answer = ask(res, question, "FAKE11_POP")
    assert cells(answer) == [ref("FAKE11_POP", item, code, year)]


@pytest.mark.parametrize("question, year, code", [
    ("2023년 출생아 수가 가장 많은 시도는?", "2023", "24"),     # nothing named one by one: every province
    ("2022년 사망자 수가 제일 많은 지역은 어디야?", "2022", "24"),
])
def test_neighbour_whole_pool_when_nothing_is_named(res, question, year, code):
    item = "D" if "사망" in question else "B"
    _, answer = ask(res, question, "FAKE11_POP")
    assert cells(answer) == [ref("FAKE11_POP", item, code, year)]


# ---------------------------------------------------------------- C-2: ranked change among named targets


@pytest.mark.parametrize("question, early, late", [
    ("2019년부터 2023년까지 서울, 부산, 대구 중 출생아 수가 가장 많이 늘어난 곳은?", "2019", "2023"),
    ("2020년과 2022년 사이 서울과 부산 가운데 출생아 수가 가장 많이 증가한 지역은 어디고 몇 명 늘었어?", "2020", "2022"),
])
def test_named_targets_ranked_by_change(res, question, early, late):
    facts, answer = ask(res, question, "FAKE11_POP")
    assert cells(answer) == [ref("FAKE11_POP", "B", "11", early), ref("FAKE11_POP", "B", "11", late)]
    (change,) = computed(answer)
    assert change.value == pytest.approx(value("FAKE11_POP", "B", "11", late) - value("FAKE11_POP", "B", "11", early))
    assert refs_in(facts, change) == [ref("FAKE11_POP", "B", "11", early), ref("FAKE11_POP", "B", "11", late)]


def test_named_targets_ranked_by_decline(res):
    facts, answer = ask(res, "2019년부터 2023년까지 서울, 부산, 대구 중 출생아 수가 가장 많이 줄어든 곳은?",
                        "FAKE11_POP")
    assert cells(answer) == [ref("FAKE11_POP", "B", "21", "2019"), ref("FAKE11_POP", "B", "21", "2023")]


def test_neighbour_each_change_without_ranking_word(res):
    facts, answer = ask(res, "2019년부터 2023년까지 서울, 부산, 대구의 출생아 수는 각각 몇 명 늘었어?", "FAKE11_POP")
    assert len(cells(answer)) == 6
    assert len(computed(answer)) == 3


# ---------------------------------------------------------------- C-3: a value measured in years


@pytest.mark.parametrize("question, year, code", [
    ("2023년 제조업, 건설업, 금융및보험업 중 평균 근속기간이 가장 긴 산업은 몇 년이야?", "2023", "K"),
    ("2021년 평균 근속기간이 가장 짧은 산업은 어디고 몇 년이야?", "2021", "F"),
    ("2022년 평균 근속기간이 제일 긴 산업은 몇 년이야?", "2022", "K"),
])
def test_years_as_the_unit_of_the_value(res, question, year, code):
    _, answer = ask(res, question, "FAKE11_TENURE")
    assert cells(answer) == [ref("FAKE11_TENURE", "T", code, year)]


def test_neighbour_period_ranked_for_a_count(res):
    _, answer = ask(res, "서울의 출생아 수가 가장 많았던 해는 몇 년이야?", "FAKE11_POP")
    assert cells(answer) == [ref("FAKE11_POP", "B", "11", "2023")]


# ---------------------------------------------------------------- F-1: per-unit with an average word


@pytest.mark.parametrize("question, year, code", [
    ("2023년 서울의 사업체수 1개당 평균 종사자수는?", "2023", "11"),
    ("2021년 부산은 사업체수 1곳당 종사자수가 평균 몇 명이야?", "2021", "21"),
])
def test_per_unit_with_an_average_word(res, question, year, code):
    facts, answer = ask(res, question, "FAKE11_WORK")
    (per,) = computed(answer)
    workers, firms = value("FAKE11_WORK", "W2", code, year), value("FAKE11_WORK", "W1", code, year)
    assert per.value == pytest.approx(round(workers / firms, 2), abs=0.01)
    assert refs_in(facts, per) == sorted([ref("FAKE11_WORK", "W2", code, year), ref("FAKE11_WORK", "W1", code, year)])


def test_neighbour_per_unit_without_an_average_word(res):
    facts, answer = ask(res, "2022년 대구의 사업체수 1개당 종사자수는?", "FAKE11_WORK")
    (per,) = computed(answer)
    assert per.value == pytest.approx(round(value("FAKE11_WORK", "W2", "22", "2022") / value("FAKE11_WORK", "W1", "22", "2022"), 2),
                                      abs=0.01)


@pytest.mark.parametrize("question, item", [
    ("2023년 서울, 부산, 대구의 사업체 수 평균은?", "W1"),
    ("2023년 서울, 부산, 대구의 인구 천명당 사업체수 평균은 얼마야?", "W3"),  # 당 inside a label
])
def test_neighbour_average_of_named_targets(res, question, item):
    facts, answer = ask(res, question, "FAKE11_WORK")
    (avg,) = computed(answer)
    want = [value("FAKE11_WORK", item, c, "2023") for c in ("11", "21", "22")]
    assert avg.value == pytest.approx(sum(want) / 3, abs=0.01)
    assert refs_in(facts, avg) == sorted(ref("FAKE11_WORK", item, c, "2023") for c in ("11", "21", "22"))


def test_neighbour_which_of_two_changed_more(res):
    facts, answer = ask(res, "2020년과 2023년 사이 서울과 부산 중 어디가 출생아 수가 더 많이 늘었어?", "FAKE11_POP")
    assert cells(answer) == sorted(ref("FAKE11_POP", "B", c, y) for c in ("11", "21") for y in ("2020", "2023"))
    assert len(computed(answer)) == 3  # both changes and their difference, as before
