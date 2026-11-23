"""Loop 5 (judge/question.py, shared/patterns.py, analyze/rules.py, compute/answer_facts.py):

- B: one named period and a word for the period right before it (전년, 작년, 지난해, 앞 해, 전월, 지난달, 앞 달,
  전년 동월, 작년 같은 달, "N개월 전"): the earlier period is put in the asked periods, as "N년 전" already was. A named
  month is compared with a month (the month before, across a year end too, or the same month a year before), never
  with the year's value. Asked for an amount ("몇 명", "몇 건"), the answer is the change and its two cells; asked
  for a rate (증가율, "몇 퍼센트"), the rate of change and its two cells.
- C-1: three or more targets of one category named one by one, no formula asked: their cells, listed.

The tables are made up (가상 데이터). Every test checks the answer's values and the cells they were computed from.
"""

from __future__ import annotations

import re

import pytest

from conftest import write_table
from kosis_chat.adapters.parquet import load_table
from kosis_chat.calc import calculate
from kosis_chat.schemas.resources import load_resources

CATALOG = {"survey_name": "가상 조사", "organization": "가상 기관", "table_name": "가상 표", "period": "년",
           "url": "https://example.invalid/t"}
REGIONS = (("00", "전국"), ("11", "서울특별시"), ("21", "부산광역시"), ("22", "대구광역시"), ("23", "인천광역시"),
           ("24", "광주광역시"))
YEARS = tuple(str(y) for y in range(2016, 2024))
AGES = (("0", "계"), ("1", "20대"), ("2", "30대"), ("3", "40대"), ("4", "50대"))


def year_value(base: int, r_no: int, y_no: int) -> int:
    return base * (6 - r_no) + 37 * y_no * (r_no + 1) - 11 * y_no * y_no


def year_rows(item, base):
    item_id, item_name, unit = item
    out = []
    for y_no, year in enumerate(YEARS):
        for r_no, (code, label) in enumerate(REGIONS):
            v = year_value(base, r_no, y_no)
            out.append({"period": year, "item_id": item_id, "item_name": item_name, "unit": unit,
                        "dim1_name": "행정구역별", "dim1_code": code, "dim1_label": label,
                        "value_raw": str(v), "value": v, "prd_se": "Y"})
    return out


def month_rows(item, base, years=("2022", "2023", "2024")):
    """A monthly table that also holds each year's value (the sum of its months)."""
    item_id, item_name, unit = item
    out = []
    for y_no, year in enumerate(years):
        for r_no, (code, label) in enumerate(REGIONS):
            months = [base * (6 - r_no) + 5 * m * (r_no + 1) - 7 * y_no + (m % 3) * 13 for m in range(1, 13)]
            values = {f"{year}{m:02d}": v for m, v in enumerate(months, 1)} | {year: sum(months)}
            out += [{"period": p, "item_id": item_id, "item_name": item_name, "unit": unit,
                     "dim1_name": "행정구역별", "dim1_code": code, "dim1_label": label,
                     "value_raw": str(v), "value": v, "prd_se": "M" if len(p) == 6 else "Y"}
                    for p, v in values.items()]
    return out


def age_rows():
    out = []
    for y_no, year in enumerate(YEARS):
        parts = {code: 3000 + 450 * a_no - 60 * y_no * a_no for a_no, (code, _) in enumerate(AGES) if code != "0"}
        values = {"0": sum(parts.values()), **parts}
        for code, label in AGES:
            out.append({"period": year, "item_id": "T1", "item_name": "취업자수", "unit": "명",
                        "dim1_name": "연령별", "dim1_code": code, "dim1_label": label,
                        "value_raw": str(values[code]), "value": values[code], "prd_se": "Y"})
    return out


SPECS = {
    "FAKE5_BIRTH": lambda: year_rows(("T1", "출생아수", "명"), 900),
    "FAKE5_DEATH": lambda: year_rows(("T1", "사망자수", "명"), 1100),
    "FAKE5_MARRY_YM": lambda: month_rows(("T1", "혼인건수", "건"), 40),
    "FAKE5_MOVE_YM": lambda: month_rows(("T1", "전입자수", "명"), 300),
    "FAKE5_AGE": age_rows,
}


@pytest.fixture(scope="module")
def res(tmp_path_factory):
    folder = tmp_path_factory.mktemp("loop5") / "data"
    return {t: load_resources(load_table(write_table(folder, t, make(), CATALOG))) for t, make in SPECS.items()}


def ask(res, question, *tbl_ids):
    got = calculate(question, [res[t] for t in tbl_ids])
    assert got.status == "ok", (got.status, got.reason, got.detail)
    facts = {f.id: f for f in got.facts}
    return facts, [facts[i] for i in got.answer]


def refs_in(facts, fact):
    """The cells a derived fact was computed from, following derived inputs down to the cells."""
    out = []
    for i in re.findall(r"f\d+", fact.derivation or ""):
        f = facts[i]
        out += refs_in(facts, f) if f.derivation else [f.source_ref]
    return sorted(out)


def ref(tbl, region, period):
    return f"{tbl}|T1|{region}|{period}"


def cell_value(facts, cell_ref):
    return next(f.value for f in facts.values() if f.source_ref == cell_ref)


# ---------------------------------------------------------------- B: the period right before a named one


@pytest.mark.parametrize("question, table, region, early, late, rate", [
    # a year and the year before it, on a yearly table and on a monthly table with yearly values
    ("2023년 서울 출생아 수는 전년보다 몇 명 늘었어?", "FAKE5_BIRTH", "11", "2022", "2023", False),
    ("2020년 부산의 출생아 수, 작년에 비해 몇 퍼센트 줄었어?", "FAKE5_BIRTH", "21", "2019", "2020", True),
    ("대구 사망자 수 2019년 증가율을 앞 해와 견주면?", "FAKE5_DEATH", "22", "2018", "2019", True),
    ("2023년 인천 혼인 건수는 지난해보다 몇 건 변했어?", "FAKE5_MARRY_YM", "23", "2022", "2023", False),
    # a month and the month before it (across a year end too), or the same month a year before
    ("2024년 3월 서울 혼인 건수는 전월 대비 몇 건 늘었나요?", "FAKE5_MARRY_YM", "11", "202402", "202403", False),
    ("2023년 1월 광주 전입자 수, 지난달보다 몇 퍼센트 변했어?", "FAKE5_MOVE_YM", "24", "202212", "202301", True),
    ("부산 전입자 수 2024년 7월은 앞 달과 비교해 증감률이 얼마야?", "FAKE5_MOVE_YM", "21", "202406", "202407", True),
    ("2024년 5월 대구 혼인 건수는 전년 동월보다 몇 건 줄었어?", "FAKE5_MARRY_YM", "22", "202305", "202405", False),
    ("2023년 10월 인천 전입자 수가 작년 같은 달에 비해 몇 % 늘었는지 알려줘", "FAKE5_MOVE_YM", "23", "202210", "202310", True),
    ("2024년 2월 서울 전입자 수는 3개월 전보다 몇 명 많아졌어?", "FAKE5_MOVE_YM", "11", "202311", "202402", False),
])
def test_the_period_right_before_a_named_one(res, question, table, region, early, late, rate):
    facts, answer = ask(res, question, table)
    derived = [f for f in answer if f.derivation]
    assert len(derived) == 1, [(f.value, f.derivation) for f in answer]
    (got,) = derived
    assert refs_in(facts, got) == sorted([ref(table, region, early), ref(table, region, late)])
    a, b = cell_value(facts, ref(table, region, early)), cell_value(facts, ref(table, region, late))
    want = (b - a) / a * 100 if rate else b - a
    assert got.value == pytest.approx(want, abs=0.01)
    cells = sorted(f.source_ref for f in answer if f.source_ref)
    assert cells == sorted([ref(table, region, early), ref(table, region, late)])


def test_no_named_period_keeps_the_default(res):
    """'작년 대비' with no named period: the latest period against the one before it, as before."""
    facts, answer = ask(res, "서울 출생아 수 작년 대비 몇 명 늘었어?", "FAKE5_BIRTH")
    cells = sorted(f.source_ref for f in answer if f.source_ref)
    assert cells == sorted([ref("FAKE5_BIRTH", "11", "2022"), ref("FAKE5_BIRTH", "11", "2023")])


@pytest.mark.parametrize("question, early, late", [
    ("2022년 서울의 출생아 수와 사망자 수는 전년보다 각각 몇 퍼센트 변했어?", "2021", "2022"),
    ("서울 출생아 수와 사망자 수 각각 2019년 증감률을 직전 연도와 비교하면?", "2018", "2019"),
])
def test_two_tables_each_against_the_year_before(res, question, early, late):
    facts, answer = ask(res, question, "FAKE5_BIRTH", "FAKE5_DEATH")
    derived = [f for f in answer if f.derivation]
    assert len(derived) == 2
    for got, table in zip(derived, ("FAKE5_BIRTH", "FAKE5_DEATH")):
        assert refs_in(facts, got) == sorted([ref(table, "11", early), ref(table, "11", late)])
        a, b = cell_value(facts, ref(table, "11", early)), cell_value(facts, ref(table, "11", late))
        assert got.value == pytest.approx((b - a) / a * 100, abs=0.01)


# ---------------------------------------------------------------- C-1: three or more targets listed


@pytest.mark.parametrize("question, table, codes, year", [
    ("2021년 서울, 부산, 대구의 출생아 수를 각각 알려줘", "FAKE5_BIRTH", ("11", "21", "22"), "2021"),
    ("2018년 인천과 광주, 부산, 서울 사망자 수는 얼마야?", "FAKE5_DEATH", ("23", "24", "21", "11"), "2018"),
    ("2022년 20대, 30대, 50대 취업자 수 알려줘", "FAKE5_AGE", ("1", "2", "4"), "2022"),
])
def test_three_or_more_named_targets_are_listed(res, question, table, codes, year):
    facts, answer = ask(res, question, table)
    assert all(not f.derivation for f in answer)
    assert sorted(f.source_ref for f in answer) == sorted(ref(table, c, year) for c in codes)
