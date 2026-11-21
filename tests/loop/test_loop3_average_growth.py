"""Loop 3, one table, the average yearly rate of change (analyze/rules.py _operation -> _cagr_plan, analyze/blocks.py
average_growth; shared/patterns.py RATE_OF_CHANGE with 씩): an average word beside a rate of change ("연평균 증가율",
"해마다 평균 몇 퍼센트씩 늘었어", "연평균 성장률") asks for ((last / first) ** (1 / years) - 1) * 100 from the first and the
last asked year's cells, not for the mean of the values in between. An average of the values stays as it was.

The tables are made up (가상 데이터). Every test checks the answer's value and the cells it was computed from.
"""

from __future__ import annotations

import pytest

from conftest import write_table
from kosis_chat.adapters.parquet import load_table
from kosis_chat.calc import calculate
from kosis_chat.schemas.resources import load_resources

CATALOG = {"survey_name": "가상 조사", "organization": "가상 기관", "table_name": "가상 표", "period": "년",
           "url": "https://example.invalid/t"}
YEARS = tuple(str(y) for y in range(2015, 2024))
REGIONS = {"00": "전국", "11": "서울특별시", "21": "부산광역시", "31": "경기도"}


def value(item, code, year):
    n = int(year) - 2015
    base = {"00": 50000, "11": 9000, "21": 3500, "31": 13000}[code]
    if item == "T1":
        return base + 37 * n * n + 11 * n
    return 300 + 9 * n + int(code)  # 사업체수: another item and unit


def rows(tbl):
    out = []
    for period in YEARS:
        for code, label in REGIONS.items():
            for item, name, unit in (("T1", "가구수", "가구"), ("T2", "사업체수", "개")):
                v = value(item, code, period)
                out.append({"period": period, "item_id": item, "item_name": name, "unit": unit,
                            "dim1_name": "행정구역별", "dim1_code": code, "dim1_label": label,
                            "value_raw": str(v), "value": v})
    return out


@pytest.fixture(scope="module")
def res(tmp_path_factory):
    folder = tmp_path_factory.mktemp("loop3_cagr") / "data"
    return load_resources(load_table(write_table(folder, "FAKE_CAGR", rows("FAKE_CAGR"), CATALOG)))


def answer(res, question):
    got = calculate(question, [res])
    assert got.status == "ok", (got.status, got.reason, got.detail)
    facts = {f.id: f for f in got.facts}
    return [facts[i] for i in got.answer]


def ref(item, code, year):
    return f"FAKE_CAGR|{item}|{code}|{year}"


def cagr(item, code, early, late):
    return ((value(item, code, late) / value(item, code, early)) ** (1 / (int(late) - int(early))) - 1) * 100


@pytest.mark.parametrize("question, item, code, early, late", [
    ("2015년부터 2020년까지 서울 가구수의 연평균 증가율은?", "T1", "11", "2015", "2020"),
    ("2017~2023년 부산 가구수는 해마다 평균 몇 퍼센트씩 늘었어?", "T1", "21", "2017", "2023"),
    ("2016년과 2022년 사이 경기도 사업체수의 연평균 성장률 알려줘", "T2", "31", "2016", "2022"),
    ("최근 5년 전국 가구수 연평균 증가율은?", "T1", "00", "2019", "2023"),
])
def test_average_yearly_rate_from_the_two_ends(res, question, item, code, early, late):
    facts = answer(res, question)
    assert sorted(f.source_ref for f in facts if f.source_ref) == [ref(item, code, early), ref(item, code, late)]
    assert [f.value for f in facts if not f.source_ref] == [pytest.approx(cagr(item, code, early, late), abs=0.005)]


def test_an_average_of_values_stays_a_mean(res):
    facts = answer(res, "2015~2017년 서울 가구수 평균은?")
    mean = sum(value("T1", "11", y) for y in ("2015", "2016", "2017")) / 3
    assert [f.value for f in facts if not f.source_ref] == [pytest.approx(mean, abs=0.005)]
