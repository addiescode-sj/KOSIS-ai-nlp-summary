"""Loop 12 A-1 (judge/rules.py _named_apart): a parent named before its child with a word between them.

A question that names a province before one of its districts with something between them (a one-letter unit
suffix "서울시", a particle "서울에", a word of where "서울 소재", a word of a kind of place "부산 지역") names where
the district is: only the district is a target. The parent stays a target only where the question names it as one:
side by side with a joining particle, as the reference or the subject of a comparison, or before a word of looking
inside (a share's whole, a ranking's pool). The same function decides an item category's whole and its details.

The tables are made up (가상 데이터); every test checks the values and the cells they come from.
"""

from __future__ import annotations

import re

import pytest

from conftest import write_table
from kosis_chat.adapters.parquet import load_table
from kosis_chat.calc import calculate
from kosis_chat.schemas.resources import load_resources

YEARS = ["2020", "2021", "2022", "2023"]
REGIONS = (("00", "전국"), ("11", "서울특별시"), ("11010", "종로구"), ("11020", "용산구"), ("11030", "마포구"),
           ("26", "부산광역시"), ("26010", "해운대구"), ("26020", "수영구"), ("47", "경상북도"), ("47010", "포항시"),
           ("47020", "경주시"))
INDUSTRY = (("E0", "취업자"), ("E1", "농림어업"), ("E2", "제조업"), ("E3", "서비스업"))


def _region_values(y, scale):
    d = {"11010": 150 + y, "11020": 230 - 2 * y, "11030": 370 + 3 * y, "26010": 410 - y, "26020": 170 + 2 * y,
         "47010": 500 + 4 * y, "47020": 250 - 3 * y}
    d = {k: v * scale for k, v in d.items()}
    d["11"] = d["11010"] + d["11020"] + d["11030"]
    d["26"] = d["26010"] + d["26020"]
    d["47"] = d["47010"] + d["47020"]
    d["00"] = d["11"] + d["26"] + d["47"]
    return d


def region_rows(item, name, unit, scale, dim):
    out = []
    for y, year in enumerate(YEARS):
        vals = _region_values(y, scale)
        out += [{"period": year, "item_id": item, "item_name": name, "unit": unit, "dim1_name": dim, "dim1_code": c,
                 "dim1_label": l, "value_raw": str(vals[c]), "value": vals[c]} for c, l in REGIONS]
    return out


def industry_rows():
    """취업자 (a marked-less whole the title names) = the three industries, by 시도."""
    out = []
    for y, year in enumerate(YEARS):
        for r, (code, label) in enumerate((("11", "서울특별시"), ("26", "부산광역시"))):
            d = [40 + y + 20 * r, 400 + 7 * y - 30 * r, 900 + 13 * y - 70 * r]
            out += [{"period": year, "item_id": c, "item_name": n, "unit": "천명", "dim1_name": "시도별",
                     "dim1_code": code, "dim1_label": label, "value_raw": str(v), "value": v}
                    for (c, n), v in zip(INDUSTRY, [sum(d), *d])]
    return out


def catalog(title):
    return {"survey_name": "가상 조사", "organization": "가상 기관", "table_name": title, "period": "년",
            "url": "https://example.invalid/t"}


SPECS = {
    "FAKE12_POP": (lambda: region_rows("T1", "인구", "천명", 1, "행정구역별"), catalog("가상 행정구역별 인구")),
    "FAKE12_HH": (lambda: region_rows("H1", "가구", "가구", 3, "시군구별"), catalog("가상 시군구별 가구")),
    "FAKE12_IND": (industry_rows, catalog("가상 시도별 산업별 취업자")),
}
ROWS = {t: make() for t, (make, _) in SPECS.items()}


@pytest.fixture(scope="module")
def res(tmp_path_factory):
    folder = tmp_path_factory.mktemp("loop12p") / "data"
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


def value(tbl, code, year, item=None):
    return next(r["value"] for r in ROWS[tbl] if r["dim1_code"] == code and r["period"] == year
                and (item is None or r["item_id"] == item))


def ref(tbl, code, year):
    item = ROWS[tbl][0]["item_id"]
    return f"{tbl}|{item}|{code}|{year}"


def cells(answer):
    return sorted(f.source_ref for f in answer if f.source_ref)


def computed(answer):
    return [f for f in answer if not f.source_ref]


# ---------------------------------------------------------------- the parent is where the child is


@pytest.mark.parametrize("question, tbl, code, year", [
    ("2023년 서울시 종로구 인구는?", "FAKE12_POP", "11010", "2023"),           # a one-letter unit suffix
    ("2021년 부산시의 수영구 인구는 몇 명이야?", "FAKE12_POP", "26020", "2021"),  # suffix and 의
    ("2022년 서울 소재 마포구 가구 수 알려줘", "FAKE12_HH", "11030", "2022"),    # a word of where
    ("2020년 경북 지역 포항시 가구는?", "FAKE12_HH", "47010", "2020"),         # a word of a kind of place
    ("2023년 경상북도에 있는 경주시 인구는?", "FAKE12_POP", "47020", "2023"),    # a particle and a verb of being
])
def test_value_of_a_district_named_after_its_province(res, question, tbl, code, year):
    _, answer = ask(res, question, tbl)
    assert cells(answer) == [ref(tbl, code, year)]
    assert answer[0].value == value(tbl, code, year)


@pytest.mark.parametrize("question, tbl, a, b, year", [
    ("2023년 서울에 있는 종로구와 용산구 인구 차이는?", "FAKE12_POP", "11010", "11020", "2023"),
    ("2021년 경북도 포항시와 경주시의 가구 수 차이는 얼마야?", "FAKE12_HH", "47010", "47020", "2021"),
])
def test_difference_of_two_districts_named_after_their_province(res, question, tbl, a, b, year):
    facts, answer = ask(res, question, tbl)
    calc = computed(answer)
    assert len(calc) == 1
    assert refs_in(facts, calc[0]) == sorted([ref(tbl, a, year), ref(tbl, b, year)])
    assert abs(calc[0].value) == pytest.approx(abs(value(tbl, a, year) - value(tbl, b, year)), abs=0.01)
    assert cells(answer) == sorted([ref(tbl, a, year), ref(tbl, b, year)])


@pytest.mark.parametrize("question, tbl, code, early, late", [
    ("2020년 대비 2023년 부산 지역 해운대구 인구는 얼마나 늘었어?", "FAKE12_POP", "26010", "2020", "2023"),
    ("2021년부터 2023년까지 서울시 마포구 가구는 몇 가구 늘었어?", "FAKE12_HH", "11030", "2021", "2023"),
])
def test_change_of_a_district_named_after_its_province(res, question, tbl, code, early, late):
    facts, answer = ask(res, question, tbl)
    assert cells(answer) == sorted([ref(tbl, code, early), ref(tbl, code, late)])
    amounts = [f for f in computed(answer) if "/" not in (f.derivation or "")]
    assert amounts and amounts[0].value == pytest.approx(value(tbl, code, late) - value(tbl, code, early), abs=0.01)


def test_detail_named_after_the_whole_with_a_word_between(res):
    """The item category's whole said like a kind ("취업자 가운데 ..." is a scope; "취업자 부문 제조업" is not)."""
    _, answer = ask(res, "2023년 서울 취업자 부문 제조업은 몇 명이야?", "FAKE12_IND")
    assert [f.source_ref for f in answer] == ["FAKE12_IND|E2|11|2023"]
    assert answer[0].value == value("FAKE12_IND", "11", "2023", "E2")


# ---------------------------------------------------------------- neighbours: the parent named as a target


def test_neighbour_parent_and_child_joined_side_by_side(res):
    _, answer = ask(res, "2023년 서울과 종로구 인구를 각각 알려줘", "FAKE12_POP")
    assert cells(answer) == sorted([ref("FAKE12_POP", "11", "2023"), ref("FAKE12_POP", "11010", "2023")])


def test_neighbour_parent_as_the_subject_of_a_ratio(res):
    facts, answer = ask(res, "2022년 서울은 종로구의 몇 배야?", "FAKE12_POP")
    calc = computed(answer)
    assert len(calc) == 1
    assert calc[0].value == pytest.approx(round(value("FAKE12_POP", "11", "2022") / value("FAKE12_POP", "11010", "2022"), 2), abs=0.01)
    assert refs_in(facts, calc[0]) == sorted([ref("FAKE12_POP", "11", "2022"), ref("FAKE12_POP", "11010", "2022")])


def test_neighbour_parent_as_the_reference_of_a_percent(res):
    facts, answer = ask(res, "2023년 종로구 인구는 서울 대비 몇 %야?", "FAKE12_POP")
    calc = computed(answer)
    assert calc[0].value == pytest.approx(round(value("FAKE12_POP", "11010", "2023") / value("FAKE12_POP", "11", "2023") * 100, 2), abs=0.01)
    assert refs_in(facts, calc[0]) == sorted([ref("FAKE12_POP", "11", "2023"), ref("FAKE12_POP", "11010", "2023")])


def test_neighbour_parent_as_the_whole_of_a_share(res):
    facts, answer = ask(res, "2023년 서울 중 종로구가 차지하는 비중은?", "FAKE12_POP")
    calc = computed(answer)
    assert calc[0].value == pytest.approx(round(value("FAKE12_POP", "11010", "2023") / value("FAKE12_POP", "11", "2023") * 100, 2), abs=0.01)


def test_neighbour_parent_as_the_pool_of_a_ranking(res):
    _, answer = ask(res, "2023년 서울 안에서 인구가 가장 많은 구는?", "FAKE12_POP")
    assert cells(answer) == [ref("FAKE12_POP", "11030", "2023")]


def test_neighbour_district_joined_right_after_its_province(res):
    _, answer = ask(res, "2023년 서울 종로구 인구는?", "FAKE12_POP")
    assert cells(answer) == [ref("FAKE12_POP", "11010", "2023")]


def test_neighbour_whole_and_detail_joined_side_by_side(res):
    _, answer = ask(res, "2023년 서울의 취업자와 제조업 취업자는 각각 몇 명이야?", "FAKE12_IND")
    assert cells(answer) == ["FAKE12_IND|E0|11|2023", "FAKE12_IND|E2|11|2023"]
