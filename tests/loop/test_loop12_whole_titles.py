"""Loop 12 C-4 (prepare/index.py title_noun, _unmarked_whole, _link_sections): the unmarked whole of an item category
read from titles of the shapes real survey tables have:

- a tail naming the regional unit joined to the end by a dash, an underscore or a space ('취업자 - 시도', '취업자_시군구');
- brackets after the title's noun, the first item's name putting their words first or writing the title's words
  together ('취업자(15세 이상)' beside '15세 이상 취업자', '경제 활동 인구' beside '경제활동인구');
- marked subtotals after the whole, the last of them with no detail in the table.

Once the whole is known the existing judgements hear it: a detail named with the whole's name after it is that
detail, a ranking of it ranks regions, the whole before a word of looking inside is a share's denominator.
Neighbours: items in different units, a title naming two measures side by side, a marked total first.
The tables are made up (가상 데이터); every test checks the values and the cells they come from.
"""

from __future__ import annotations

import re

import pytest

from conftest import write_table
from kosis_chat.adapters.parquet import load_table
from kosis_chat.analyze.rules import _total
from kosis_chat.calc import calculate
from kosis_chat.prepare.index import title_noun
from kosis_chat.schemas.resources import load_resources

YEARS = ["2021", "2022", "2023"]
REGIONS = (("00", "전국"), ("11", "서울특별시"), ("26", "부산광역시"), ("27", "대구광역시"))
IND = (("E0", "취업자"), ("E1", "농림어업"), ("E2", "제조업"), ("E3", "서비스업"))
IND15 = (("E0", "15세 이상 취업자"), ("E1", "농림어업"), ("E2", "제조업"), ("E3", "서비스업"))
ACT = (("P0", "경제활동인구"), ("P1", "남자"), ("P2", "여자"))
STATUS = (("S0", "취업자"), ("S1", "임금근로자-계"), ("S2", "상용근로자"), ("S3", "임시근로자"), ("S4", "비임금근로자-계"))
TWO = (("N0", "출생아"), ("N1", "사망자"), ("N2", "전입자"))
MIXED = (("M0", "사업체"), ("M1", "종사자"), ("M2", "매출액"))
MARKED = (("K0", "근로자-계"), ("K1", "정규직"), ("K2", "비정규직"))


def rows(items, units, sums):
    """Each detail by region and year; a whole is the sum of its parts. 부산 has the most of the second item."""
    out = []
    units = units if isinstance(units, tuple) else (units,) * len(items)
    for y, year in enumerate(YEARS):
        for r, (code, label) in enumerate(REGIONS):
            vals = {ic: 100 + 40 * k + 9 * y - 13 * r + 90 * (r == 2) * (k == 1) for k, (ic, _) in enumerate(items)}
            for whole, parts in sums:
                vals[whole] = sum(vals[p] for p in parts)
            out += [{"period": year, "item_id": ic, "item_name": n, "unit": u, "dim1_name": "시도별", "dim1_code": code,
                     "dim1_label": label, "value_raw": str(vals[ic]), "value": vals[ic]}
                    for (ic, n), u in zip(items, units)]
    return out


def catalog(title):
    return {"survey_name": "가상 조사", "organization": "가상 기관", "table_name": title, "period": "년",
            "url": "https://example.invalid/t"}


SPECS = {
    "FAKE12_DASH": (lambda: rows(IND, "천명", [("E0", ["E1", "E2", "E3"])]), catalog("가상 산업별 취업자 - 시도")),
    "FAKE12_UNDER": (lambda: rows(IND, "천명", [("E0", ["E1", "E2", "E3"])]), catalog("가상 산업별 취업자_시군구")),
    "FAKE12_PAREN": (lambda: rows(IND15, "천명", [("E0", ["E1", "E2", "E3"])]), catalog("가상 시도별 취업자(15세 이상)")),
    "FAKE12_SPACED": (lambda: rows(ACT, "천명", [("P0", ["P1", "P2"])]), catalog("가상 성별 경제 활동 인구-시도")),
    "FAKE12_STATUS": (lambda: rows(STATUS, "천명", [("S1", ["S2", "S3"]), ("S0", ["S1", "S4"])]),
                      catalog("가상 종사상지위별 취업자 - 시도")),
    "FAKE12_TWO": (lambda: rows(TWO, "명", []), catalog("가상 시도별 출생아 및 사망자 - 시도")),
    "FAKE12_MIXED": (lambda: rows(MIXED, ("개", "명", "백만원"), []), catalog("가상 시도별 사업체 - 시도")),
    "FAKE12_MARKED": (lambda: rows(MARKED, "명", [("K0", ["K1", "K2"])]), catalog("가상 고용형태별 근로자 - 시도")),
}
ROWS = {t: make() for t, (make, _) in SPECS.items()}


@pytest.fixture(scope="module")
def res(tmp_path_factory):
    folder = tmp_path_factory.mktemp("loop12w") / "data"
    return {t: load_resources(load_table(write_table(folder, t, ROWS[t], cat))) for t, (_, cat) in SPECS.items()}


def ask(res, question, tbl):
    got = calculate(question, [res[tbl]])
    assert got.status == "ok", (got.status, got.reason, got.detail)
    facts = {f.id: f for f in got.facts}
    return facts, [facts[i] for i in got.answer]


def refs_in(facts, fact):
    return [facts[i].source_ref for i in re.findall(r"f\d+", fact.derivation or "")]


def value(tbl, item, region, year):
    return next(r["value"] for r in ROWS[tbl] if r["item_id"] == item and r["dim1_code"] == region
                and r["period"] == year)


def ref(tbl, item, region, year):
    return f"{tbl}|{item}|{region}|{year}"


def cells(answer):
    return sorted(f.source_ref for f in answer if f.source_ref)


def computed(answer):
    return [f for f in answer if not f.source_ref]


@pytest.mark.parametrize("title, noun", [
    ("가상 산업별 취업자 - 시도", "취업자"),
    ("가상 산업별 취업자_시군구", "취업자"),
    ("가상 산업별 취업자-시도/시군구", "취업자"),
    ("가상 성별 경제활동인구 (시도)", "경제활동인구"),
    ("가상 시도별 산업별 취업자", "취업자"),          # loop 11's shape stays
])
def test_title_noun_without_its_regional_tail(title, noun):
    assert title_noun(title) == noun


@pytest.mark.parametrize("tbl, whole", [
    ("FAKE12_DASH", "E0"), ("FAKE12_UNDER", "E0"), ("FAKE12_PAREN", "E0"), ("FAKE12_SPACED", "P0"),
    ("FAKE12_STATUS", "S0"),
])
def test_unmarked_whole_of_real_title_shapes(res, tbl, whole):
    assert _total(res[tbl].index, "항목") == whole


def test_last_subtotal_without_details_keeps_the_sections(res):
    parents = res["FAKE12_STATUS"].index.axes["항목"].parents
    assert parents == {"S0": None, "S1": "S0", "S2": "S1", "S3": "S1", "S4": "S0"}


@pytest.mark.parametrize("question, tbl, item, region, year", [
    ("2023년 제조업 취업자가 가장 많은 시도는?", "FAKE12_DASH", "E2", "11", "2023"),
    ("2022년 농림어업 취업자가 제일 많은 지역은 어디야?", "FAKE12_UNDER", "E1", "26", "2022"),
    ("2023년 서비스업 취업자가 가장 적은 시도는?", "FAKE12_PAREN", "E3", "27", "2023"),
])
def test_ranking_of_a_detail_named_with_the_whole(res, question, tbl, item, region, year):
    _, answer = ask(res, question, tbl)
    assert cells(answer) == [ref(tbl, item, region, year)]


@pytest.mark.parametrize("question, tbl, item, region, year", [
    ("2023년 서울 제조업 취업자는 몇 명이야?", "FAKE12_DASH", "E2", "11", "2023"),
    ("2021년 부산 상용근로자 취업자 수 알려줘", "FAKE12_STATUS", "S2", "26", "2021"),
    ("2022년 서울 취업자 중 제조업은 몇 명이야?", "FAKE12_UNDER", "E2", "11", "2022"),
])
def test_detail_alone_when_the_whole_is_named_beside_it(res, question, tbl, item, region, year):
    _, answer = ask(res, question, tbl)
    assert cells(answer) == [ref(tbl, item, region, year)]


@pytest.mark.parametrize("question, tbl, part, whole, region, year", [
    ("2022년 서울 취업자 중 제조업 비중은?", "FAKE12_DASH", "E2", "E0", "11", "2022"),
    ("2023년 부산 경제활동인구 가운데 여자가 차지하는 비중은?", "FAKE12_SPACED", "P2", "P0", "26", "2023"),
    ("2023년 서울 임금근로자 중 상용근로자 비중은?", "FAKE12_STATUS", "S2", "S1", "11", "2023"),
    ("2023년 서울 취업자 중 비임금근로자 비중은?", "FAKE12_STATUS", "S4", "S0", "11", "2023"),
])
def test_share_of_the_whole_or_its_subtotal(res, question, tbl, part, whole, region, year):
    facts, answer = ask(res, question, tbl)
    (got,) = computed(answer)
    assert got.value == pytest.approx(round(value(tbl, part, region, year) / value(tbl, whole, region, year) * 100, 2),
                                      abs=0.01)
    assert refs_in(facts, got) == [ref(tbl, part, region, year), ref(tbl, whole, region, year)]


# ---------------------------------------------------------------- neighbours


def test_neighbour_title_naming_two_measures_has_no_whole(res):
    index = res["FAKE12_TWO"].index
    assert _total(index, "항목") is None
    _, answer = ask(res, "2023년 서울 사망자는 몇 명이야?", "FAKE12_TWO")
    assert cells(answer) == [ref("FAKE12_TWO", "N1", "11", "2023")]


def test_neighbour_items_in_different_units_have_no_whole(res):
    assert _total(res["FAKE12_MIXED"].index, "항목") is None
    assert all(p is None for p in res["FAKE12_MIXED"].index.axes["항목"].parents.values())


def test_neighbour_marked_total_stays(res):
    assert _total(res["FAKE12_MARKED"].index, "항목") == "K0"


def test_neighbour_whole_and_detail_side_by_side(res):
    _, answer = ask(res, "2023년 서울의 취업자와 제조업 취업자는 각각 몇 명이야?", "FAKE12_DASH")
    assert cells(answer) == sorted([ref("FAKE12_DASH", "E0", "11", "2023"), ref("FAKE12_DASH", "E2", "11", "2023")])
