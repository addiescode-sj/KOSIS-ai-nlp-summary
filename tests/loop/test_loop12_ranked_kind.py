"""Loop 12 F-1 (analyze/rules.py _ranked_axis, shared/patterns.py candidate_nouns): a ranking that names its
candidates by a noun of their kind ("가장 많은 산업은", "제일 적은 대표자 연령대는", "어느 산업의") ranks the category
whose name's stem that noun starts with; the others are held at their totals.

Neighbours (loop 10's regression place): candidates named by a noun of a region with another category said only as
a condition ("산업 구분 없이 ... 가장 많은 시도") rank the regions; a ranking with no candidate noun of a category
ranks the regions; candidates named one by one stay loop 11's. The tables are made up (가상 데이터).
"""

from __future__ import annotations

from collections import defaultdict

import pytest

from conftest import write_table
from kosis_chat.adapters.parquet import load_table
from kosis_chat.calc import calculate
from kosis_chat.schemas.resources import load_resources

YEARS = ["2022", "2023"]
REGIONS = (("00", "전국"), ("11", "서울특별시"), ("26", "부산광역시"), ("27", "대구광역시"))
INDUSTRY = ("산업별", (("0", "계"), ("C", "제조업"), ("F", "건설업"), ("G", "도소매업")))
AGE = ("대표자연령별", (("0", "계"), ("1", "30세 미만"), ("2", "30~49세"), ("3", "50세 이상")))


def rows(second, third):
    """A count by region, and two other categories with totals; the details are spread so that each ranking differs."""
    base = {}
    for y, year in enumerate(YEARS):
        for r, (rc, _) in enumerate(REGIONS[1:], start=1):
            for i, (ic, _) in enumerate(second[1][1:], start=1):
                for a, (ac, _) in enumerate(third[1][1:], start=1):
                    base[(year, rc, ic, ac)] = 50 + 17 * ((i * 7 + r * 3 + a * 5) % 11) + y
    totals: dict = defaultdict(int)
    for (year, rc, ic, ac), v in base.items():
        for rr in (rc, "00"):
            for ii in (ic, "0"):
                for aa in (ac, "0"):
                    totals[(year, rr, ii, aa)] += v
    return [{"period": year, "item_id": "T1", "item_name": "사업체수", "unit": "개", "dim1_name": "시도별",
             "dim1_code": rc, "dim1_label": dict(REGIONS)[rc], "dim2_name": second[0], "dim2_code": ic,
             "dim2_label": dict(second[1])[ic], "dim3_name": third[0], "dim3_code": ac,
             "dim3_label": dict(third[1])[ac], "value_raw": str(v), "value": v}
            for (year, rc, ic, ac), v in totals.items()]


def catalog(title):
    return {"survey_name": "가상 조사", "organization": "가상 기관", "table_name": title, "period": "년",
            "url": "https://example.invalid/t"}


SPECS = {
    "FAKE12_IA": (lambda: rows(INDUSTRY, AGE), catalog("가상 시도별 산업별 대표자연령별 사업체")),
    "FAKE12_AI": (lambda: rows(AGE, INDUSTRY), catalog("가상 시도별 대표자연령별 산업별 사업체")),
}
ROWS = {t: make() for t, (make, _) in SPECS.items()}


@pytest.fixture(scope="module")
def res(tmp_path_factory):
    folder = tmp_path_factory.mktemp("loop12f") / "data"
    return {t: load_resources(load_table(write_table(folder, t, ROWS[t], cat))) for t, (_, cat) in SPECS.items()}


def answer_cells(res, question, tbl):
    got = calculate(question, [res[tbl]])
    assert got.status == "ok", (got.status, got.reason, got.detail)
    facts = {f.id: f for f in got.facts}
    return [(facts[i].source_ref, facts[i].value) for i in got.answer]


def expected(tbl, region, year, ranked, pick=max):
    """The cell the ranking picks: over `ranked` ("dim2" or "dim3"), the other category at its total."""
    other = "dim3" if ranked == "dim2" else "dim2"
    pool = [r for r in ROWS[tbl] if r["dim1_code"] == region and r["period"] == year and r[f"{other}_code"] == "0"
            and r[f"{ranked}_code"] != "0"]
    best = pick(pool, key=lambda r: r["value"])
    return f"{tbl}|T1|{region}|{best['dim2_code']}|{best['dim3_code']}|{year}", best["value"]


@pytest.mark.parametrize("question, tbl, region, year, ranked, pick", [
    ("2023년 서울 사업체 수가 가장 많은 대표자 연령대는?", "FAKE12_IA", "11", "2023", "dim3", max),
    ("2022년 부산에서 사업체가 제일 적은 대표자연령은?", "FAKE12_IA", "26", "2022", "dim3", min),
    ("2023년 서울에서 사업체 수가 가장 많은 산업은?", "FAKE12_AI", "11", "2023", "dim3", max),
    ("2023년 대구 사업체가 제일 적은 산업 분야는 어디야?", "FAKE12_AI", "27", "2023", "dim3", min),
    ("2022년 대구에서 어느 산업의 사업체가 가장 많아?", "FAKE12_AI", "27", "2022", "dim3", max),
])
def test_ranking_of_the_category_its_candidate_noun_names(res, question, tbl, region, year, ranked, pick):
    assert answer_cells(res, question, tbl) == [expected(tbl, region, year, ranked, pick)]


@pytest.mark.parametrize("question, tbl", [
    ("2023년 산업 구분 없이 사업체가 가장 많은 시도는?", "FAKE12_AI"),
    ("2023년 대표자 연령과 관계없이 사업체가 가장 많은 지역은 어디야?", "FAKE12_IA"),
    ("2023년 사업체가 가장 많은 곳은?", "FAKE12_IA"),
])
def test_neighbour_regions_ranked_when_named_by_a_region_noun(res, question, tbl):
    (got,) = answer_cells(res, question, tbl)
    totals = [r for r in ROWS[tbl] if r["period"] == "2023" and r["dim2_code"] == "0" and r["dim3_code"] == "0"
              and r["dim1_code"] != "00"]
    best = max(totals, key=lambda r: r["value"])
    assert got == (f"{tbl}|T1|{best['dim1_code']}|0|0|2023", best["value"])


def test_neighbour_candidates_named_one_by_one(res):
    (got,) = answer_cells(res, "2023년 서울의 제조업, 건설업, 도소매업 중 사업체가 가장 많은 산업은?", "FAKE12_IA")
    assert got == expected("FAKE12_IA", "11", "2023", "dim2", max)
