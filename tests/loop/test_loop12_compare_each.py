"""Loop 12 D (analyze/rules.py _compared_each, analyze/blocks.py repeated): a comparison of two codes of one category
asked for each of the codes of another ("서울은 부산보다 남자와 여자가 각각 몇 명 많아"). The category the word naming the
reference binds is compared; the other, its codes each named with a word asking for each, is repeated. Each repeat is
the comparison one target gets (difference, ratio), in question order.

Neighbours: the values of every combination asked without a calculation (listed); a pair of two names across the
categories with no word for each (left as it was). The tables are made up (가상 데이터).
"""

from __future__ import annotations

import re

import pytest

from conftest import write_table
from kosis_chat.adapters.parquet import load_table
from kosis_chat.calc import calculate
from kosis_chat.schemas.resources import load_resources

YEARS = ["2021", "2022", "2023"]
REGIONS = (("00", "전국"), ("11", "서울특별시"), ("26", "부산광역시"), ("27", "대구광역시"))
SEXES = (("0", "계"), ("1", "남자"), ("2", "여자"))
AGES = (("A0", "계"), ("A1", "청년"), ("A2", "중장년"), ("A3", "노년"))


def rows_by(second, second_name, items):
    out = []
    for y, year in enumerate(YEARS):
        for r, (c, l) in enumerate(REGIONS):
            for s, (sc, sl) in enumerate(second):
                for k, (ic, iname, u) in enumerate(items):
                    v = 1000 + 100 * k + 37 * r + 11 * s + 5 * y + r * s * 3 + k * s * 7
                    out.append({"period": year, "item_id": ic, "item_name": iname, "unit": u, "dim1_name": "시도별",
                                "dim1_code": c, "dim1_label": l, "dim2_name": second_name, "dim2_code": sc,
                                "dim2_label": sl, "value_raw": str(v), "value": v})
    return out


def catalog(title):
    return {"survey_name": "가상 조사", "organization": "가상 기관", "table_name": title, "period": "년",
            "url": "https://example.invalid/t"}


SPECS = {
    "FAKE12_SEX": (lambda: rows_by(SEXES, "성별", (("B", "출생아수", "명"), ("D", "사망자수", "명"))),
                   catalog("가상 시도별 성별 출생 사망")),
    "FAKE12_AGE": (lambda: rows_by(AGES, "연령별", (("E", "취업자수", "명"), ("U", "실업자수", "명"))),
                   catalog("가상 시도별 연령별 고용")),
}
ROWS = {t: make() for t, (make, _) in SPECS.items()}


@pytest.fixture(scope="module")
def res(tmp_path_factory):
    folder = tmp_path_factory.mktemp("loop12d") / "data"
    return {t: load_resources(load_table(write_table(folder, t, ROWS[t], cat))) for t, (_, cat) in SPECS.items()}


def ask(res, question, tbl):
    got = calculate(question, [res[tbl]])
    assert got.status == "ok", (got.status, got.reason, got.detail)
    facts = {f.id: f for f in got.facts}
    return facts, [facts[i] for i in got.answer]


def refs_in(facts, fact):
    return [facts[i].source_ref for i in re.findall(r"f\d+", fact.derivation or "")]


def value(tbl, item, region, second, year):
    return next(r["value"] for r in ROWS[tbl] if r["item_id"] == item and r["dim1_code"] == region
                and r["dim2_code"] == second and r["period"] == year)


def ref(tbl, item, region, second, year):
    return f"{tbl}|{item}|{region}|{second}|{year}"


def computed(answer):
    return [f for f in answer if not f.source_ref]


def cells(answer):
    return sorted(f.source_ref for f in answer if f.source_ref)


@pytest.mark.parametrize("question, year", [
    ("2023년 서울은 부산보다 남자와 여자 출생아 수가 각각 몇 명 많아?", "2023"),   # the repeated category: 성별
    ("2021년 남자와 여자 각각 서울 출생아 수는 부산보다 얼마나 많아?", "2021"),
])
def test_difference_of_two_regions_for_each_sex(res, question, year):
    facts, answer = ask(res, question, "FAKE12_SEX")
    diffs = computed(answer)
    assert len(diffs) == 2
    for got, sex in zip(diffs, ("1", "2")):  # question order
        assert got.value == pytest.approx(value("FAKE12_SEX", "B", "11", sex, year)
                                          - value("FAKE12_SEX", "B", "26", sex, year), abs=0.01)
        assert refs_in(facts, got) == [ref("FAKE12_SEX", "B", "11", sex, year), ref("FAKE12_SEX", "B", "26", sex, year)]
    assert len(cells(answer)) == 4


def test_difference_of_two_items_for_each_region(res):
    """The repeated category: the regions; the compared one: the items."""
    facts, answer = ask(res, "2022년 서울과 대구 각각 취업자 수는 실업자 수보다 몇 명 많아?", "FAKE12_AGE")
    diffs = computed(answer)
    assert len(diffs) == 2
    for got, region in zip(diffs, ("11", "27")):
        assert got.value == pytest.approx(value("FAKE12_AGE", "E", region, "A0", "2022")
                                          - value("FAKE12_AGE", "U", region, "A0", "2022"), abs=0.01)
        assert refs_in(facts, got) == [ref("FAKE12_AGE", "E", region, "A0", "2022"),
                                       ref("FAKE12_AGE", "U", region, "A0", "2022")]


def test_ratio_of_two_items_for_each_region(res):
    facts, answer = ask(res, "2022년 서울과 부산의 출생아 수는 각각 사망자 수의 몇 배야?", "FAKE12_SEX")
    ratios = computed(answer)
    assert len(ratios) == 2
    for got, region in zip(ratios, ("11", "26")):
        a, b = value("FAKE12_SEX", "B", region, "0", "2022"), value("FAKE12_SEX", "D", region, "0", "2022")
        assert got.value == pytest.approx(round(a / b, 2), abs=0.01)
        assert refs_in(facts, got) == [ref("FAKE12_SEX", "B", region, "0", "2022"),
                                       ref("FAKE12_SEX", "D", region, "0", "2022")]


def test_difference_for_each_of_three_age_groups(res):
    facts, answer = ask(res, "2023년 청년, 중장년, 노년 각각 부산 실업자 수는 대구보다 몇 명 적어?", "FAKE12_AGE")
    diffs = computed(answer)
    assert len(diffs) == 3
    for got, age in zip(diffs, ("A1", "A2", "A3")):
        assert sorted(refs_in(facts, got)) == sorted([ref("FAKE12_AGE", "U", "26", age, "2023"),
                                                      ref("FAKE12_AGE", "U", "27", age, "2023")])
        assert abs(got.value) == pytest.approx(abs(value("FAKE12_AGE", "U", "26", age, "2023")
                                                   - value("FAKE12_AGE", "U", "27", age, "2023")), abs=0.01)


def test_neighbour_one_target_compared(res):
    facts, answer = ask(res, "2023년 서울 남자 출생아 수는 부산보다 몇 명 많아?", "FAKE12_SEX")
    (got,) = computed(answer)
    assert refs_in(facts, got) == [ref("FAKE12_SEX", "B", "11", "1", "2023"), ref("FAKE12_SEX", "B", "26", "1", "2023")]


def test_neighbour_values_of_every_combination(res):
    _, answer = ask(res, "2023년 서울과 부산의 남자와 여자 출생아 수 알려줘", "FAKE12_SEX")
    assert cells(answer) == sorted(ref("FAKE12_SEX", "B", r, s, "2023") for r in ("11", "26") for s in ("1", "2"))
    assert not computed(answer)


def test_neighbour_pair_across_categories_without_a_word_for_each(res):
    got = calculate("2023년 서울 남자와 부산 여자의 출생아 수 차이는?", [res["FAKE12_SEX"]])
    assert got.status == "no_plan"
