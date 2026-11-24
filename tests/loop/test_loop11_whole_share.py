"""Loop 11 (prepare/index.py, judge/rules.py, analyze/rules.py):

- A-1: an item category whose first name is the whole of the others without a total mark (the noun the table's title
  ends with, every item in one unit) is the category's total: the item a question leaves out, the denominator of a
  share, never a ranking candidate. Two shapes: the whole and its details; the whole, marked subtotals and details.
- A-2: the whole's name put after a detail's name ("제조업 취업자") names the detail; named apart from it ("취업자와
  제조업 취업자", "취업자는 제조업의 몇 배") it stays a target.
- E-3: a share inside a group of other categories ("서울 남자 중"): the whole is the item category's total; the whole
  named before the word of looking inside is the denominator; a subtotal so named stays the denominator.
- E-2: a category whose only code is named before the word of looking inside is where the share is taken; the part
  is the other category's code named after it.
- E-1: a percent asked of a part after a word of looking inside, without a share word, is that share; a rate cell
  asked so is its value.

Neighbours: tables whose first item is not such a whole (items in different units; a marked total) keep their index
and answers. The tables are made up (가상 데이터); every test checks the values and the cells they come from.
"""

from __future__ import annotations

import re

import pytest

from conftest import write_table
from kosis_chat.adapters.parquet import load_table
from kosis_chat.analyze.rules import _total
from kosis_chat.calc import calculate
from kosis_chat.schemas.resources import load_resources

REGIONS = (("00", "전국"), ("11", "서울특별시"), ("21", "부산광역시"), ("22", "대구광역시"))
SEXES = (("0", "계"), ("1", "남자"), ("2", "여자"))
YEARS = ["2020", "2021", "2022", "2023"]
INDUSTRY = (("E0", "취업자"), ("E1", "농림어업"), ("E2", "제조업"), ("E3", "서비스업"))
STATUS = (("S0", "취업자"), ("S1", "임금근로자-계"), ("S2", "상용근로자"), ("S3", "임시근로자"),
          ("S4", "비임금근로자-계"), ("S5", "자영업자"), ("S6", "무급가족종사자"))
FORM = (("K0", "근로자-계"), ("K1", "정규직"), ("K2", "비정규직"))
MIXED = (("M0", "사업체", "개"), ("M1", "종사자", "명"), ("M2", "매출액", "백만원"))


def _row(year, item, name, unit, region, sex, v):
    out = {"period": year, "item_id": item, "item_name": name, "unit": unit, "dim1_name": "시도별",
           "dim1_code": region[0], "dim1_label": region[1], "value_raw": str(v), "value": v}
    if sex is not None:
        out.update(dim2_name="성별", dim2_code=sex[0], dim2_label=sex[1])
    return out


def industry_rows():
    """취업자 (the whole, no mark) = 농림어업 + 제조업 + 서비스업, by region and sex. 부산 has the most 농림어업 of all
    and 서울 the most 취업자: a ranking of the whole and of a detail differ."""
    out = []
    for y, year in enumerate(YEARS):
        for r, region in enumerate(REGIONS[1:], start=1):
            for s, sex in enumerate(SEXES[1:], start=1):
                d = [41 + 30 * (r == 2) + y + s, 400 + 7 * y - 30 * r + 11 * s, 900 + 13 * y - 70 * r - 17 * s]
                out += [_row(year, c, n, "천명", region, sex, v) for (c, n), v in zip(INDUSTRY, [sum(d), *d])]
    return _with_totals(out)


def status_rows():
    """취업자 (no mark) = 임금근로자-계 + 비임금근로자-계; each subtotal = its details, by region and sex."""
    out = []
    for y, year in enumerate(YEARS):
        for r, region in enumerate(REGIONS[1:], start=1):
            for s, sex in enumerate(SEXES[1:], start=1):
                w = [700 + 9 * y - 40 * r + 5 * s, 200 - 3 * y - 10 * r + 7 * s]
                n = [150 + y - 5 * r - 3 * s, 30 - y + s]
                vals = [sum(w) + sum(n), sum(w), *w, sum(n), *n]
                out += [_row(year, c, l, "천명", region, sex, v) for (c, l), v in zip(STATUS, vals)]
    return _with_totals(out)


def form_rows():
    """근로자-계 (a marked total) = 정규직 + 비정규직, by region and sex."""
    out = []
    for y, year in enumerate(YEARS):
        for r, region in enumerate(REGIONS[1:], start=1):
            for s, sex in enumerate(SEXES[1:], start=1):
                d = [500 + 6 * y - 20 * r + 31 * s, 300 - 2 * y - 9 * r + 47 * s]
                out += [_row(year, c, l, "명", region, sex, v) for (c, l), v in zip(FORM, [sum(d), *d])]
    return _with_totals(out)


def mixed_rows():
    """Three measures in three units; the first is named by the title but is no whole of the others."""
    out = []
    for y, year in enumerate(YEARS):
        for r, region in enumerate(REGIONS):
            vals = [900 - 100 * r + y, 7000 - 800 * r + 9 * y, 50000 - 3000 * r + 70 * y]
            out += [_row(year, c, n, u, region, None, v) for (c, n, u), v in zip(MIXED, vals)]
    return out


def _with_totals(rows):
    """Add the 계 sex and the 전국 region (sums), so every category has its marked total."""
    sums: dict[tuple, dict] = {}
    for row in rows:
        for region in (row["dim1_code"], "00"):
            for sex in (row["dim2_code"], "0"):
                key = (row["period"], row["item_id"], region, sex)
                if key not in sums:
                    sums[key] = {**row, "dim1_code": region, "dim1_label": dict(REGIONS)[region], "dim2_code": sex,
                                 "dim2_label": dict(SEXES)[sex], "value": 0}
                if (region, sex) != (row["dim1_code"], row["dim2_code"]):
                    sums[key]["value"] += row["value"]
    out = list(rows)
    for (_, _, region, sex), row in sums.items():
        if region == "00" or sex == "0":
            out.append({**row, "value_raw": str(row["value"])})
    return out


def rate_rows():
    """One item already in % (고용률) by region and sex."""
    return [_row(year, "R1", "고용률", "%", region, sex, round(60.1 + 0.7 * y - 1.3 * r + 4.1 * (s == 1), 1))
            for y, year in enumerate(YEARS) for r, region in enumerate(REGIONS) for s, sex in enumerate(SEXES)]


def catalog(title):
    return {"survey_name": "가상 조사", "organization": "가상 기관", "table_name": title, "period": "년",
            "url": "https://example.invalid/t"}


SPECS = {
    "FAKE11_IND": (industry_rows, catalog("가상 시도별 성별 산업별 취업자")),
    "FAKE11_STA": (status_rows, catalog("가상 종사상지위별 취업자(시도별)")),
    "FAKE11_FORM": (form_rows, catalog("가상 고용형태별 근로자")),
    "FAKE11_MIX": (mixed_rows, catalog("가상 시도별 사업체")),
    "FAKE11_RATE": (rate_rows, catalog("가상 시도별 성별 고용률")),
}
ROWS = {t: make() for t, (make, _) in SPECS.items()}


@pytest.fixture(scope="module")
def res(tmp_path_factory):
    folder = tmp_path_factory.mktemp("loop11w") / "data"
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


def value(tbl, item, region, year, sex="0"):
    return next(r["value"] for r in ROWS[tbl] if r["item_id"] == item and r["dim1_code"] == region
                and r["period"] == year and r.get("dim2_code", "0") == sex)


def ref(tbl, item, region, year, sex="0"):
    return f"{tbl}|{item}|{region}|{sex}|{year}" if tbl != "FAKE11_MIX" else f"{tbl}|{item}|{region}|{year}"


def cells(answer):
    return sorted(f.source_ref for f in answer if f.source_ref)


def computed(answer):
    return [f for f in answer if not f.source_ref]


# ---------------------------------------------------------------- A-1: the unmarked whole


@pytest.mark.parametrize("tbl, whole", [("FAKE11_IND", "E0"), ("FAKE11_STA", "S0")])
def test_unmarked_whole_is_the_total(res, tbl, whole):
    assert _total(res[tbl].index, "항목") == whole


@pytest.mark.parametrize("tbl, whole, region, early, late", [
    ("FAKE11_IND", "E0", "11", "2020", "2023"),
    ("FAKE11_STA", "S0", "21", "2021", "2023"),
])
def test_item_left_out_is_the_whole(res, tbl, whole, region, early, late):
    name = "서울" if region == "11" else "부산"
    facts, answer = ask(res, f"{early}년부터 {late}년까지 {name}은 몇 명 늘었어?", tbl)
    (change,) = computed(answer)
    assert change.value == pytest.approx(value(tbl, whole, region, late) - value(tbl, whole, region, early))
    assert refs_in(facts, change) == sorted([ref(tbl, whole, region, early), ref(tbl, whole, region, late)])


@pytest.mark.parametrize("tbl, whole, year", [("FAKE11_IND", "E0", "2022"), ("FAKE11_STA", "S0", "2023")])
def test_ranking_without_item_ranks_the_whole(res, tbl, whole, year):
    _, answer = ask(res, f"{year}년 가장 많은 시도는 어디야?", tbl)
    best = max(("11", "21", "22"), key=lambda r: value(tbl, whole, r, year))
    assert cells(answer) == [ref(tbl, whole, best, year)]


def test_neighbour_items_of_different_units_have_no_whole(res):
    index = res["FAKE11_MIX"].index
    assert _total(index, "항목") is None
    assert all(p is None for p in index.axes["항목"].parents.values())
    _, answer = ask(res, "2023년 서울의 종사자는 몇 명이야?", "FAKE11_MIX")
    assert cells(answer) == [ref("FAKE11_MIX", "M1", "11", "2023")]


def test_neighbour_marked_total_stays(res):
    assert _total(res["FAKE11_FORM"].index, "항목") == "K0"


# ---------------------------------------------------------------- A-2: the whole's name after a detail's


@pytest.mark.parametrize("question, item, region, year", [
    ("2023년 서울의 제조업 취업자는 몇 명이야?", "E2", "11", "2023"),
    ("2021년 부산 서비스업의 취업자 수 알려줘", "E3", "21", "2021"),
])
def test_detail_named_with_the_whole_after_it(res, question, item, region, year):
    _, answer = ask(res, question, "FAKE11_IND")
    assert cells(answer) == [ref("FAKE11_IND", item, region, year)]


@pytest.mark.parametrize("question, tbl, a, b, year", [
    ("2023년 서울의 제조업 취업자와 서비스업 취업자 차이는?", "FAKE11_IND", "E2", "E3", "2023"),
    ("2022년 서울 상용근로자 취업자는 임시근로자 취업자의 몇 배야?", "FAKE11_STA", "S2", "S3", "2022"),
])
def test_two_details_named_with_the_whole_after_each(res, question, tbl, a, b, year):
    facts, answer = ask(res, question, tbl)
    (calc,) = computed(answer)
    assert refs_in(facts, calc) == sorted([ref(tbl, a, "11", year), ref(tbl, b, "11", year)])
    va, vb = value(tbl, a, "11", year), value(tbl, b, "11", year)
    assert calc.value == pytest.approx(round(abs(va - vb), 2) if "차이" in question else round(va / vb, 2), abs=0.01)


def test_neighbour_whole_and_detail_both_asked(res):
    _, answer = ask(res, "2023년 서울의 취업자와 제조업 취업자는 각각 몇 명이야?", "FAKE11_IND")
    assert cells(answer) == sorted([ref("FAKE11_IND", "E0", "11", "2023"), ref("FAKE11_IND", "E2", "11", "2023")])


def test_neighbour_whole_as_the_subject_of_a_ratio(res):
    facts, answer = ask(res, "2023년 서울 취업자는 제조업의 몇 배야?", "FAKE11_IND")
    (ratio,) = computed(answer)
    assert ratio.value == pytest.approx(round(value("FAKE11_IND", "E0", "11", "2023") / value("FAKE11_IND", "E2", "11", "2023"), 2))


# ---------------------------------------------------------------- E-3: the share's whole


@pytest.mark.parametrize("question, part, region, sex", [
    ("2023년 서울 남자 중 상용근로자 비중은?", "S2", "11", "1"),
    ("2023년 부산 여자 가운데 자영업자가 차지하는 비중은?", "S5", "21", "2"),
])
def test_share_inside_other_categories_is_of_the_whole(res, question, part, region, sex):
    facts, answer = ask(res, question, "FAKE11_STA")
    (share,) = computed(answer)
    num, den = value("FAKE11_STA", part, region, "2023", sex), value("FAKE11_STA", "S0", region, "2023", sex)
    assert share.value == pytest.approx(round(num / den * 100, 2), abs=0.01)
    assert refs_in(facts, share) == sorted([ref("FAKE11_STA", part, region, "2023", sex),
                                            ref("FAKE11_STA", "S0", region, "2023", sex)])


@pytest.mark.parametrize("question, tbl, part, whole", [
    ("2023년 서울 취업자 중 상용근로자 비중은?", "FAKE11_STA", "S2", "S0"),
    ("2023년 서울 취업자 가운데 제조업이 차지하는 비중은?", "FAKE11_IND", "E2", "E0"),
])
def test_whole_named_before_the_scope_word_is_the_denominator(res, question, tbl, part, whole):
    facts, answer = ask(res, question, tbl)
    (share,) = computed(answer)
    assert refs_in(facts, share) == sorted([ref(tbl, part, "11", "2023"), ref(tbl, whole, "11", "2023")])
    assert share.value == pytest.approx(round(value(tbl, part, "11", "2023") / value(tbl, whole, "11", "2023") * 100, 2),
                                        abs=0.01)


@pytest.mark.parametrize("question, part, whole", [
    ("2023년 서울 임금근로자 중 상용근로자 비중은?", "S2", "S1"),
    ("2023년 서울 비임금근로자 중 자영업자 비중은?", "S5", "S4"),
])
def test_neighbour_subtotal_named_before_the_scope_word(res, question, part, whole):
    facts, answer = ask(res, question, "FAKE11_STA")
    (share,) = computed(answer)
    assert refs_in(facts, share) == sorted([ref("FAKE11_STA", part, "11", "2023"), ref("FAKE11_STA", whole, "11", "2023")])


# ---------------------------------------------------------------- E-2 and E-1: the scope and the part


@pytest.mark.parametrize("question, item, sex, region", [
    ("2023년 서울 정규직 중 여자 비중은?", "K1", "2", "11"),
    ("2023년 부산 비정규직 가운데 남자가 차지하는 비중은 얼마야?", "K2", "1", "21"),
    ("2023년 서울 정규직 중 여자는 몇 퍼센트야?", "K1", "2", "11"),          # E-1: no share word
    ("2023년 대구 비정규직 가운데 남자는 몇 %나 돼?", "K2", "1", "22"),
])
def test_part_after_the_scope(res, question, item, sex, region):
    facts, answer = ask(res, question, "FAKE11_FORM")
    (share,) = computed(answer)
    num, den = value("FAKE11_FORM", item, region, "2023", sex), value("FAKE11_FORM", item, region, "2023", "0")
    assert share.value == pytest.approx(round(num / den * 100, 2), abs=0.01)
    assert refs_in(facts, share) == sorted([ref("FAKE11_FORM", item, region, "2023", sex),
                                            ref("FAKE11_FORM", item, region, "2023", "0")])


def test_neighbour_item_share_without_a_scope(res):
    facts, answer = ask(res, "2023년 서울 여자 정규직 비중은?", "FAKE11_FORM")
    (share,) = computed(answer)
    assert refs_in(facts, share) == sorted([ref("FAKE11_FORM", "K1", "11", "2023", "2"),
                                            ref("FAKE11_FORM", "K0", "11", "2023", "2")])


def test_neighbour_group_holding_the_part(res):
    facts, answer = ask(res, "2023년 서울 근로자 중 정규직 비중은?", "FAKE11_FORM")
    (share,) = computed(answer)
    assert refs_in(facts, share) == sorted([ref("FAKE11_FORM", "K1", "11", "2023"), ref("FAKE11_FORM", "K0", "11", "2023")])


@pytest.mark.parametrize("question, region, sex", [
    ("2023년 서울 남자 중 고용률은 몇 %야?", "11", "1"),
    ("2022년 부산 여자 가운데 고용률은 몇 퍼센트야?", "21", "2"),
])
def test_neighbour_rate_cell_asked_inside_a_group(res, question, region, sex):
    year = question[:4]
    _, answer = ask(res, question, "FAKE11_RATE")
    assert cells(answer) == [ref("FAKE11_RATE", "R1", region, year, sex)]
    assert not computed(answer)


def test_neighbour_one_target_as_a_percent_of_another(res):
    facts, answer = ask(res, "2023년 서울 정규직 중 여자는 남자의 몇 퍼센트야?", "FAKE11_FORM")
    (level,) = computed(answer)
    women, men = value("FAKE11_FORM", "K1", "11", "2023", "2"), value("FAKE11_FORM", "K1", "11", "2023", "1")
    assert level.value == pytest.approx(round(women / men * 100, 2), abs=0.01)
    assert refs_in(facts, level) == sorted([ref("FAKE11_FORM", "K1", "11", "2023", "2"),
                                            ref("FAKE11_FORM", "K1", "11", "2023", "1")])
