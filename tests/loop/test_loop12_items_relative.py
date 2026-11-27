"""Loop 12 B-1, B-2 (analyze/multi_rules.py) and C-1 (judge/question.py head_parts).

- B-1: two tables over the same items (two or more), the metric differs by table. Items named one by one with a word
  asking for each ("남자와 여자 각각") repeat the calculation between the tables per item, as two regions do.
- B-2: one table's value as more or less than another's in percent ("몇 퍼센트 많아"): (subject / reference - 1) * 100,
  the reference being the table named before the word naming it (보다, 에 비해, 대비), read as one table reads it.
- C-1: item names made of an attribute and a head noun the names share ('개인사업체수', '법인사업체수' beside
  '총사업체수'; '맞벌이 가구' beside '전체 가구'): asked by the attribute (with or without the head's stem), and the
  whole by the head alone.

The tables are made up (가상 데이터); every test checks the values and the cells they come from.
"""

from __future__ import annotations

import re

import pytest

from conftest import write_table
from kosis_chat.adapters.parquet import load_table
from kosis_chat.calc import calculate
from kosis_chat.judge.question import head_parts_of
from kosis_chat.schemas.resources import load_resources

YEARS = ["2021", "2022", "2023"]
REGIONS = (("00", "전국"), ("11", "서울특별시"), ("26", "부산광역시"), ("27", "대구광역시"))
SEX = (("M", "남자"), ("F", "여자"))


def _row(year, item, name, unit, region, v, extra=None):
    out = {"period": year, "item_id": item, "item_name": name, "unit": unit, "dim1_name": "시도별",
           "dim1_code": region[0], "dim1_label": region[1], "value_raw": str(v), "value": v}
    out.update(extra or {})
    return out


def per_item(items, unit, base, step):
    """Each item's value by region and year; an item k adds 3k + something per region."""
    return [_row(year, ic, name, unit, region, base + step * y + 3 * k + 11 * r + k * r)
            for y, year in enumerate(YEARS) for r, region in enumerate(REGIONS) for k, (ic, name) in enumerate(items)]


def parts_with_whole(items, unit):
    """The first item is the sum of the others."""
    out = []
    for y, year in enumerate(YEARS):
        for r, region in enumerate(REGIONS):
            vals = [100 + 10 * k + 3 * y + 7 * r for k in range(1, len(items))]
            out += [_row(year, ic, name, unit, region, v) for (ic, name), v in zip(items, [sum(vals), *vals])]
    return out


def sex_by_item(items, unit):
    """Items in words whose attribute is also a name of another category (성별)."""
    out = []
    for y, year in enumerate(YEARS):
        for r, region in enumerate(REGIONS):
            for s, (sc, sl) in enumerate((("0", "계"), ("1", "남자"), ("2", "여자"))):
                for k, (ic, name) in enumerate(items):
                    out.append(_row(year, ic, name, unit, region, 50 + k + s + y + r,
                                    {"dim2_name": "성별", "dim2_code": sc, "dim2_label": sl}))
    return out


def catalog(title):
    return {"survey_name": "가상 조사", "organization": "가상 기관", "table_name": title, "period": "년",
            "url": "https://example.invalid/t"}


SPECS = {
    "FAKE12_FIRST": (lambda: per_item(SEX, "세", 30, 1), catalog("가상 시도별 평균 초혼연령")),
    "FAKE12_AGAIN": (lambda: per_item(SEX, "세", 45, 1), catalog("가상 시도별 평균 재혼연령")),
    "FAKE12_DEATH": (lambda: per_item((("P", "남자 사망자수"), ("Q", "여자 사망자수")), "명", 500, 7),
                     catalog("가상 시도별 성별 사망")),
    "FAKE12_BIRTH": (lambda: per_item((("X", "출생아수"),), "명", 900, 13), catalog("가상 시도별 출생")),
    "FAKE12_MOVE": (lambda: per_item((("I", "전입자수"),), "명", 700, -9), catalog("가상 시도별 전입")),
    "FAKE12_MARRY": (lambda: per_item((("W", "혼인건수"),), "건", 400, -5), catalog("가상 시도별 혼인")),
    "FAKE12_FIRM": (lambda: parts_with_whole((("T", "총사업체수"), ("A", "개인사업체수"), ("B", "법인사업체수"),
                                              ("C", "비법인단체사업체수")), "개"), catalog("가상 조직형태별 사업체")),
    "FAKE12_HOME": (lambda: parts_with_whole((("T", "전체 가구"), ("A", "맞벌이 가구"), ("B", "외벌이 가구")), "가구"),
                    catalog("가상 맞벌이 가구 현황")),
    "FAKE12_MOVERS": (lambda: parts_with_whole((("T", "사망자수"), ("A", "전입자수"), ("B", "전출자수")), "명"),
                      catalog("가상 인구 이동")),
    "FAKE12_SEXED": (lambda: sex_by_item((("A", "남자 취업자"), ("B", "여자 취업자"), ("T", "전체 취업자")), "천명"),
                     catalog("가상 성별 취업자")),
}
ROWS = {t: make() for t, (make, _) in SPECS.items()}


@pytest.fixture(scope="module")
def res(tmp_path_factory):
    folder = tmp_path_factory.mktemp("loop12i") / "data"
    return {t: load_resources(load_table(write_table(folder, t, ROWS[t], cat))) for t, (_, cat) in SPECS.items()}


def ask(res, question, *tbls):
    got = calculate(question, [res[t] for t in tbls])
    assert got.status == "ok", (got.status, got.reason, got.detail)
    facts = {f.id: f for f in got.facts}
    return facts, [facts[i] for i in got.answer]


def refs_in(facts, fact):
    out = []
    for i in re.findall(r"f\d+", fact.derivation or ""):
        f = facts[i]
        out += refs_in(facts, f) if f.derivation else [f.source_ref]
    return out


def value(tbl, item, region, year):
    return next(r["value"] for r in ROWS[tbl] if r["item_id"] == item and r["dim1_code"] == region
                and r["period"] == year and r.get("dim2_code", "0") == "0")


def ref(tbl, item, region, year):
    return f"{tbl}|{item}|{region}|{year}"


def computed(answer):
    return [f for f in answer if not f.source_ref]


def cells(answer):
    return sorted(f.source_ref for f in answer if f.source_ref)


# ---------------------------------------------------------------- B-1: per item across two tables


@pytest.mark.parametrize("question, region, year", [
    ("2023년 서울의 남자와 여자 각각 평균 재혼연령은 초혼연령보다 몇 세 높아?", "11", "2023"),
    ("2022년 부산 남자, 여자 저마다 재혼연령이 초혼연령보다 얼마나 많아?", "26", "2022"),
])
def test_difference_repeated_per_item_title_tables(res, question, region, year):
    facts, answer = ask(res, question, "FAKE12_FIRST", "FAKE12_AGAIN")
    calc = computed(answer)
    assert len(calc) == 2
    for got, item in zip(calc, ("M", "F")):  # question order
        assert got.value == pytest.approx(value("FAKE12_AGAIN", item, region, year)
                                          - value("FAKE12_FIRST", item, region, year), abs=0.01)
        assert refs_in(facts, got) == [ref("FAKE12_AGAIN", item, region, year), ref("FAKE12_FIRST", item, region, year)]
    assert len(cells(answer)) == 4


def test_difference_repeated_per_item_against_a_one_item_table(res):
    """Only one table has the items: its items are each compared with the other table's one value."""
    facts, answer = ask(res, "2021년 대구의 남자 사망자수와 여자 사망자수는 각각 출생아 수보다 몇 명 적어?",
                        "FAKE12_DEATH", "FAKE12_BIRTH")
    calc = computed(answer)
    assert len(calc) == 2
    for got, item in zip(calc, ("P", "Q")):
        assert got.value == pytest.approx(value("FAKE12_BIRTH", "X", "27", "2021")
                                          - value("FAKE12_DEATH", item, "27", "2021"), abs=0.01)
        assert sorted(refs_in(facts, got)) == sorted([ref("FAKE12_BIRTH", "X", "27", "2021"),
                                                      ref("FAKE12_DEATH", item, "27", "2021")])


def test_values_listed_per_item_and_table(res):
    _, answer = ask(res, "2023년 서울 남자와 여자의 초혼연령, 재혼연령을 각각 알려줘", "FAKE12_FIRST", "FAKE12_AGAIN")
    assert cells(answer) == sorted(ref(t, i, "11", "2023") for t in ("FAKE12_FIRST", "FAKE12_AGAIN") for i in ("M", "F"))
    assert not computed(answer)


def test_neighbour_one_item_each_table(res):
    facts, answer = ask(res, "2023년 서울의 남자 평균 재혼연령은 초혼연령보다 몇 세 높아?", "FAKE12_FIRST", "FAKE12_AGAIN")
    (got,) = computed(answer)
    assert refs_in(facts, got) == [ref("FAKE12_AGAIN", "M", "11", "2023"), ref("FAKE12_FIRST", "M", "11", "2023")]


def test_neighbour_two_items_without_a_word_for_each_stays_open(res):
    got = calculate("2023년 서울 남자와 여자의 초혼연령, 재혼연령 알려줘", [res["FAKE12_FIRST"], res["FAKE12_AGAIN"]])
    assert got.status == "no_plan"


# ---------------------------------------------------------------- B-2: a relative difference across two tables


@pytest.mark.parametrize("question, tbls, subject, reference, region, year", [
    ("2023년 서울의 출생아 수는 전입자 수보다 몇 퍼센트 많아?", ("FAKE12_BIRTH", "FAKE12_MOVE"),
     ("FAKE12_BIRTH", "X"), ("FAKE12_MOVE", "I"), "11", "2023"),
    ("2022년 부산 전입자는 출생아에 비해 몇 % 적어?", ("FAKE12_BIRTH", "FAKE12_MOVE"),
     ("FAKE12_MOVE", "I"), ("FAKE12_BIRTH", "X"), "26", "2022"),
    ("2021년 대구 전입자 수 대비 출생아 수는 몇 퍼센트 더 많아?", ("FAKE12_MOVE", "FAKE12_BIRTH"),
     ("FAKE12_BIRTH", "X"), ("FAKE12_MOVE", "I"), "27", "2021"),
])
def test_relative_difference_between_tables(res, question, tbls, subject, reference, region, year):
    facts, answer = ask(res, question, *tbls)
    (got,) = computed(answer)
    a, b = value(*subject, region, year), value(*reference, region, year)
    assert got.value == pytest.approx(round((a / b - 1) * 100, 2), abs=0.01)
    assert refs_in(facts, got) == [ref(*subject, region, year), ref(*reference, region, year)]
    assert len(cells(answer)) == 2


def test_neighbour_level_stays_a_share_of_the_other(res):
    facts, answer = ask(res, "2023년 서울의 전입자 수는 출생아 수의 몇 퍼센트야?", "FAKE12_BIRTH", "FAKE12_MOVE")
    (got,) = computed(answer)
    a, b = value("FAKE12_MOVE", "I", "11", "2023"), value("FAKE12_BIRTH", "X", "11", "2023")
    assert got.value == pytest.approx(round(a / b * 100, 2), abs=0.01)


def test_neighbour_rate_of_change_per_table(res):
    facts, answer = ask(res, "2021년부터 2023년까지 서울 출생아 수와 전입자 수는 각각 몇 퍼센트 늘었어?",
                        "FAKE12_BIRTH", "FAKE12_MOVE")
    rates = computed(answer)
    assert len(rates) == 2
    a, b = value("FAKE12_BIRTH", "X", "11", "2021"), value("FAKE12_BIRTH", "X", "11", "2023")
    assert rates[0].value == pytest.approx(round((b / a - 1) * 100, 2), abs=0.01)


def test_neighbour_units_differ_still_refused(res):
    got = calculate("2023년 서울 출생아 수는 혼인 건수보다 몇 퍼센트 많아?", [res["FAKE12_BIRTH"], res["FAKE12_MARRY"]])
    assert got.status == "refused"


# ---------------------------------------------------------------- C-1: attribute + shared head noun


@pytest.mark.parametrize("question, tbl, item, region, year", [
    ("2023년 서울 개인 사업체는 몇 개야?", "FAKE12_FIRM", "A", "11", "2023"),      # attribute + the head's stem
    ("2022년 부산 법인은 몇 곳이야?", "FAKE12_FIRM", "B", "26", "2022"),           # the attribute alone
    ("2021년 대구 사업체 수는?", "FAKE12_FIRM", "T", "27", "2021"),               # the whole by the head
    ("2021년 부산 맞벌이 가구는 몇 가구야?", "FAKE12_HOME", "A", "26", "2021"),     # a name in words
    ("2023년 서울 가구는 몇 가구야?", "FAKE12_HOME", "T", "11", "2023"),
])
def test_item_named_by_its_attribute_or_head(res, question, tbl, item, region, year):
    _, answer = ask(res, question, tbl)
    assert cells(answer) == [ref(tbl, item, region, year)]
    assert answer[0].value == value(tbl, item, region, year)


@pytest.mark.parametrize("question, region, year", [
    ("2023년 서울 개인과 법인의 사업체 수 차이는?", "11", "2023"),
    ("2022년 대구 개인 사업체와 법인 사업체 차이는 얼마야?", "27", "2022"),
])
def test_two_attributes_compared(res, question, region, year):
    facts, answer = ask(res, question, "FAKE12_FIRM")
    (got,) = computed(answer)
    a, b = value("FAKE12_FIRM", "A", region, year), value("FAKE12_FIRM", "B", region, year)
    assert abs(got.value) == pytest.approx(abs(a - b), abs=0.01)
    assert sorted(refs_in(facts, got)) == sorted([ref("FAKE12_FIRM", "A", region, year),
                                                  ref("FAKE12_FIRM", "B", region, year)])


def test_neighbour_names_sharing_only_a_person_ending_are_not_split(res):
    assert head_parts_of(res["FAKE12_MOVERS"].index).get("항목", {}) == {}
    _, answer = ask(res, "2023년 서울 전입자 수는?", "FAKE12_MOVERS")
    assert cells(answer) == [ref("FAKE12_MOVERS", "A", "11", "2023")]


def test_neighbour_attribute_another_category_names_is_not_split(res):
    assert head_parts_of(res["FAKE12_SEXED"].index).get("항목", {}) == {}


def test_neighbour_full_name_still_matches(res):
    _, answer = ask(res, "2023년 서울 개인사업체수는?", "FAKE12_FIRM")
    assert cells(answer) == [ref("FAKE12_FIRM", "A", "11", "2023")]
