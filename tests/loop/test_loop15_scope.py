"""Loop 15 C (analyze/rules.py _within, _scoped_percent; judge/rules.py _named_apart; shared/patterns.py within_split):

- C-1: a question with one word of looking inside a group (중, 가운데, 안에서) and a name after it: the names before
  the word are where the share is taken (another category: fixed; a parent of the same category: the whole), the
  names after it are the parts, whatever words come between the names and the word (a person noun, a describing
  clause, a group noun, a word of the whole).
- C-2: a part of such a group asked by the noun 비율 is a share (cells not in %, 비율 no part of a name).

Neighbours: the name right before the word (as before), no name after the word, a difference with the word, a
ranking inside a parent (pool as before), a ratio of two targets, and a cell already in %.
The tables are made up (가상 데이터); every test checks the value and the cells it is computed from.
"""

from __future__ import annotations

import re

import pytest

from conftest import write_table
from kosis_chat.adapters.parquet import load_table
from kosis_chat.calc import calculate
from kosis_chat.schemas.resources import load_resources

YEARS = ("2022", "2023")
REGIONS = (("00", "전국"), ("11", "서울특별시"), ("11010", "종로구"), ("11020", "성북구"), ("21", "부산광역시"),
           ("21010", "해운대구"), ("21020", "사하구"), ("21030", "북구"))
SEXES = (("0", "계"), ("1", "남자"), ("2", "여자"))
KINDS = (("0", "계"), ("1", "상용근로자"), ("2", "임시근로자"), ("3", "일용근로자"))


def catalog(title):
    return {"survey_name": "가상 고용조사", "organization": "가상 기관", "table_name": title, "period": "년",
            "url": "https://example.invalid/t"}


def base(region: int, sex: int, kind: int, y: int) -> int:
    return 50 + 37 * region + 13 * sex + 7 * kind + 3 * y + (region * kind) % 5


def workers_rows(item="E1", name="취업자", unit="명"):
    # sums are kept: a total is the sum of its parts on every category (regions: a province of its districts)
    leaves = {}
    for y, year in enumerate(YEARS):
        for r, (rc, _) in enumerate(REGIONS):
            if len(rc) != 5:
                continue
            for s in (1, 2):
                for k in (1, 2, 3):
                    leaves[(year, rc, s, k)] = base(r, s, k, y)
    rows = []
    for year in YEARS:
        for rc, rl in REGIONS:
            for sc, sl in SEXES:
                for kc, kl in KINDS:
                    v = sum(val for (yy, lc, s, k), val in leaves.items()
                            if yy == year and (rc == "00" or lc.startswith(rc))
                            and (sc == "0" or s == int(sc)) and (kc == "0" or k == int(kc)))
                    rows.append({"period": year, "item_id": item, "item_name": name, "unit": unit,
                                 "dim1_name": "행정구역별", "dim1_code": rc, "dim1_label": rl,
                                 "dim2_name": "성별", "dim2_code": sc, "dim2_label": sl,
                                 "dim3_name": "고용형태별", "dim3_code": kc, "dim3_label": kl,
                                 "value_raw": str(v), "value": v})
    return rows


def rate_rows():
    return [{**r, "item_id": "R1", "item_name": "고용형태 비율", "unit": "%", "value": round(r["value"] / 7.0, 1),
             "value_raw": str(round(r["value"] / 7.0, 1))} for r in workers_rows()]


SPECS = {
    "FAKE15_WORK": (workers_rows, catalog("가상 시군구별 성별 고용형태별 취업자")),
    "FAKE15_RATE": (rate_rows, catalog("가상 시군구별 성별 고용형태 비율")),
}
ROWS = {t: make() for t, (make, _) in SPECS.items()}
T = "FAKE15_WORK"


@pytest.fixture(scope="module")
def res(tmp_path_factory):
    folder = tmp_path_factory.mktemp("loop15c") / "data"
    return {t: load_resources(load_table(write_table(folder, t, ROWS[t], cat))) for t, (_, cat) in SPECS.items()}


def ask(res, question, tbl=T):
    got = calculate(question, [res[tbl]])
    assert got.status == "ok", (got.status, got.reason, got.detail)
    facts = {f.id: f for f in got.facts}
    return facts, [facts[i] for i in got.answer]


def cells(answer):
    return sorted(f.source_ref for f in answer if f.source_ref)


def computed(answer):
    return [f for f in answer if not f.source_ref]


def refs_in(facts, fact):
    return sorted(facts[i].source_ref for i in re.findall(r"f\d+", fact.derivation or ""))


def ref(region, sex, kind, year="2023", tbl=T, item="E1"):
    return f"{tbl}|{item}|{region}|{sex}|{kind}|{year}"


def value(region, sex, kind, year="2023"):
    for r in ROWS[T]:
        if (r["period"], r["dim1_code"], r["dim2_code"], r["dim3_code"]) == (year, region, sex, kind):
            return r["value"]
    raise KeyError((region, sex, kind, year))


def assert_share(facts, answer, part, whole):
    (got,) = computed(answer)
    assert got.value == pytest.approx(round(100 * value(*part) / value(*whole), 1), abs=0.051)
    assert refs_in(facts, got) == sorted([ref(*part), ref(*whole)])


# ---------------------------------------------------------------- C-1


@pytest.mark.parametrize("question, part, whole", [
    # a name, a person noun, a describing clause and a group noun before the word
    ("2023년 여자 근로자로 일하고 있는 사람들 중 상용근로자 비중은?", ("00", "2", "1"), ("00", "2", "0")),
    ("2022년 남자 취업자로 일하는 사람 가운데 일용근로자가 차지하는 비중은?", ("00", "1", "3", "2022"), ("00", "1", "0", "2022")),
    # a name, …인, a group noun
    ("2023년 서울에 사는 여자인 취업자 집단 중 임시근로자 구성비는?", ("11", "2", "2"), ("11", "2", "0")),
])
def test_c1_names_before_the_word_fix_the_share(res, question, part, whole):
    facts, answer = ask(res, question)
    assert_share(facts, answer, part, whole)


@pytest.mark.parametrize("question, part, whole", [
    ("2023년 여자 근로자로 일하는 사람들 중 상용근로자는 몇 퍼센트야?", ("00", "2", "1"), ("00", "2", "0")),
    ("2022년 남자인 사람들 가운데 임시근로자는 몇 %야?", ("00", "1", "2", "2022"), ("00", "1", "0", "2022")),
])
def test_c1_percent_only(res, question, part, whole):
    facts, answer = ask(res, question)
    assert_share(facts, answer, part, whole)


@pytest.mark.parametrize("question, part, whole", [
    ("2023년 부산 전체 취업자 중 해운대구 비중은?", ("21010", "0", "0"), ("21", "0", "0")),
    ("2022년 서울 지역 전체 근로자 가운데 성북구가 차지하는 비중은?", ("11020", "0", "0", "2022"), ("11", "0", "0", "2022")),
])
def test_c1_parent_region_and_a_word_of_the_whole(res, question, part, whole):
    facts, answer = ask(res, question)
    assert_share(facts, answer, part, whole)


# ---------------------------------------------------------------- C-2


@pytest.mark.parametrize("question, part, whole", [
    ("2023년 여자인 사람들 중 일용근로자의 비율은?", ("00", "2", "3"), ("00", "2", "0")),
    ("2022년 부산 전체 취업자 가운데 사하구의 비율은 얼마야?", ("21020", "0", "0", "2022"), ("21", "0", "0", "2022")),
    ("2023년 여자 중 상용근로자 비율은?", ("00", "2", "1"), ("00", "2", "0")),   # the name right before the word
])
def test_c2_ratio_noun_of_a_scoped_part(res, question, part, whole):
    facts, answer = ask(res, question)
    assert_share(facts, answer, part, whole)


# ---------------------------------------------------------------- neighbours


def test_neighbour_name_right_before_the_word(res):
    facts, answer = ask(res, "2023년 여자 중 상용근로자 비중은?")
    assert_share(facts, answer, ("00", "2", "1"), ("00", "2", "0"))


def test_neighbour_no_name_after_the_word(res):
    _, answer = ask(res, "2023년 서울 여자 상용근로자 수는?")
    assert cells(answer) == [ref("11", "2", "1")]


def test_neighbour_difference_with_the_word(res):
    facts, answer = ask(res, "2023년 여자 근로자로 일하는 사람 가운데 상용근로자와 임시근로자의 차이는?")
    (got,) = computed(answer)
    assert abs(got.value) == value("00", "2", "1") - value("00", "2", "2") or abs(got.value) == abs(
        value("00", "2", "1") - value("00", "2", "2"))
    assert refs_in(facts, got) == sorted([ref("00", "2", "1"), ref("00", "2", "2")])


def test_neighbour_ranking_inside_a_parent(res):
    _, answer = ask(res, "2023년 부산 중 취업자가 가장 많은 구는?")
    best = max(("21010", "21020", "21030"), key=lambda c: value(c, "0", "0"))
    assert ref(best, "0", "0") in cells(answer)


def test_neighbour_ratio_of_two_targets(res):
    facts, answer = ask(res, "2023년 여자 중 상용근로자는 임시근로자의 몇 배야?")
    (got,) = computed(answer)
    assert got.value == pytest.approx(round(value("00", "2", "1") / value("00", "2", "2"), 2), abs=0.01)


def test_neighbour_cell_already_in_percent(res):
    _, answer = ask(res, "2023년 여자인 사람들 중 상용근로자의 고용형태 비율은?", "FAKE15_RATE")
    assert cells(answer) == [ref("00", "2", "1", tbl="FAKE15_RATE", item="R1")]
