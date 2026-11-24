"""Loop 14 A (judge/question.py member_forms, match_labels; shared/labels.py MEMBER_TAILS, SEX_NAMES; analyze/roles.py
reference_index said=):

- A-1: a name of a place people attend is also said by its members: the tail's last part changed to the member word
  ('중학교' as '중학생', '대학교' as '대학생'), on a table where the names are items and one where they are a category,
  and on two tables (students and teachers).
- A-2: a sex's first character before a member noun of the table ('남학생', '여중학생').
- A-3: a sex's first character, the name's first character and the member tail's last character ('남고생', '여중생').
- The reference of "의 몇 배" is read on the form the question named it by (a ratio of two names said by members).

Neighbours: a member word holding another name of the table stays that name ('대학생' with '대학' in the table); a
member word two names give is neither's; words that start with a sex's first character but are no member noun ('남쪽',
'여행', a region '전남' before 학생) are no sex; the names said as they are read as before.
The tables are made up (가상 데이터); every test checks the values and the cells they come from.
"""

from __future__ import annotations

import re

import pytest

from conftest import write_table
from kosis_chat.adapters.parquet import load_table
from kosis_chat.calc import calculate
from kosis_chat.schemas.resources import load_resources

YEARS = ("2022", "2023")
SCHOOLS = (("S0", "계"), ("S1", "유치원"), ("S2", "초등학교"), ("S3", "중학교"), ("S4", "고등학교(일반계 포함)"),
           ("S5", "전문대학"), ("S6", "대학교"), ("S7", "대학원"))
SEXES = (("0", "계"), ("1", "남자"), ("2", "여자"))
# a table whose names include '대학' beside '대학교': '대학생' stays '대학' (the shorter name, as before)
NEAR = (("S0", "계"), ("S3", "중학교"), ("S5", "대학"), ("S6", "대학교"))


def catalog(title):
    return {"survey_name": "가상 교육조사", "organization": "가상 기관", "table_name": title, "period": "년",
            "url": "https://example.invalid/t"}


def by_item():
    rows = []
    for y, year in enumerate(YEARS):
        for i, (code, label) in enumerate(SCHOOLS[1:], start=1):
            men, women = 1000 * i + 10 * y + 3, 1000 * i + 10 * y + 7 * i
            for sex, v in (("0", men + women), ("1", men), ("2", women)):
                rows.append({"period": year, "item_id": code, "item_name": label, "unit": "명", "dim1_name": "성별",
                             "dim1_code": sex, "dim1_label": dict(SEXES)[sex], "value_raw": str(v), "value": v})
    return rows


def by_category(schools, item="학생수", base=700):
    rows = []
    for y, year in enumerate(YEARS):
        for i, (code, label) in enumerate(schools):
            for s, (sex, sex_label) in enumerate(SEXES):
                v = base * (i + 1) + 5 * y + 2 * s + 1
                rows.append({"period": year, "item_id": "T1", "item_name": item, "unit": "명", "dim1_name": "학교급별",
                             "dim1_code": code, "dim1_label": label, "dim2_name": "성별", "dim2_code": sex,
                             "dim2_label": sex_label, "value_raw": str(v), "value": v})
    return rows


def teachers():
    return [{"period": year, "item_id": "T9", "item_name": "교원수", "unit": "명", "dim1_name": "학교급별",
             "dim1_code": code, "dim1_label": label, "value_raw": str(90 * i + 3 * y + 7), "value": 90 * i + 3 * y + 7}
            for y, year in enumerate(YEARS) for i, (code, label) in enumerate(SCHOOLS)]


SPECS = {
    "FAKE14_ITEM": (by_item, catalog("가상 학교급별 성별 학생 수")),
    "FAKE14_CAT": (lambda: by_category(SCHOOLS), catalog("가상 학교급별 성별 학생수")),
    "FAKE14_NEAR": (lambda: by_category(NEAR, base=300), catalog("가상 학교급별 성별 재학생수")),
    "FAKE14_TEACH": (teachers, catalog("가상 학교급별 교원 수")),
}
ROWS = {t: make() for t, (make, _) in SPECS.items()}


@pytest.fixture(scope="module")
def res(tmp_path_factory):
    folder = tmp_path_factory.mktemp("loop14a") / "data"
    return {t: load_resources(load_table(write_table(folder, t, ROWS[t], cat))) for t, (_, cat) in SPECS.items()}


def ask(res, question, *tbls):
    got = calculate(question, [res[t] for t in tbls])
    assert got.status == "ok", (got.status, got.reason, got.detail)
    facts = {f.id: f for f in got.facts}
    return facts, [facts[i] for i in got.answer]


def cells(answer):
    return sorted(f.source_ref for f in answer if f.source_ref)


def computed(answer):
    return [f for f in answer if not f.source_ref]


def refs_in(facts, fact):
    return [facts[i].source_ref for i in re.findall(r"f\d+", fact.derivation or "")]


def value(tbl, school, sex="0", year="2023"):
    for r in ROWS[tbl]:
        if r["period"] != year:
            continue
        if tbl == "FAKE14_ITEM" and r["item_id"] == school and r["dim1_code"] == sex:
            return r["value"]
        if tbl == "FAKE14_TEACH" and r["dim1_code"] == school:
            return r["value"]
        if tbl in ("FAKE14_CAT", "FAKE14_NEAR") and r["dim1_code"] == school and r["dim2_code"] == sex:
            return r["value"]
    raise KeyError(school)


def ref(tbl, school, sex="0", year="2023"):
    if tbl == "FAKE14_ITEM":
        return f"{tbl}|{school}|{sex}|{year}"
    if tbl == "FAKE14_TEACH":
        return f"{tbl}|T9|{school}|{year}"
    return f"{tbl}|T1|{school}|{sex}|{year}"


# ---------------------------------------------------------------- A-1: the members of a place


@pytest.mark.parametrize("question, tbl, school, year", [
    ("2023년 중학생은 몇 명이야?", "FAKE14_ITEM", "S3", "2023"),
    ("2022년 우리나라 고등학생 수 알려줘", "FAKE14_ITEM", "S4", "2022"),   # a name with a note in brackets
    ("2023년 중학생은 몇 명이야?", "FAKE14_CAT", "S3", "2023"),
    ("2022년 대학생 수는?", "FAKE14_CAT", "S6", "2022"),
])
def test_a_name_said_by_its_members(res, question, tbl, school, year):
    _, answer = ask(res, question, tbl)
    assert cells(answer) == [ref(tbl, school, year=year)]
    assert answer[0].value == value(tbl, school, year=year)


@pytest.mark.parametrize("tbl", ["FAKE14_ITEM", "FAKE14_CAT"])
def test_two_names_said_by_their_members_listed(res, tbl):
    _, answer = ask(res, "2023년 초등학생과 고등학생 수를 알려줘", tbl)
    assert cells(answer) == sorted([ref(tbl, "S2"), ref(tbl, "S4")])


@pytest.mark.parametrize("question, tbl, top, bottom", [
    ("2023년 초등학생 수는 중학생 수의 몇 배야?", "FAKE14_ITEM", "S2", "S3"),
    ("2023년 대학생은 고등학생의 몇 배야?", "FAKE14_CAT", "S6", "S4"),
])
def test_ratio_of_two_names_said_by_members_reads_the_reference(res, question, tbl, top, bottom):
    facts, answer = ask(res, question, tbl)
    (got,) = computed(answer)
    assert got.value == pytest.approx(round(value(tbl, top) / value(tbl, bottom), 2), abs=0.01)
    assert refs_in(facts, got) == [ref(tbl, top), ref(tbl, bottom)]


@pytest.mark.parametrize("question, school", [
    ("2023년 고등학생 수는 고등학교 교원 수의 몇 배야?", "S4"),
    ("2023년 중학교 교원 1명당 중학생 수는?", "S3"),
])
def test_two_tables_the_members_on_the_table_of_items(res, question, school):
    facts, answer = ask(res, question, "FAKE14_ITEM", "FAKE14_TEACH")
    (got,) = computed(answer)
    assert got.value == pytest.approx(round(value("FAKE14_ITEM", school) / value("FAKE14_TEACH", school), 2), abs=0.01)
    assert refs_in(facts, got) == [ref("FAKE14_ITEM", school), ref("FAKE14_TEACH", school)]


# ---------------------------------------------------------------- A-2, A-3: a sex before the members


@pytest.mark.parametrize("question, tbl, school", [
    ("2023년 고등학교 남학생과 여학생 수는?", "FAKE14_CAT", "S4"),
    ("2023년 중학생 가운데 남학생과 여학생은 각각 몇 명이야?", "FAKE14_ITEM", "S3"),
    ("2023년 남고생과 여고생은 몇 명이야?", "FAKE14_CAT", "S4"),
    ("2023년 남중생 여중생 수 알려줘", "FAKE14_ITEM", "S3"),
])
def test_both_sexes_said_before_a_member_noun(res, question, tbl, school):
    _, answer = ask(res, question, tbl)
    assert cells(answer) == sorted([ref(tbl, school, "1"), ref(tbl, school, "2")])
    assert sorted(f.value for f in answer) == sorted([value(tbl, school, "1"), value(tbl, school, "2")])


def test_one_sex_shortened_with_the_place(res):
    _, answer = ask(res, "2022년 여중생은 몇 명?", "FAKE14_CAT")
    assert cells(answer) == [ref("FAKE14_CAT", "S3", "2", "2022")]


# ---------------------------------------------------------------- neighbours


def test_neighbour_member_word_holding_another_name_stays_that_name(res):
    _, answer = ask(res, "2023년 대학생은 몇 명이야?", "FAKE14_NEAR")
    assert cells(answer) == [ref("FAKE14_NEAR", "S5")]


@pytest.mark.parametrize("question", [
    "2023년 남쪽 지역 학교 학생 수는?",
    "2023년 여행을 간 학생 수는?",
])
def test_neighbour_words_starting_like_a_sex_are_no_sex(res, question):
    _, answer = ask(res, question, "FAKE14_CAT")
    assert cells(answer) == [ref("FAKE14_CAT", "S0")]


def test_neighbour_names_said_as_they_are(res):
    _, answer = ask(res, "2023년 고등학교 여자 학생 수는?", "FAKE14_CAT")
    assert cells(answer) == [ref("FAKE14_CAT", "S4", "2")]
    _, answer = ask(res, "2023년 대학원생은 몇 명이야?", "FAKE14_CAT")
    assert cells(answer) == [ref("FAKE14_CAT", "S7")]
