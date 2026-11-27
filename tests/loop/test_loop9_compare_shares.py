"""Loop 9 (analyze/rules.py, analyze/blocks.py, analyze/multi_rules.py, shared/patterns.py, compute/plan_engine.py):

- C-1: one target's difference from another as a percent of the other ("A는 B보다 몇 퍼센트 많아/적어"):
  (subject - reference) / reference * 100, also when a word of the table's kind of target sits between the reference's
  name and the comparing word ("부산 사업체보다").
- D-1: two parts' shares of the same whole compared: each share and their difference in percent points, computed from
  the shares before rounding.
- E-1: a division of two items of different units in one table (a ratio said with its verb, a value per unit of
  another) is computed, not refused; a difference of them still is refused.
- B-2: a word of the whole right before 몇 + a counting word ("모두 몇 명") adds two targets or more named one by one
  (one table), or the two tables' values (two tables); one target, or a request for the values, stays as it was.
- B-1: a label that holds a word of a comparison ('남녀 임금차이') asked alone is its value.

The tables are made up (가상 데이터). Every test checks the answer's values and the cells they were computed from.
"""

from __future__ import annotations

import re

import pytest

from conftest import write_table
from kosis_chat.adapters.parquet import load_table
from kosis_chat.calc import calculate
from kosis_chat.schemas.resources import load_resources

REGIONS = (("00", "전국"), ("11", "서울특별시"), ("21", "부산광역시"), ("22", "대구광역시"))
YEARS = [str(y) for y in range(2018, 2024)]
SIZES = (("0", "계"), ("1", "1인"), ("2", "2인"), ("3", "3인"), ("4", "4인이상"))


def _row(year, item, name, unit, code, label, v, **dims):
    return {"period": year, "item_id": item, "item_name": name, "unit": unit, "dim1_name": "시도별", "dim1_code": code,
            "dim1_label": label, "value_raw": str(v), "value": v, **dims}


def firm_rows():
    """One item: 사업체수 (개) by region."""
    out = []
    for y, year in enumerate(YEARS):
        vals = {"11": 41237 + 613 * y, "21": 17389 + 211 * y, "22": 12011 + 97 * y}
        vals["00"] = sum(vals.values()) + 30011 + 500 * y
        out += [_row(year, "F1", "사업체수", "개", c, l, vals[c]) for c, l in REGIONS]
    return out


def work_rows():
    """Two items of different units: 사업체수 (개) and 종사자수 (명)."""
    out = []
    for y, year in enumerate(YEARS):
        for r, (code, label) in enumerate(REGIONS):
            firms = 9000 - 1700 * r + 37 * y if code != "00" else 30000 + 111 * y
            workers = 77123 - 13001 * r + 451 * y if code != "00" else 260000 + 1999 * y
            out += [_row(year, "W1", "사업체수", "개", code, label, firms),
                    _row(year, "W2", "종사자수", "명", code, label, workers)]
    return out


def house_rows():
    """가구수 by region and household size; the sizes add up to 계."""
    out = []
    for y, year in enumerate(YEARS):
        for r, (code, label) in enumerate(REGIONS):
            sizes = [3331 + 71 * y - 300 * r, 2777 + 13 * y - 211 * r, 1999 - 7 * y - 100 * r, 1501 - 29 * y - 50 * r]
            for (s_code, s_label), v in zip(SIZES, [sum(sizes), *sizes]):
                out.append(_row(year, "H1", "가구수", "가구", code, label, v, dim2_name="가구원수별",
                                dim2_code=s_code, dim2_label=s_label))
    return out


def wage_rows():
    """Two items in 만원: 월평균임금 and 남녀 임금차이 (a label holding a word of a comparison)."""
    out = []
    for y, year in enumerate(YEARS):
        for r, (code, label) in enumerate(REGIONS):
            out += [_row(year, "G1", "월평균임금", "만원", code, label, 301 + 9 * y - 13 * r),
                    _row(year, "G2", "남녀 임금차이", "만원", code, label, 97 - 2 * y + 5 * r)]
    return out


def birth_rows():
    return [_row(year, "B1", "출생아수", "명", c, l, 21000 - 900 * y - 5000 * r)
            for y, year in enumerate(YEARS) for r, (c, l) in enumerate(REGIONS)]


def death_rows():
    return [_row(year, "D1", "사망자수", "명", c, l, 15000 + 410 * y - 3300 * r)
            for y, year in enumerate(YEARS) for r, (c, l) in enumerate(REGIONS)]


def catalog(title):
    return {"survey_name": "가상 조사", "organization": "가상 기관", "table_name": title, "period": "년",
            "url": "https://example.invalid/t"}


SPECS = {
    "FAKE9_FIRM": (firm_rows, catalog("가상 시도별 사업체 현황")),
    "FAKE9_WORK": (work_rows, catalog("가상 시도별 사업체와 종사자")),
    "FAKE9_HOUSE": (house_rows, catalog("가상 시도별 가구원수별 가구")),
    "FAKE9_WAGE": (wage_rows, catalog("가상 시도별 임금")),
    "FAKE9_BIRTH": (birth_rows, catalog("가상 시도별 출생")),
    "FAKE9_DEATH": (death_rows, catalog("가상 시도별 사망")),
}


@pytest.fixture(scope="module")
def res(tmp_path_factory):
    folder = tmp_path_factory.mktemp("loop9") / "data"
    return {t: load_resources(load_table(write_table(folder, t, make(), cat))) for t, (make, cat) in SPECS.items()}


def ask(res, question, *tbl_ids):
    got = calculate(question, [res[t] for t in tbl_ids])
    assert got.status == "ok", (got.status, got.reason, got.detail)
    facts = {f.id: f for f in got.facts}
    return facts, [facts[i] for i in got.answer]


def refs_in(facts, fact):
    out = []
    for i in re.findall(r"f\d+", fact.derivation or ""):
        f = facts[i]
        out += refs_in(facts, f) if f.derivation else [f.source_ref]
    return sorted(out)


def value(tbl, item, region, year, size=None):
    return next(r["value"] for r in SPECS[tbl][0]() if r["item_id"] == item and r["dim1_code"] == region
                and r["period"] == year and (size is None or r.get("dim2_code") == size))


def ref(tbl, item, region, year, size=None):
    dims = f"{region}|{size}" if size is not None else region
    return f"{tbl}|{item}|{dims}|{year}"


def cells(answer):
    return sorted(f.source_ref for f in answer if f.source_ref)


def derived(answer):
    return [f for f in answer if f.derivation]


# ---------------------------------------------------------------- C-1: a relative difference in percent


@pytest.mark.parametrize("question, table, item, subject, reference, year", [
    ("2021년 서울 사업체는 부산 사업체보다 몇 퍼센트 많아?", "FAKE9_FIRM", "F1", "11", "21", "2021"),
    ("2023년 대구 사업체수는 서울에 비해 몇 % 적어?", "FAKE9_FIRM", "F1", "22", "11", "2023"),
    ("2019년 부산 종사자수는 대구보다 몇 퍼센트 더 많아?", "FAKE9_WORK", "W2", "21", "22", "2019"),
])
def test_relative_difference_in_percent(res, question, table, item, subject, reference, year):
    facts, answer = ask(res, question, table)
    (got,) = derived(answer)
    a, b = value(table, item, subject, year), value(table, item, reference, year)
    assert got.value == pytest.approx(round((a / b - 1) * 100, 2), abs=0.005)
    assert got.unit == "%"
    # the subject over the reference: (subject / reference - 1) * 100
    ids = re.findall(r"f\d+", got.derivation)
    assert [facts[i].source_ref.split("|")[2] for i in ids] == [subject, reference]
    assert len(answer) == 3 and len(cells(answer)) == 2


def test_relative_neighbours_stay(res):
    # a percent of the other (loop 8) stays a level, an amount more stays a difference
    facts, answer = ask(res, "2021년 부산 사업체수는 서울의 몇 퍼센트야?", "FAKE9_FIRM")
    (got,) = derived(answer)
    assert got.value == pytest.approx(round(value("FAKE9_FIRM", "F1", "21", "2021") / value("FAKE9_FIRM", "F1", "11", "2021") * 100, 2), abs=0.005)
    facts, answer = ask(res, "2021년 서울 사업체수는 부산보다 몇 개 많아?", "FAKE9_FIRM")
    (got,) = derived(answer)
    assert got.value == value("FAKE9_FIRM", "F1", "11", "2021") - value("FAKE9_FIRM", "F1", "21", "2021")


# ---------------------------------------------------------------- D-1: two shares of one whole compared


@pytest.mark.parametrize("question, table, axis_codes, fixed, year, subject_first", [
    ("2021년 서울 1인 가구 비중은 2인 가구보다 몇 %p 높아?", "FAKE9_HOUSE", ("1", "2"), "11", "2021", True),
    ("2019년 부산의 3인 가구 구성비와 1인 가구 구성비 차이는?", "FAKE9_HOUSE", ("3", "1"), "21", "2019", None),
    ("2020년 대구에서 2인 가구가 차지하는 비중은 1인 가구 비중보다 얼마나 낮아?", "FAKE9_HOUSE", ("2", "1"), "22", "2020", True),
])
def test_two_shares_compared_by_size(res, question, table, axis_codes, fixed, year, subject_first):
    facts, answer = ask(res, question, table)
    total = value(table, "H1", fixed, year, "0")
    shares = [value(table, "H1", fixed, year, c) / total * 100 for c in axis_codes]
    gap = [f for f in answer if f.unit == "%p"]
    assert len(gap) == 1
    expected = shares[0] - shares[1] if subject_first else max(shares) - min(shares)
    assert gap[0].value == pytest.approx(round(expected, 2), abs=0.005)  # of the shares before rounding
    assert sorted(f.value for f in answer if f.unit == "%") == pytest.approx(sorted(round(s, 2) for s in shares), abs=0.005)
    assert cells(answer) == sorted([ref(table, "H1", fixed, year, c) for c in (*axis_codes, "0")])


def test_two_shares_compared_by_region(res):
    facts, answer = ask(res, "2022년 전국 사업체 중 서울 비중은 부산 비중보다 몇 퍼센트포인트 높아?", "FAKE9_FIRM")
    total = value("FAKE9_FIRM", "F1", "00", "2022")
    seoul, busan = (value("FAKE9_FIRM", "F1", c, "2022") / total * 100 for c in ("11", "21"))
    (gap,) = [f for f in answer if f.unit == "%p"]
    assert gap.value == pytest.approx(round(seoul - busan, 2), abs=0.005)
    assert cells(answer) == sorted(ref("FAKE9_FIRM", "F1", c, "2022") for c in ("00", "11", "21"))


def test_shares_joined_stay_one_share(res):
    # a word that joins the parts: one share of the parts together, as before
    facts, answer = ask(res, "2021년 서울 1인 가구와 2인 가구를 합친 비중은?", "FAKE9_HOUSE")
    shares = [f for f in answer if f.unit == "%"]
    total = value("FAKE9_HOUSE", "H1", "11", "2021", "0")
    part = value("FAKE9_HOUSE", "H1", "11", "2021", "1") + value("FAKE9_HOUSE", "H1", "11", "2021", "2")
    assert [f.value for f in shares] == [pytest.approx(round(part / total * 100, 2), abs=0.005)]
    assert not [f for f in answer if f.unit == "%p"]


# ---------------------------------------------------------------- E-1: a division of two units in one table


@pytest.mark.parametrize("question, region, year", [
    ("2021년 서울 종사자수를 사업체수로 나누면 얼마야?", "11", "2021"),
    ("2019년 부산의 사업체수 1개당 종사자수는?", "21", "2019"),
    ("2023년 대구 종사자수는 사업체수의 몇 배야?", "22", "2023"),
])
def test_division_of_two_units(res, question, region, year):
    facts, answer = ask(res, question, "FAKE9_WORK")
    (got,) = derived(answer)
    expected = value("FAKE9_WORK", "W2", region, year) / value("FAKE9_WORK", "W1", region, year)
    assert got.value == pytest.approx(round(expected, 2), abs=0.005)
    ids = re.findall(r"f\d+", got.derivation)
    assert [facts[i].source_ref for i in ids] == [ref("FAKE9_WORK", "W2", region, year), ref("FAKE9_WORK", "W1", region, year)]


def test_difference_of_two_units_still_refused(res):
    got = calculate("2021년 서울 종사자수와 사업체수의 차이는?", [res["FAKE9_WORK"]])
    assert got.status != "ok"


# ---------------------------------------------------------------- B-2: a word of the whole before 몇 + a counting word


@pytest.mark.parametrize("question, table, item, codes, year, size", [
    ("2022년 서울과 부산의 사업체는 모두 몇 개야?", "FAKE9_FIRM", "F1", ("11", "21"), "2022", None),
    ("2020년 서울, 부산, 대구 사업체수는 전부 몇 개?", "FAKE9_FIRM", "F1", ("11", "21", "22"), "2020", None),
    ("2021년 서울의 1인 가구와 2인 가구는 총 몇 가구야?", "FAKE9_HOUSE", "H1", ("1", "2"), "2021", "11"),
])
def test_total_adverb_sums_targets(res, question, table, item, codes, year, size):
    facts, answer = ask(res, question, table)
    (got,) = derived(answer)
    if size is None:
        want = [ref(table, item, c, year) for c in codes]
        expected = sum(value(table, item, c, year) for c in codes)
    else:
        want = [ref(table, item, size, year, c) for c in codes]
        expected = sum(value(table, item, size, year, c) for c in codes)
    assert got.value == expected
    assert refs_in(facts, got) == sorted(want)


def test_total_adverb_two_tables(res):
    for question, region, year in (("2021년 부산 출생아수와 사망자수는 모두 몇 명이야?", "21", "2021"),
                                   ("2019년 대구의 출생아수와 사망자수를 통틀어 몇 명이야?", "22", "2019")):
        facts, answer = ask(res, question, "FAKE9_BIRTH", "FAKE9_DEATH")
        (got,) = derived(answer)
        assert got.value == value("FAKE9_BIRTH", "B1", region, year) + value("FAKE9_DEATH", "D1", region, year)
        assert refs_in(facts, got) == sorted([ref("FAKE9_BIRTH", "B1", region, year), ref("FAKE9_DEATH", "D1", region, year)])


def test_total_adverb_neighbours_stay(res):
    # one target with the adverb: its value; the values asked for all: listed, no sum
    facts, answer = ask(res, "2022년 서울 사업체는 모두 몇 개야?", "FAKE9_FIRM")
    assert not derived(answer) and cells(answer) == [ref("FAKE9_FIRM", "F1", "11", "2022")]
    facts, answer = ask(res, "2022년 서울과 부산 사업체수를 모두 알려줘", "FAKE9_FIRM")
    assert not derived(answer)
    assert cells(answer) == sorted(ref("FAKE9_FIRM", "F1", c, "2022") for c in ("11", "21"))


# ---------------------------------------------------------------- B-1: a label holding a word of a comparison


@pytest.mark.parametrize("question, region, year", [
    ("2021년 부산의 남녀 임금차이는 얼마야?", "21", "2021"),
    ("서울 남녀 임금차이 알려줘 2019년 기준으로", "11", "2019"),
])
def test_label_with_comparison_word_is_a_value(res, question, region, year):
    facts, answer = ask(res, question, "FAKE9_WAGE")
    assert not derived(answer)
    assert [f.value for f in answer] == [value("FAKE9_WAGE", "G2", region, year)]
    assert cells(answer) == [ref("FAKE9_WAGE", "G2", region, year)]


def test_label_with_comparison_word_compared(res):
    # two targets compared: still a comparison (the comparing word is outside the label)
    facts, answer = ask(res, "2021년 서울과 부산의 남녀 임금차이를 비교해줘", "FAKE9_WAGE")
    assert cells(answer) == sorted(ref("FAKE9_WAGE", "G2", c, "2021") for c in ("11", "21"))
