"""Loop 8 (analyze/rules.py, analyze/roles.py, analyze/blocks.py, shared/patterns.py):

- C-2: three targets or more of one category, each named by its own name, two periods and a change asked: each
  target's change and its two cells (as two targets were, loop 7), no difference between the changes.
- D-1: the average of targets named one by one at one period (as their sum is), and the average of periods named one
  by one (as their sum is): one average and its cells.
- B-1: a division asked with particles ("A를 B로 나누면"): the target with 로 is the denominator, whatever counting
  words come between its name and the particle.
- B-2: one target as a percent of another that does not hold it ("부산은 서울의 몇 퍼센트"): target / reference * 100,
  not a share of the category's total, not a difference.
- C-3: the shares of several parts asked each ("각각"): one share per part over the same whole.

The tables are made up (가상 데이터). Every test checks the answer's values and the cells they were computed from.
"""

from __future__ import annotations

import re

import pytest

from conftest import write_table
from kosis_chat.adapters.parquet import load_table
from kosis_chat.calc import calculate
from kosis_chat.schemas.resources import load_resources

REGIONS = (("00", "전국"), ("11", "서울특별시"), ("21", "부산광역시"), ("22", "대구광역시"), ("23", "인천광역시"))
YEARS = [str(y) for y in range(2012, 2024)]


def vital_rows():
    """출생아수 and 사망자수 by region; the regions do not add up to 전국 (some are left out)."""
    out = []
    for y_no, year in enumerate(YEARS):
        for r_no, (code, label) in enumerate(REGIONS):
            vals = {"V1": ("출생아수", 52000 - 1300 * y_no - 9100 * r_no + 41 * r_no * y_no),
                    "V2": ("사망자수", 30000 + 450 * y_no - 4700 * r_no + 13 * r_no * y_no)}
            if code == "00":
                vals = {"V1": ("출생아수", 90000 - 2100 * y_no), "V2": ("사망자수", 70000 + 800 * y_no)}
            for item, (name, v) in vals.items():
                out.append({"period": year, "item_id": item, "item_name": name, "unit": "명", "dim1_name": "시도별",
                            "dim1_code": code, "dim1_label": label, "value_raw": str(v), "value": v})
    return out


def household_rows():
    out = []
    for y_no, year in enumerate(YEARS):
        for r_no, (code, label) in enumerate(REGIONS):
            v = 4000 + 120 * y_no if code == "00" else 700 + 29 * y_no + 157 * r_no - 3 * r_no * y_no
            out.append({"period": year, "item_id": "H1", "item_name": "1인가구수", "unit": "가구", "dim1_name": "시도별",
                        "dim1_code": code, "dim1_label": label, "value_raw": str(v), "value": v})
    return out


def catalog(survey, title):
    return {"survey_name": survey, "organization": "가상 기관", "table_name": title, "period": "년",
            "url": "https://example.invalid/t"}


SPECS = {
    "FAKE8_VITAL": (vital_rows, catalog("가상 인구동향조사", "가상 시도별 출생과 사망")),
    "FAKE8_HOUSE": (household_rows, catalog("가상 가구조사", "가상 시도별 1인가구")),
}


@pytest.fixture(scope="module")
def res(tmp_path_factory):
    folder = tmp_path_factory.mktemp("loop8") / "data"
    return {t: load_resources(load_table(write_table(folder, t, make(), cat))) for t, (make, cat) in SPECS.items()}


def ask(res, question, *tbl_ids):
    got = calculate(question, [res[t] for t in tbl_ids])
    assert got.status == "ok", (got.status, got.reason, got.detail)
    facts = {f.id: f for f in got.facts}
    return facts, [facts[i] for i in got.answer]


def refs_in(facts, fact):
    """The cells a derived fact was computed from, following derived inputs down to the cells."""
    out = []
    for i in re.findall(r"f\d+", fact.derivation or ""):
        f = facts[i]
        out += refs_in(facts, f) if f.derivation else [f.source_ref]
    return sorted(out)


def value_of(tbl, item, region, year):
    rows = SPECS[tbl][0]()
    return next(r["value"] for r in rows if r["item_id"] == item and r["dim1_code"] == region and r["period"] == year)


def ref(tbl, item, region, year):
    return f"{tbl}|{item}|{region}|{year}"


def cells(answer):
    return [f.source_ref for f in answer if f.source_ref]


def derived(answer):
    return [f for f in answer if f.derivation]


# ---------------------------------------------------------------- C-2: each of three targets' change


@pytest.mark.parametrize("question, table, item, regions, years, rate", [
    ("2015년과 2020년 서울, 부산, 대구의 출생아 수는 각각 얼마나 줄었어?", "FAKE8_VITAL", "V1", ("11", "21", "22"),
     ("2015", "2020"), False),
    ("2014년 대비 2021년 부산, 대구, 인천 1인가구는 저마다 몇 퍼센트 늘었어?", "FAKE8_HOUSE", "H1", ("21", "22", "23"),
     ("2014", "2021"), True),
    ("2013년부터 2019년까지 전국, 서울, 부산, 인천의 사망자 수는 각각 몇 명 늘었어?", "FAKE8_VITAL", "V2",
     ("00", "11", "21", "23"), ("2013", "2019"), False),
])
def test_each_of_three_targets_change(res, question, table, item, regions, years, rate):
    facts, answer = ask(res, question, table)
    changes = derived(answer)
    assert len(changes) == len(regions)  # one per target, nothing that compares them
    early, late = years
    for change, region in zip(sorted(changes, key=lambda f: refs_in(facts, f)), sorted(regions)):
        assert refs_in(facts, change) == sorted([ref(table, item, region, early), ref(table, item, region, late)])
        a, b = value_of(table, item, region, early), value_of(table, item, region, late)
        want = (b / a - 1) * 100 if rate else b - a
        assert change.value == pytest.approx(round(want, 2), abs=0.01)
    assert sorted(cells(answer)) == sorted(ref(table, item, r, y) for r in regions for y in years)


# ---------------------------------------------------------------- D-1: the average of named targets / periods


@pytest.mark.parametrize("question, table, wanted", [
    ("2023년 서울, 부산, 대구 출생아 수 평균은?", "FAKE8_VITAL",
     [("V1", "11", "2023"), ("V1", "21", "2023"), ("V1", "22", "2023")]),
    ("2020년 부산과 인천의 1인가구수 평균 알려줘", "FAKE8_HOUSE", [("H1", "21", "2020"), ("H1", "23", "2020")]),
    ("2018년 대구의 출생아 수와 사망자 수를 평균 내면?", "FAKE8_VITAL", [("V1", "22", "2018"), ("V2", "22", "2018")]),
    # periods named one by one
    ("2015년, 2018년, 2021년 부산 출생아 수 평균은?", "FAKE8_VITAL",
     [("V1", "21", "2015"), ("V1", "21", "2018"), ("V1", "21", "2021")]),
    ("2019년과 2022년의 서울 1인가구수 평균은 얼마야?", "FAKE8_HOUSE", [("H1", "11", "2019"), ("H1", "11", "2022")]),
])
def test_average_of_named(res, question, table, wanted):
    facts, answer = ask(res, question, table)
    averages = derived(answer)
    assert len(averages) == 1
    want_refs = sorted(ref(table, i, r, y) for i, r, y in wanted)
    assert refs_in(facts, averages[0]) == want_refs
    want = sum(value_of(table, i, r, y) for i, r, y in wanted) / len(wanted)
    assert averages[0].value == pytest.approx(round(want, 2), abs=0.01)
    assert set(cells(answer)) <= set(want_refs)


def test_sum_of_named_targets_unchanged(res):
    """Neighbour: the same targets with a word of a sum: their sum, as before."""
    facts, answer = ask(res, "2023년 서울, 부산, 대구 출생아 수를 합치면?", "FAKE8_VITAL")
    total = derived(answer)[0]
    assert total.value == sum(value_of("FAKE8_VITAL", "V1", r, "2023") for r in ("11", "21", "22"))


def test_range_average_unchanged(res):
    """Neighbour: a range averaged, as before."""
    facts, answer = ask(res, "2015~2018년 부산 출생아 수 평균은?", "FAKE8_VITAL")
    avg = derived(answer)[0]
    assert refs_in(facts, avg) == sorted(ref("FAKE8_VITAL", "V1", "21", str(y)) for y in range(2015, 2019))


# ---------------------------------------------------------------- B-1: the denominator named by its particle


@pytest.mark.parametrize("question, table, sub, den", [
    ("2023년 부산 출생아 수를 서울 출생아 수로 나누면?", "FAKE8_VITAL", ("V1", "21", "2023"), ("V1", "11", "2023")),
    ("2016년 대구 출생아를 부산으로 나눈 값은 얼마야?", "FAKE8_VITAL", ("V1", "22", "2016"), ("V1", "21", "2016")),
    ("2020년 서울 1인가구수를 인천 1인가구수로 나눠 줘", "FAKE8_HOUSE", ("H1", "11", "2020"), ("H1", "23", "2020")),
    # items: the death count over the birth count of one region
    ("2022년 서울의 사망자 수를 출생아 수로 나누면 얼마야?", "FAKE8_VITAL", ("V2", "11", "2022"), ("V1", "11", "2022")),
])
def test_division_by_particle(res, question, table, sub, den):
    facts, answer = ask(res, question, table)
    ratios = derived(answer)
    assert len(ratios) == 1
    assert refs_in(facts, ratios[0]) == sorted([ref(table, *sub), ref(table, *den)])
    assert ratios[0].derivation.count("/") == 1 and "-" not in ratios[0].derivation
    want = value_of(table, *sub) / value_of(table, *den)
    assert ratios[0].value == pytest.approx(round(want, 2), abs=0.01)


# ---------------------------------------------------------------- B-2: one target as a percent of another


@pytest.mark.parametrize("question, table, sub, base", [
    ("2023년 부산 출생아 수는 서울의 몇 퍼센트야?", "FAKE8_VITAL", ("V1", "21", "2023"), ("V1", "11", "2023")),
    ("2019년 인천 출생아 수는 대구 출생아 수의 몇 %인가요?", "FAKE8_VITAL", ("V1", "23", "2019"), ("V1", "22", "2019")),
    ("2021년 서울 대비 대구의 1인가구수 비율은 몇 퍼센트야?", "FAKE8_HOUSE", ("H1", "22", "2021"), ("H1", "11", "2021")),
    # items of one region
    ("2017년 부산의 출생아 수는 사망자 수의 몇 퍼센트였어?", "FAKE8_VITAL", ("V1", "21", "2017"), ("V2", "21", "2017")),
])
def test_percent_of_another_target(res, question, table, sub, base):
    facts, answer = ask(res, question, table)
    levels = derived(answer)
    assert len(levels) == 1
    assert refs_in(facts, levels[0]) == sorted([ref(table, *sub), ref(table, *base)])
    want = value_of(table, *sub) / value_of(table, *base) * 100
    assert levels[0].value == pytest.approx(round(want, 2), abs=0.01)
    assert sorted(cells(answer)) == sorted([ref(table, *sub), ref(table, *base)])


def test_percent_of_the_whole_unchanged(res):
    """Neighbour: a percent of the category's total is a share, as before."""
    facts, answer = ask(res, "2023년 부산 출생아 수는 전국의 몇 퍼센트야?", "FAKE8_VITAL")
    share = derived(answer)[-1]
    assert refs_in(facts, share) == sorted([ref("FAKE8_VITAL", "V1", "00", "2023"), ref("FAKE8_VITAL", "V1", "21", "2023")])


# ---------------------------------------------------------------- C-3: each part's share


@pytest.mark.parametrize("question, table, item, year, parts, whole", [
    ("2023년 전국 출생아 중 서울과 부산의 비중은 각각?", "FAKE8_VITAL", "V1", "2023", ("11", "21"), "00"),
    ("2020년 1인가구 가운데 부산, 대구, 인천이 차지하는 비중을 각각 알려줘", "FAKE8_HOUSE", "H1", "2020",
     ("21", "22", "23"), "00"),
])
def test_each_part_share(res, question, table, item, year, parts, whole):
    facts, answer = ask(res, question, table)
    shares = derived(answer)
    assert len(shares) == len(parts)
    total = value_of(table, item, whole, year)
    for share, part in zip(sorted(shares, key=lambda f: refs_in(facts, f)), sorted(parts)):
        assert refs_in(facts, share) == sorted([ref(table, item, whole, year), ref(table, item, part, year)])
        assert share.value == pytest.approx(round(value_of(table, item, part, year) / total * 100, 2), abs=0.01)
    assert sorted(cells(answer)) == sorted(ref(table, item, c, year) for c in (*parts, whole))


def test_combined_share_unchanged(res):
    """Neighbour: no word of each: the parts' share together, as before."""
    facts, answer = ask(res, "2023년 전국 출생아 중 서울과 부산을 합친 비중은?", "FAKE8_VITAL")
    shares = derived(answer)
    assert len(shares) == 1
    assert len(refs_in(facts, shares[0])) == 3
