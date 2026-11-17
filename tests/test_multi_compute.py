"""T1-30: one plan over two or three tables (compute/plan_engine.py tables=, compute/scale.py).

The tables are made up (가상 데이터): births and deaths in 명, mid-year population in 천명, households in 가구.
"""

import pytest

from conftest import write_table
from eval.grade import evaluate
from kosis_chat.compute import CalcRefused, build, compute
from kosis_chat.compute.scale import common_unit, conversion, scale_of
from kosis_chat.data.loader import load_table
from kosis_chat.data.preprocessing import clean

from kosis_ai.schemas import PlanV3  # isort: skip

CATALOG = {"survey_name": "가상 조사", "organization": "가상 기관", "table_name": "가상 표", "period": "년",
           "url": "https://example.invalid/t"}
REGIONS = (("00", "전국"), ("11", "서울특별시"), ("21", "부산광역시"))
PERIODS = ("2022", "2023", "2024")


def rows(item_id, item_name, unit, base):
    out = []
    for p_no, period in enumerate(PERIODS):
        for r_no, (code, label) in enumerate(REGIONS):
            value = base * (3 - r_no) + 10 * p_no
            out.append({"period": period, "item_id": item_id, "item_name": item_name, "unit": unit,
                        "dim1_name": "행정구역별", "dim1_code": code, "dim1_label": label,
                        "value_raw": str(value), "value": value})
    return out


@pytest.fixture
def tables(tmp_path):
    folder = tmp_path / "data"
    specs = {"FAKE_BIRTH": ("T1", "출생아수", "명", 1000), "FAKE_DEATH": ("T1", "사망자수", "명", 900),
             "FAKE_POPK": ("T1", "연앙인구", "천명", 50), "FAKE_HH": ("T1", "가구수", "가구", 400)}
    out = {}
    for tbl_id, spec in specs.items():
        path = write_table(folder, tbl_id, rows(*spec), CATALOG)
        out[tbl_id] = clean(load_table(path))
    return out


def plan(steps, show, present="compare"):
    return build.written(PlanV3(steps=steps, show=show, present=present))


def sel(step_id, table, region="11", period="2024"):
    return {"id": step_id, "op": "select", "table": table, "item": {"codes": ["T1"]},
            "dims": {"행정구역별": {"codes": [region]}}, "periods": {"kind": "list", "periods": [period]}}


def comb(step_id, a, b, fn, **kw):
    return {"id": step_id, "op": "combine", "a": a, "b": b, "fn": fn, **kw}


def run(tables, names, steps, show, present="compare"):
    picked = {n: tables[n] for n in names}
    return compute(picked[names[0]], plan(steps, show, present), tables=picked)


def recomputes(facts):
    by_id = {f.id: f.value for f in facts if f.value is not None}
    for f in facts:
        if f.derivation:
            assert abs(round(evaluate(f.derivation, by_id), f.round) - f.value) < 1e-9, f
    return True


def test_scale_of_reads_power_of_ten_prefixes_only():
    assert scale_of("천명") == ("명", 1000)
    assert scale_of("백만원") == ("원", 10 ** 6)
    assert scale_of("명") == ("명", 1)
    assert scale_of("천명당") is None and scale_of("%") is None and scale_of(None) is None
    assert conversion("천명", "명") == ("*", 1000) and conversion("명", "천명") == ("/", 1000)
    assert conversion("명", "가구") is None and conversion("명", "명") is None
    assert common_unit("천명", "명") == "명" and common_unit("명", "가구") is None


@pytest.fixture
def natural(tables):
    return run(tables, ["FAKE_BIRTH", "FAKE_DEATH"],
               [sel("b", "FAKE_BIRTH"), sel("d", "FAKE_DEATH"), comb("n", "b", "d", "diff", note="자연증가")], ["b", "d", "n"])


def test_births_minus_deaths(natural):
    facts = {f.id: f for f in natural.result.facts}
    assert facts["f1"].source_ref == "FAKE_BIRTH|T1|11|2024" and facts["f1"].value == 2020
    assert facts["f2"].source_ref == "FAKE_DEATH|T1|11|2024" and facts["f2"].value == 1820
    assert facts["f3"].derivation == "f1 - f2" and facts["f3"].value == 200 and facts["f3"].unit == "명"
    assert natural.main == ["f1", "f2", "f3"]
    assert recomputes(natural.result.facts)


def test_per_thousand_converts_thousands_first(tables):
    """분자·분모: births per 1,000 people, the population in 천명 turned into 명 by a conversion fact."""
    out = run(tables, ["FAKE_BIRTH", "FAKE_POPK"],
              [sel("b", "FAKE_BIRTH"), sel("p", "FAKE_POPK"), comb("r", "b", "p", "per", scale=1000)], ["r"], "value")
    facts = {f.id: f for f in out.result.facts}
    assert facts["f2"].unit == "천명" and facts["f2"].value == 120
    assert facts["f3"].derivation == "f2 * 1000" and facts["f3"].value == 120000 and facts["f3"].unit == "명"
    assert "환산" in facts["f3"].note
    assert facts["f4"].derivation == "f1 / f3 * 1000" and facts["f4"].value == pytest.approx(16.8, abs=0.05)
    assert recomputes(out.result.facts)


def test_diff_converts_the_coarser_unit(tables):
    out = run(tables, ["FAKE_POPK", "FAKE_BIRTH"],
              [sel("p", "FAKE_POPK"), sel("b", "FAKE_BIRTH"), comb("d", "p", "b", "diff")], ["d"])
    facts = {f.id: f for f in out.result.facts}
    assert facts["f3"].derivation == "f1 * 1000" and facts["f4"].derivation == "f3 - f2"
    assert facts["f4"].value == 120000 - 2020 and facts["f4"].unit == "명"


def test_units_of_another_base_are_refused_for_a_diff(tables):
    with pytest.raises(CalcRefused) as exc:
        run(tables, ["FAKE_BIRTH", "FAKE_HH"],
            [sel("b", "FAKE_BIRTH"), sel("h", "FAKE_HH"), comb("d", "b", "h", "diff")], ["d"])
    assert exc.value.reason == "unit_mismatch"


def test_per_allows_another_base(tables):
    out = run(tables, ["FAKE_BIRTH", "FAKE_HH"],
              [sel("b", "FAKE_BIRTH"), sel("h", "FAKE_HH"), comb("r", "b", "h", "per", scale=100)], ["r"], "value")
    assert out.result.facts[-1].derivation == "f1 / f2 * 100"


def test_share_across_tables_converts_the_whole(tables):
    out = run(tables, ["FAKE_BIRTH", "FAKE_POPK"],
              [sel("b", "FAKE_BIRTH"), sel("p", "FAKE_POPK"), comb("s", "b", "p", "share")], ["s"], "share")
    facts = {f.id: f for f in out.result.facts}
    assert facts["f3"].derivation == "f2 * 1000" and facts["f4"].derivation == "f1 / f3 * 100"
    assert recomputes(out.result.facts)


def test_share_across_tables_needs_the_same_period(tables):
    with pytest.raises(CalcRefused) as exc:
        run(tables, ["FAKE_BIRTH", "FAKE_POPK"],
            [sel("b", "FAKE_BIRTH"), sel("p", "FAKE_POPK", period="2023"), comb("s", "b", "p", "share")], ["s"], "share")
    assert exc.value.reason == "class_mismatch"


def test_trend_of_a_difference_over_two_periods(tables):
    """Three steps deep: each period's natural increase, then the change between them."""
    out = run(tables, ["FAKE_BIRTH", "FAKE_DEATH"],
              [sel("b1", "FAKE_BIRTH", period="2022"), sel("d1", "FAKE_DEATH", period="2022"), comb("n1", "b1", "d1", "diff"),
               sel("b2", "FAKE_BIRTH"), sel("d2", "FAKE_DEATH"), comb("n2", "b2", "d2", "diff"),
               comb("g", "n2", "n1", "diff")], ["n1", "n2", "g"], "trend")
    assert out.result.facts[-1].value == 0 and recomputes(out.result.facts)


def test_a_table_the_question_was_not_given_is_out_of_scope(tables):
    with pytest.raises(CalcRefused) as exc:
        run(tables, ["FAKE_BIRTH", "FAKE_DEATH"], [sel("b", "FAKE_BIRTH"), sel("x", "FAKE_HH")], ["b", "x"])
    assert exc.value.reason == "out_of_scope"


def test_one_table_given_as_tables_is_the_one_table_path(tables):
    one = {"FAKE_BIRTH": tables["FAKE_BIRTH"]}
    steps = [sel("a", "FAKE_BIRTH"), sel("b", "FAKE_BIRTH", region="21"), comb("d", "a", "b", "diff")]
    with_tables = compute(tables["FAKE_BIRTH"], plan(steps, ["d"]), tables=one)
    without = compute(tables["FAKE_BIRTH"], plan(steps, ["d"]))
    assert with_tables.result == without.result


def test_a_per_thousand_is_worded_as_a_rate_not_a_share():
    from kosis_chat.shared.numbers import derived_kind

    from kosis_ai.schemas import Fact  # isort: skip

    assert derived_kind(Fact(id="f3", value=16.8, derivation="f1 / f2 * 1000", round=1)) == "비율"
    assert derived_kind(Fact(id="f3", value=16.8, derivation="f1 / f2 * 100", round=1)) == "비중"
    assert derived_kind(Fact(id="f3", value=1.2, derivation="(f1 / f2 - 1) * 100", round=1)) == "증감률"
