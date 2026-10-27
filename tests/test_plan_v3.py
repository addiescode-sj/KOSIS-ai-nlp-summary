"""v3 plans (T0-04): the PlanV3 contract, the plan engine's blocks, and the plans the rules build (build.py)."""

import pytest
from pydantic import ValidationError

from conftest import write_table
from eval.grade import evaluate
from kosis_chat.compute import CalcRefused, build, compute
from kosis_chat.compute.plan_engine import run_plan
from kosis_chat.data.loader import load_table
from kosis_chat.data.preprocessing import clean

from kosis_ai.schemas import PlanV3, StatQuery, check_derivation  # isort: skip
from test_compute import CATALOG, ref, value


@pytest.fixture
def households(household_path):
    return clean(load_table(household_path))


def plan(steps, show, present="compare", **kw):
    return PlanV3(steps=steps, show=show, present=present, **kw)


def sel(step_id, *cells):
    return {"id": step_id, "op": "select", "cells": list(cells)}


def comb(step_id, a, b, fn, **kw):
    return {"id": step_id, "op": "combine", "a": a, "b": b, "fn": fn, **kw}


def recomputes(facts):
    by_id = {f.id: f.value for f in facts if f.value is not None}
    for f in facts:
        if f.derivation:
            assert abs(round(evaluate(f.derivation, by_id), f.round) - f.value) < 1e-9, f
    return True


# ---------------------------------------------------------------- contract


def test_steps_may_only_use_earlier_steps_and_show_existing_ones():
    with pytest.raises(ValidationError, match="not earlier steps"):
        plan([comb("d", "a", "b", "diff"), sel("a", "x|y|1"), sel("b", "x|y|2")], ["d"])
    with pytest.raises(ValidationError, match="do not exist"):
        plan([sel("a", "x|y|1")], ["z"], "value")
    with pytest.raises(ValidationError, match="used twice"):
        plan([sel("a", "x|y|1"), sel("a", "x|y|2")], ["a"], "value")


def test_block_shapes_and_fields_are_checked():
    with pytest.raises(ValidationError, match="not both or neither"):
        plan([{"id": "a", "op": "select", "cells": ["x|y|1"], "periods": {"kind": "recent", "n": 3}}], ["a"], "value")
    with pytest.raises(ValidationError, match="needs cells"):
        plan([sel("a", "x|y|1"), {"id": "m", "op": "aggregate", "of": "a", "fn": "avg"},
              {"id": "n", "op": "aggregate", "of": "m", "fn": "avg"}], ["n"], "value")
    with pytest.raises(ValidationError, match="years goes with cagr"):
        plan([sel("a", "x|y|1"), sel("b", "x|y|2"), comb("g", "b", "a", "cagr")], ["g"])
    with pytest.raises(ValidationError, match="exactly one"):
        plan([{"id": "a", "op": "select", "dims": {"지역": {"codes": ["11"], "total": True}}}], ["a"], "value")
    with pytest.raises(ValidationError, match="missing its fields"):
        plan([{"id": "a", "op": "select", "periods": {"kind": "anchor"}}], ["a"], "value")
    with pytest.raises(ValidationError, match="shows a rank step"):
        plan([sel("a", "x|y|1")], ["a"], "extreme")
    with pytest.raises(ValidationError, match="not supported yet"):
        plan([sel("a", "x|y|1")], ["a"], "value", based_on="earlier")


def test_spec_selects_are_part_of_the_contract(households):
    p = plan([{"id": "a", "op": "select", "item": {"codes": ["T1"]}, "dims": {"행정구역별(읍면동)": {"path": ["서울특별시"]}},
               "periods": {"kind": "anchor", "event": "코로나", "offset": -1}}], ["a"], "value")
    assert StatQuery(question="q", plan=p).plan.steps[0].periods.event == "코로나"
    with pytest.raises(CalcRefused, match="2019"):  # resolved to 2019, which the made-up table does not have
        run_plan(households, p)


def test_derivations_allow_a_constant_exponent_only():
    assert check_derivation("((f2 / f1) ** (1 / 9) - 1) * 100") == ["f2", "f1"]
    with pytest.raises(ValueError, match="exponent must be a number"):
        check_derivation("f2 ** f1")
    with pytest.raises(ValueError, match="not a real number"):
        evaluate("f1 ** 0.5", {"f1": -4.0})


# ---------------------------------------------------------------- new shapes from the same blocks


def test_gap_between_two_targets_over_two_periods(households):
    a15, b15, a24, b24 = ref("T1", "21", "2015"), ref("T1", "00", "2015"), ref("T1", "21", "2024"), ref("T1", "00", "2024")
    p = plan([sel("a15", a15), sel("b15", b15), sel("a24", a24), sel("b24", b24),
              comb("g15", "a15", "b15", "diff"), comb("g24", "a24", "b24", "diff"), comb("w", "g24", "g15", "diff")],
             ["g15", "g24", "w"])
    run = run_plan(households, p)
    by_id = {f.id: f for f in run.facts}
    assert [by_id[i].derivation for i in run.main] == ["f1 - f2", "f3 - f4", "f6 - f5"]
    assert by_id["f5"].value == value("T1", "21", "2015") - value("T1", "00", "2015")
    assert by_id["f7"].value == by_id["f6"].value - by_id["f5"].value
    assert recomputes(run.facts)


def test_growth_race_and_average_annual_growth(households):
    cells = {k: ref("T1", r, p) for k, (r, p) in {"a1": ("21", "2015"), "a2": ("21", "2024"),
                                                  "b1": ("11", "2015"), "b2": ("11", "2024")}.items()}
    p = plan([*(sel(k, v) for k, v in cells.items()),
              comb("ga", "a2", "a1", "pct_change"), comb("gb", "b2", "b1", "pct_change"), comb("d", "ga", "gb", "diff"),
              comb("c", "a2", "a1", "cagr", years=9)], ["ga", "gb", "d", "c"])
    run = run_plan(households, p)
    by_id = {f.id: f for f in run.facts}
    d, c = by_id[run.main[2]], by_id[run.main[3]]
    assert d.unit == "%p" and d.derivation == "f5 - f6"
    assert c.derivation == "((f2 / f1) ** (1 / 9) - 1) * 100" and c.unit == "%"
    expected = ((value("T1", "21", "2024") / value("T1", "21", "2015")) ** (1 / 9) - 1) * 100
    assert c.value == round(expected, 1)
    assert recomputes(run.facts)


def test_find_points_at_cells_and_per_scales(households):
    series = [ref("T1", "21", p) for p in ("2015", "2020", "2023", "2024")]
    threshold = value("T1", "21", "2020")
    p = plan([sel("s", *series), {"id": "f", "op": "find", "of": "s", "cmp": ">", "value": threshold},
              sel("t", ref("T0", "21", "2024")), sel("x", ref("T1", "21", "2024")), comb("k", "x", "t", "per", scale=1000)],
             ["f", "k"], "value")
    run = run_plan(households, p)
    by_id = {f.id: f for f in run.facts}
    assert by_id[run.main[0]].period == "2023"  # the first period above the 2020 value
    assert by_id[run.main[1]].derivation == "f6 / f5 * 1000"
    with pytest.raises(CalcRefused, match="조건에 맞는 값이 없음"):
        run_plan(households, plan([sel("s", *series), {"id": "f", "op": "find", "of": "s", "cmp": "<", "value": 0}], ["f"], "value"))


def test_rank_can_keep_more_than_two(households):
    regions = ["00", "11", "21", "39"]
    p = plan([{"id": "k", "op": "rank", "candidates": [ref("T1", r, "2023") for r in regions], "pick": "max", "keep": 3}],
             ["k"], "extreme")
    run = run_plan(households, p)
    assert [f.note for f in run.facts] == ["4개 중 가장 높음", "두 번째", "3번째"]
    assert run.results["k"].picked == {"winner": "f1", "runner_up": "f2", "place_3": "f3"}


def test_optional_combines_are_skipped_not_refused(pop_path, tmp_path):
    pop = clean(load_table(pop_path))
    count = plan([sel("a", "FAKE_TBL_001|T1|0|2023"), sel("b", "FAKE_TBL_001|T1|0|2022"),
                  comb("d", "a", "b", "diff"), comb("g", "a", "b", "pct_change", optional=True)], ["d", "g"], "trend")
    assert len(run_plan(pop, count).main) == 2  # a count: the growth rate is kept
    rows = [{"period": p, "item_id": "T1", "item_name": "가상 출생률", "unit": "%", "value_raw": v, "value": float(v)}
            for p, v in (("2022", "1.5"), ("2023", "1.2"))]
    rates = clean(load_table(write_table(tmp_path / "data", "FAKE_RATE", rows, CATALOG)))
    rate = plan([sel("a", "FAKE_RATE|T1||2023"), sel("b", "FAKE_RATE|T1||2022"),
                 comb("d", "a", "b", "diff"), comb("g", "a", "b", "pct_change", optional=True)], ["d", "g"], "trend")
    run = run_plan(rates, rate)
    assert len(run.main) == 1 and run.facts[-1].unit == "%p"  # a growth rate of a % means nothing: skipped
    with pytest.raises(CalcRefused, match="비교할 값이 부족함"):
        run_plan(pop, plan([sel("a", "FAKE_TBL_001|T1|0|2023"), {"id": "b", "op": "aggregate", "of": "a", "fn": "first"},
                            comb("d", "a", "b", "diff")], ["d"]))


# ---------------------------------------------------------------- plans the rules build (build.py)


def test_compute_records_the_plan(households):
    computed = compute(households, build.trend([ref("T1", "21", p) for p in ("2015", "2024")]))
    assert computed.plan.present == "trend" and [s.op for s in computed.plan.steps] == ["select", "aggregate", "aggregate", "combine", "combine"]
    assert computed.pattern == "trend"


def test_gap_is_the_change_of_the_gap_between_two_targets(households):
    early, late = (ref("T1", "11", "2015"), ref("T1", "21", "2015")), (ref("T1", "11", "2024"), ref("T1", "21", "2024"))
    computed = compute(households, build.gap(late, early))  # given in any order: the earlier period comes first
    by_id = {f.id: f for f in computed.result.facts}
    assert [by_id[f"f{n}"].source_ref for n in range(1, 5)] == [early[0], early[1], late[0], late[1]]
    # 부산 (region 3) is larger than 서울 (region 2) in the made-up data: each gap is larger minus smaller.
    assert (by_id["f5"].derivation, by_id["f6"].derivation, by_id["f7"].derivation) == ("f2 - f1", "f4 - f3", "f6 - f5")
    assert by_id["f5"].value == 1000 and by_id["f7"].value == 0 and computed.pattern == "gap" and recomputes(computed.result.facts)


def test_growth_compares_two_growth_rates(households):
    seoul, busan = (ref("T1", "11", "2015"), ref("T1", "11", "2024")), (ref("T1", "21", "2015"), ref("T1", "21", "2024"))
    computed = compute(households, build.growth(seoul, busan))
    by_id = {f.id: f for f in computed.result.facts}
    assert by_id["f3"].derivation == "(f2 / f1 - 1) * 100" and by_id["f6"].derivation == "(f5 / f4 - 1) * 100"
    assert by_id["f7"].unit == "%p" and by_id["f7"].value >= 0 and recomputes(computed.result.facts)  # the larger rate first
    decline = compute(households, build.growth(seoul, busan, decline=True))
    assert {f.id: f for f in decline.result.facts}["f7"].value <= 0  # the smaller (more negative) change first
    amount = compute(households, build.growth(seoul, busan, amount=True))
    by_id = {f.id: f for f in amount.result.facts}
    assert by_id["f3"].derivation == "f2 - f1" and by_id["f7"].unit == "가구" and by_id["f7"].value == 0

