"""Meaning checks of v3 plans (T0-04 S3, kosis_chat/compute/semantics.py): computable but meaningless is refused."""

from dataclasses import replace

import pytest

from kosis_chat.compute import CalcRefused
from kosis_chat.compute import semantics
from kosis_chat.compute.engine import _read
from kosis_chat.compute.plan_engine import run_plan
from kosis_chat.data.loader import load_table
from kosis_chat.data.preprocessing import clean

from kosis_ai.schemas import PlanV3  # isort: skip
from test_compute import ref

MEN, WOMEN, ALL, RATIO = (f"FAKE_TBL_001|{i}|{s}|2023" for i, s in (("T1", "1"), ("T1", "2"), ("T1", "0"), ("T2", "0")))


@pytest.fixture
def pop(pop_path):
    return clean(load_table(pop_path))


@pytest.fixture
def households(household_path):
    return clean(load_table(household_path))


def sel(step_id, *cells):
    return {"id": step_id, "op": "select", "cells": list(cells)}


def run(table, steps, show, present="value"):
    return run_plan(table, PlanV3(steps=steps, show=show, present=present))


def refused(table, steps, show, reason, present="value"):
    with pytest.raises(CalcRefused) as exc:
        run(table, steps, show, present)
    assert exc.value.reason == reason
    return exc.value.message


def test_r1_sums_over_periods_need_a_known_flow(households):
    reads = [_read(households, ref("T1", "11", p)) for p in ("2015", "2020")]
    for kind in ("count_stock", "ratio", None):
        with pytest.raises(CalcRefused, match="의미가 없습니다"):
            semantics.check_sum([replace(r, kind=kind) for r in reads])
    semantics.check_sum([replace(r, kind="count_flow") for r in reads])  # births, marriages: allowed


def test_r2_an_average_of_rates_is_kept_with_a_caution(households):
    reads = [_read(households, ref("T1", "11", p)) for p in ("2015", "2020")]
    assert semantics.average_caution([replace(r, kind="count_stock") for r in reads]) is None
    assert semantics.average_caution([replace(r, kind="ratio") for r in reads]).startswith("simple_average")


def test_r3_r4_differences_and_growth_compare_like_with_like(pop):
    count_vs_rate = [sel("a", ALL), sel("b", RATIO)]
    for fn in ("pct_change", "cagr"):
        extra = {"years": 2} if fn == "cagr" else {}
        refused(pop, [*count_vs_rate, {"id": "c", "op": "combine", "a": "a", "b": "b", "fn": fn, **extra}], ["c"], "unit_mismatch")
    message = refused(pop, [sel("a", MEN), sel("b", WOMEN), {"id": "c", "op": "combine", "a": "a", "b": "b", "fn": "pp_diff"}],
                      ["c"], "unit_mismatch")
    assert "%p 차이는 두 값이 모두 %" in message


def test_r5_the_denominator_must_hold_the_numerator(pop, households):
    share = {"id": "s", "op": "combine", "a": "p", "b": "t", "fn": "share"}
    ok = run(pop, [sel("p", MEN), sel("t", ALL), share], ["s"], "share")
    assert ok.facts[-1].value == round(500 / 1050 * 100, 1)
    assert "성별 남자⊄여자" in refused(pop, [sel("p", MEN), sel("t", WOMEN), share], ["s"], "class_mismatch", "share")
    # 종로구 is inside 서울 by code prefix but not inside 부산.
    assert "⊄부산광역시" in refused(households, [sel("p", ref("T1", "11010", "2023")), sel("t", ref("T1", "21", "2023")), share],
                                  ["s"], "class_mismatch", "share")
    assert "시점" in refused(pop, [sel("p", MEN), sel("t", "FAKE_TBL_001|T1|0|2022"), share], ["s"], "class_mismatch", "share")


def test_r5_whole_labels_and_a_year_holding_a_month():
    assert semantics._period_covers("2025", "202512") and not semantics._period_covers("2024", "202512")
    assert not semantics._period_covers("202512", "2025")
    assert semantics._whole_label("총인구") and semantics._whole_label("전국") and not semantics._whole_label("남자")


def test_r6_a_share_above_100_percent_is_refused(households):
    # 종로구 is inside 서울 by its code, but the made-up values make it larger than 서울.
    share = {"id": "s", "op": "combine", "a": "p", "b": "t", "fn": "share"}
    message = refused(households, [sel("p", ref("T1", "11010", "2023")), sel("t", ref("T1", "11", "2023")), share],
                      ["s"], "class_mismatch", "share")
    assert "0~100% 밖" in message


def test_r8_r9_rankings_and_thresholds_share_a_unit(pop):
    refused(pop, [{"id": "k", "op": "rank", "candidates": [ALL, RATIO], "pick": "max"}], ["k"], "unit_mismatch", "extreme")
    series = ["FAKE_TBL_001|T1|0|2022", ALL]
    refused(pop, [sel("s", *series), sel("r", RATIO), {"id": "f", "op": "find", "of": "s", "cmp": ">", "ref": "r"}],
            ["f"], "unit_mismatch")
    found = run(pop, [sel("s", *series), sel("r", "FAKE_TBL_001|T1|0|2022"),
                      {"id": "f", "op": "find", "of": "s", "cmp": ">", "ref": "r"}], ["f"])
    assert [f.period for f in found.facts if f.id in found.main] == ["2023"]
