"""Periods and targets by meaning (T0-04 S4, kosis_chat/compute/resolve.py), the map block and ranking a map."""

import pytest
from pydantic import ValidationError

from conftest import write_table
from kosis_chat.compute import CalcRefused
from kosis_chat.compute.plan_engine import run_plan
from kosis_chat.compute.resolve import NeedsClarify, load_events, resolve_select
from kosis_chat.data.loader import load_table
from kosis_chat.data.preprocessing import clean

from kosis_ai.schemas import PlanV3, SelectStep  # isort: skip
from test_compute import ref, value

REGION = "행정구역별(읍면동)"


@pytest.fixture
def households(household_path):
    return clean(load_table(household_path))


@pytest.fixture
def monthly(tmp_path):
    periods = [f"{y}{m:02d}" for y in (2023, 2024) for m in range(1, 13)] + ["2023", "2024"]
    rows = [{"period": p, "prd_se": "M", "item_id": "T1", "item_name": "가상 혼인건수", "unit": "건",
             "value_raw": str(n + 1), "value": float(n + 1)} for n, p in enumerate(periods)]
    catalog = {"survey_name": "가상 동향", "organization": "가상 기관", "table_name": "가상 월별 혼인", "period": "월", "url": "https://example.invalid/m"}
    return clean(load_table(write_table(tmp_path / "data", "FAKE_MONTH", rows, catalog)))


def select(table, **spec):
    return resolve_select(table, SelectStep(id="s", **spec))


def periods_of(resolved):
    return [r.rsplit("|", 1)[-1] for r in resolved.cells]


def test_targets_by_code_children_total_and_path(households):
    one = {"codes": ["T1"]}
    at = {"kind": "list", "periods": ["2023"]}
    assert select(households, item=one, dims={REGION: {"path": ["서울특별시", "중구"]}}, periods=at).cells == [ref("T1", "11020", "2023")]
    assert select(households, item=one, dims={REGION: {"children_of": "11"}}, periods=at).cells == [
        ref("T1", "11010", "2023"), ref("T1", "11020", "2023")]
    assert select(households, item=one, dims={REGION: {"total": True}}, periods=at).cells == [ref("T1", "00", "2023")]
    with pytest.raises(CalcRefused) as exc:
        select(households, item={"codes": ["T9"]}, periods=at)
    assert exc.value.reason == "class_mismatch"


def test_a_label_in_several_places_asks_back(households):
    with pytest.raises(NeedsClarify) as exc:
        select(households, item={"codes": ["T1"]}, dims={REGION: {"path": ["중구"]}}, periods={"kind": "list", "periods": ["2023"]})
    assert exc.value.axis == REGION
    assert exc.value.options == [("11020", "서울특별시 > 중구"), ("21010", "부산광역시 > 중구")]


def test_categories_the_plan_leaves_out_are_set_to_their_total_with_a_caution(households):
    got = select(households, periods={"kind": "list", "periods": ["2023"]})
    assert got.cells == [ref("T0", "00", "2023")]  # 일반가구_계, 전국
    assert got.cautions == ["default_applied: 항목 = 일반가구_계 (가구)", f"default_applied: {REGION} = 전국"]
    latest = select(households, item={"codes": ["T1"]}, dims={REGION: {"codes": ["11"]}})
    assert periods_of(latest) == ["2024"] and "default_applied: 시점 = 최신" in latest.cautions  # 2025 is blank


def test_periods_list_range_recent_and_anchor(households):
    base = {"item": {"codes": ["T1"]}, "dims": {REGION: {"codes": ["11"]}}}
    listed = select(households, **base, periods={"kind": "list", "periods": ["2019", "2023"]})
    assert periods_of(listed) == ["2023"] and "missing_years: 2019" in listed.cautions
    with pytest.raises(CalcRefused) as none:
        select(households, **base, periods={"kind": "list", "periods": ["2019"]})
    assert none.value.reason == "no_period"
    assert periods_of(select(households, **base, periods={"kind": "range", "start": "2015", "end": "2023"})) == ["2015", "2020", "2023"]
    assert periods_of(select(households, **base, periods={"kind": "recent", "n": 2})) == ["2023", "2024"]
    assert periods_of(select(households, **base, periods={"kind": "anchor", "period": "2024", "offset": -1})) == ["2023"]
    assert periods_of(select(households, **base, periods={"kind": "anchor", "period": "2020", "to_latest": True})) == [
        "2020", "2023", "2024", "2025"]


def test_events_come_from_the_calendar_or_the_lookup(households):
    events = load_events()
    assert events["코로나19"] == "2020-01" and events["외환위기"] == "1997-11" and events["리먼사태"] == "2008-09"
    base = {"item": {"codes": ["T1"]}, "dims": {REGION: {"codes": ["11"]}}}
    with pytest.raises(NeedsClarify, match="언제인지"):
        select(households, **base, periods={"kind": "anchor", "event": "가상 사건", "offset": 1})
    got = resolve_select(households, SelectStep(id="s", **base, periods={"kind": "anchor", "event": "가상 사건", "offset": 1}),
                         lookup=lambda name: "2019")
    assert periods_of(got) == ["2020"] and "event_source: model (가상 사건 = 2019)" in got.cautions


def test_monthly_tables_span_and_offsets(monthly):
    assert periods_of(select(monthly, periods={"kind": "span", "year": "2024", "part": "H1"})) == [f"2024{m:02d}" for m in range(1, 7)]
    assert periods_of(select(monthly, periods={"kind": "anchor", "period": "202401", "offset": -1})) == ["202312"]
    assert periods_of(select(monthly, periods={"kind": "anchor", "period": "2024", "offset": -1, "unit": "year"})) == ["2023"]
    assert periods_of(select(monthly, periods={"kind": "list", "periods": ["2023"]})) == ["2023"]  # the year the table holds
    # a range of years uses the yearly values a monthly table holds; the months of a year are a span
    assert periods_of(select(monthly, periods={"kind": "range", "start": "2023", "end": "2024"})) == ["2023", "2024"]
    assert periods_of(select(monthly, periods={"kind": "range", "start": "202311", "end": "202402"})) == ["202311", "202312", "202401", "202402"]


def test_map_then_rank_finds_the_fastest_growing_region(households):
    body = [{"id": "a", "op": "select", "item": {"codes": ["T1"]}, "dims": {REGION: {"codes": ["*"]}}, "periods": {"kind": "list", "periods": ["2015"]}},
            {"id": "b", "op": "select", "item": {"codes": ["T1"]}, "dims": {REGION: {"codes": ["*"]}}, "periods": {"kind": "list", "periods": ["2024"]}},
            {"id": "g", "op": "combine", "a": "b", "b": "a", "fn": "pct_change"}]
    p = PlanV3(steps=[{"id": "m", "op": "map", "axis": REGION, "codes": ["21", "11", "39"], "body": body, "result": "g"},
                      {"id": "k", "op": "rank", "of": "m", "pick": "max", "keep": 2},
                      {"id": "x", "op": "aggregate", "of": "m", "fn": "min"}], show=["k", "x"], present="extreme")
    run = run_plan(households, p)
    by_id = {f.id: f for f in run.facts}
    assert len(run.facts) == 9  # three facts per region
    winner = by_id[run.results["k"].picked["winner"]]
    assert winner.note == "서울특별시: 3개 중 가장 높음"  # the smallest base grows fastest in the made-up data
    expected = round((value("T1", "11", "2024") / value("T1", "11", "2015") - 1) * 100, 1)
    assert winner.value == expected
    assert [r["dims"][REGION] for r in run.results["k"].ranking] == ["서울특별시", "부산광역시", "제주특별자치도"]
    assert run.results["x"].id == run.results["m"].values[2].id  # 제주 has the lowest growth


def test_map_and_rank_contracts():
    body = [{"id": "a", "op": "select", "cells": ["x|y|1"]}]
    with pytest.raises(ValidationError, match="codes \\['\\*'\\]"):
        PlanV3(steps=[{"id": "m", "op": "map", "axis": REGION, "codes": ["11", "21"], "body": body, "result": "a"}], show=["m"], present="table")
    starred = [{"id": "m", "op": "select", "dims": {REGION: {"codes": ["*"]}}}]
    with pytest.raises(ValidationError, match="must differ"):
        PlanV3(steps=[{"id": "m", "op": "map", "axis": REGION, "codes": ["11", "21"], "body": starred, "result": "m"}], show=["m"], present="table")
    with pytest.raises(ValidationError, match="exactly one"):
        PlanV3(steps=[{"id": "k", "op": "rank", "candidates": ["a|b|1", "a|b|2"], "of": "k", "pick": "max"}], show=["k"], present="extreme")


def test_a_star_outside_a_map_is_refused(households):
    with pytest.raises(CalcRefused, match="map 안에서만"):
        select(households, dims={REGION: {"codes": ["*"]}}, periods={"kind": "list", "periods": ["2023"]})


def test_a_total_named_with_chong_wins_over_a_subtotal(tmp_path):
    rows = [{"period": "2024", "item_id": "T1", "item_name": "가상 인구", "unit": "명", "dim1_name": "거주외국인별",
             "dim1_code": code, "dim1_label": label, "value_raw": v, "value": float(v)}
            for code, label, v in (("A000", "합계", "845"), ("A0AA", "총인구", "13914"), ("A001", "외국인근로자", "300"))]
    catalog = {"survey_name": "가상", "organization": "가상", "table_name": "가상 외국인주민", "period": "년", "url": "https://example.invalid/f"}
    table = clean(load_table(write_table(tmp_path / "data", "FAKE_FOREIGN", rows, catalog)))
    got = select(table, periods={"kind": "list", "periods": ["2024"]})
    assert got.cells == ["FAKE_FOREIGN|T1|A0AA|2024"] and "default_applied: 거주외국인별 = 총인구" in got.cautions
