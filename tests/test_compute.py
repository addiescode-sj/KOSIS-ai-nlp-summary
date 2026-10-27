"""Calculation engine (T1-07) on the made-up tables, and the M1 golden check wiring (eval/compute_check.py)."""

import pytest
from pydantic import ValidationError

from conftest import write_table
from eval.compute_check import check, oracle_request
from eval.golden import GoldenItem
from eval.grade import evaluate
from kosis_chat.compute import CalcRefused, CalcRequest, compute
from kosis_chat.compute.request import Term
from kosis_chat.compute.roles import pick_of, reference_index
from kosis_chat.data.loader import load_table, sha256_of
from kosis_chat.data.preprocessing import clean

# FAKE_TBL_002 value = 1000 * (region no + 1) + 100 * item no + period no (conftest.household_rows).
REGION_NO = {"00": 0, "03": 1, "11": 2, "21": 3, "39": 4}
ITEM_NO = {"T0": 0, "T1": 1, "T2": 2}
PERIOD_NO = {"2015": 0, "2020": 1, "2023": 2, "2024": 3}
CATALOG = {"survey_name": "가상 조사", "organization": "가상 기관", "table_name": "가상 표", "period": "년", "url": "https://example.invalid/t"}


def ref(item, region, period):
    return f"FAKE_TBL_002|{item}|{region}|{period}"


def value(item, region, period):
    return 1000 * (REGION_NO[region] + 1) + 100 * ITEM_NO[item] + PERIOD_NO[period]


@pytest.fixture
def households(household_path):
    return clean(load_table(household_path))


def facts_by_id(computed):
    return {f.id: f for f in computed.result.facts}


def recomputes(computed):
    """Every derived fact equals its derivation over the other facts (what eval/grade.py checks)."""
    by_id = facts_by_id(computed)
    for f in computed.result.facts:
        if f.derivation:
            got = round(evaluate(f.derivation, {k: v.value for k, v in by_id.items() if v.value is not None}), f.round)
            assert abs(got - f.value) < 1e-9, (f.id, f.derivation, got, f.value)
    return True


def test_value_cell_fact(households):
    computed = compute(households, CalcRequest(type="value", cells=[ref("T1", "11", "2023")]))
    (fact,) = computed.result.facts
    assert fact.value == value("T1", "11", "2023") and fact.source_ref == ref("T1", "11", "2023")
    assert fact.unit == "가구" and fact.period == "2023" and fact.dims["행정구역별(읍면동)"] == "서울특별시"
    assert computed.main == ["f1"]


def test_trend_keeps_every_point_and_derives_difference_and_growth(households):
    cells = [ref("T1", "21", p) for p in ("2024", "2015", "2020", "2023")]  # any order: sorted by period
    computed = compute(households, CalcRequest(type="trend", cells=cells))
    by_id = facts_by_id(computed)
    assert [by_id[f"f{n}"].period for n in range(1, 5)] == ["2015", "2020", "2023", "2024"]
    diff, growth = by_id["f5"], by_id["f6"]
    assert diff.derivation == "f4 - f1" and diff.value == 3 and diff.round == 0 and diff.unit == "가구"
    assert growth.derivation == "(f4 / f1 - 1) * 100" and growth.round == 1 and growth.unit == "%"
    assert computed.main == ["f1", "f4", "f5", "f6"] and recomputes(computed)


def test_trend_of_a_percent_is_a_point_difference(tmp_path):
    rows = [{"period": p, "item_id": "T1", "item_name": "가상 비율", "unit": "%", "value_raw": raw, "value": float(raw)}
            for p, raw in (("2015", "48.3"), ("2016", "55"), ("2017", "60"))]
    path = write_table(tmp_path / "data", "FAKE_PCT", rows, CATALOG)
    table = clean(load_table(path))
    computed = compute(table, CalcRequest(type="trend", cells=[f"FAKE_PCT|T1||{p}" for p in ("2015", "2016", "2017")]))
    diff = facts_by_id(computed)["f4"]
    assert (diff.derivation, diff.value, diff.unit, diff.round) == ("f3 - f1", 11.7, "%p", 1)
    assert len(computed.result.facts) == 4  # no growth rate of a percentage


def test_compare_signs(households):
    a, b = ref("T1", "11", "2023"), ref("T1", "00", "2023")
    named = compute(households, CalcRequest(type="compare", cells=[b, a], reference=1))  # "전국은 서울보다"
    by_id = facts_by_id(named)
    assert by_id["f3"].derivation == "f1 - f2" and by_id["f3"].value == value("T1", "00", "2023") - value("T1", "11", "2023")
    assert by_id["f4"].derivation == "f1 / f2" and by_id["f4"].unit == "배"
    plain = facts_by_id(compute(households, CalcRequest(type="compare", cells=[b, a])))  # larger minus smaller
    assert plain["f3"].derivation == "f2 - f1" and plain["f3"].value > 0
    assert recomputes(named)


def test_share_uses_the_total_cell_and_sums_parts(households):
    computed = compute(households, CalcRequest(type="share", cells=[ref("T1", "21", "2023"), ref("T2", "21", "2023")],
                                               total=ref("T0", "21", "2023")))
    share = facts_by_id(computed)["f4"]
    assert share.derivation == "(f1 + f2) / f3 * 100" and share.unit == "%" and share.round == 1
    assert recomputes(computed)


def test_extreme_ranks_shares_and_keeps_runner_up_and_context(households):
    regions = ("11", "21", "39")
    computed = compute(households, CalcRequest(
        type="extreme", candidates=[ref("T1", r, "2023") for r in regions],
        candidate_totals=[ref("T0", r, "2023") for r in regions], pick="max", context=[ref("T1", "00", "2023")]))
    by_id = facts_by_id(computed)
    winner = by_id[computed.picked["winner"]]
    assert winner.note == "3개 중 가장 높음" and winner.derivation == "f1 / f2 * 100"
    assert by_id["f1"].source_ref == ref("T1", "11", "2023")  # 서울 has the largest 1인 share in the made-up data
    assert by_id[computed.picked["runner_up"]].note == "두 번째"
    assert by_id["f7"].source_ref == ref("T1", "00", "2023") and computed.main == ["f1", "f2", "f3"]
    assert recomputes(computed)
    plain = compute(households, CalcRequest(type="extreme", candidates=[ref("T1", r, "2023") for r in regions], pick="min"))
    assert facts_by_id(plain)["f1"].source_ref == ref("T1", "11", "2023")  # smallest value


def test_symbols_missing_cells_and_mixed_units_are_refused(households, pop_path):
    with pytest.raises(CalcRefused) as blank:
        compute(households, CalcRequest(type="value", cells=[ref("T1", "11", "2025")]))
    assert blank.value.reason == "missing_value"
    with pytest.raises(CalcRefused) as nowhere:
        compute(households, CalcRequest(type="value", cells=["FAKE_TBL_002|T9|11|2023"]))
    assert nowhere.value.reason == "class_mismatch"
    pop = clean(load_table(pop_path))
    with pytest.raises(CalcRefused) as units:
        compute(pop, CalcRequest(type="compare", cells=["FAKE_TBL_001|T1|0|2023", "FAKE_TBL_001|T2|0|2023"]))
    assert units.value.reason == "unit_mismatch"
    # A trend whose ends are symbols has fewer than two numbers.
    with pytest.raises(CalcRefused) as trend:
        compute(pop, CalcRequest(type="trend", cells=["FAKE_TBL_001|T1|1|2024", "FAKE_TBL_001|T1|1|2023"]))
    assert trend.value.reason == "missing_value"


def test_summing_rates_is_not_aggregatable(tmp_path):
    rows = [{"period": "2023", "item_id": i, "item_name": n, "unit": "%", "value_raw": v, "value": float(v)}
            for i, n, v in (("T1", "가상 출생률", "1.5"), ("T2", "가상 사망률", "2.5"), ("T0", "가상 전체율", "10"))]
    path = write_table(tmp_path / "data", "FAKE_RATE", rows, CATALOG)
    with pytest.raises(CalcRefused) as exc:
        compute(clean(load_table(path)), CalcRequest(type="share", cells=["FAKE_RATE|T1||2023", "FAKE_RATE|T2||2023"],
                                                     total="FAKE_RATE|T0||2023"))
    assert exc.value.reason == "not_aggregatable"


@pytest.mark.parametrize("payload", [
    {"type": "value", "cells": []}, {"type": "compare", "cells": ["a"]}, {"type": "share", "cells": ["a"]},
    {"type": "extreme", "candidates": ["a", "b"]}, {"type": "value", "cells": ["a"], "reference": 0},
    {"type": "extreme", "candidates": ["a", "b"], "candidate_totals": ["t"], "pick": "max"}, {"type": "value", "cells": ["a"], "x": 1},
])
def test_request_shape_is_checked(payload):
    with pytest.raises(ValidationError):
        CalcRequest.model_validate(payload)


@pytest.mark.parametrize("question, expected", [
    ("2020년 서울 합계출산율은 전국보다 얼마나 낮았어?", 1),
    ("2025년 경기도 인구밀도는 전국 평균의 몇 배인가요?", 1),
    ("2024년 부산과 대구 중 조사망률이 더 높은 곳은 어디고 차이는 얼마야?", None),
    ("서울의 출생아 수와 전국을 비교해줘", None),
])
def test_reference_words(question, expected):
    labels = ["서울특별시", "전국"] if "서울" in question else ["경기도", "전국"] if "경기" in question else ["부산광역시", "대구광역시"]
    assert reference_index(question, labels) == expected


def test_province_labels():
    from kosis_chat.compute.roles import is_province
    assert [is_province(x) for x in ("부산광역시", "경기도", "세종특별자치시", "동부", "전국", "읍부")] == [1, 1, 1, 0, 0, 0]


def test_pick_words():
    assert pick_of("2020년에 인구밀도가 제일 낮은 시도는?") == "min"
    assert pick_of("2023년 조출생률이 가장 높은 시도는?") == "max"


def golden(path, **fields):
    base = {"id": "G1", "kind": "answerable", "question": "질문", "table": str(path), "table_sha256": sha256_of(path)}
    return GoldenItem.model_validate(base | fields)


def test_m1_check_grades_engine_output_like_a_model_answer(household_path):
    share = golden(household_path, type="share", question="2023년 부산 1인과 2인 가구 비중은?", required_facts=[
        {"id": "f1", "value": value("T1", "21", "2023"), "source_ref": ref("T1", "21", "2023")},
        {"id": "f2", "value": value("T2", "21", "2023"), "source_ref": ref("T2", "21", "2023")},
        {"id": "f3", "value": value("T0", "21", "2023"), "source_ref": ref("T0", "21", "2023")},
        {"id": "f4", "value": 207.5, "derivation": "(f1 + f2) / f3 * 100", "round": 1, "unit": "%"}])
    row = check(share)
    assert row["verdict"] == "PASS", row["details"]
    wrong = golden(household_path, type="share", required_facts=[
        {"id": "f1", "value": value("T1", "21", "2023"), "source_ref": ref("T1", "21", "2023")},
        {"id": "f2", "value": value("T0", "21", "2023"), "source_ref": ref("T0", "21", "2023")},
        {"id": "f3", "value": 99.9, "derivation": "f1 / f2 * 100", "round": 1, "unit": "%"}])
    assert check(wrong)["error_types"] == ["wrong_value"]


def test_oracle_trend_takes_every_period_in_range(household_path):
    item = golden(household_path, type="trend", required_facts=[
        {"id": "f1", "value": value("T1", "11", "2015"), "source_ref": ref("T1", "11", "2015")},
        {"id": "f2", "value": value("T1", "11", "2023"), "source_ref": ref("T1", "11", "2023")},
        {"id": "f3", "value": 2, "derivation": "f2 - f1", "round": 0}])
    request = oracle_request(item, clean(load_table(household_path)))
    assert request.cells == [ref("T1", "11", p) for p in ("2015", "2020", "2023")]
    assert check(item)["verdict"] == "PASS"


def test_oracle_extreme_over_regions_excludes_totals(household_path):
    item = golden(household_path, type="extreme", question="2023년 1인 가구가 가장 많은 시도는?",
                  scope={"dims": {"행정구역별(읍면동)": ["00", "11", "21", "39"]}, "periods": ["2023", "2023"]},
                  required_facts=[{"id": "f1", "value": value("T1", "39", "2023"), "source_ref": ref("T1", "39", "2023")}])
    request = oracle_request(item, clean(load_table(household_path)))
    assert request.pick == "max" and ref("T1", "00", "2023") not in request.candidates
    assert request.context == [ref("T1", "00", "2023")] and len(request.candidates) == 3
    assert check(item)["verdict"] == "PASS"


# ---------------------------------------------------------------- T0-03 v2: ops over periods


def seoul(*periods):
    return [ref("T1", "11", p) for p in periods]


@pytest.fixture
def flow(tmp_path):
    """FAKE_FLOW: 혼인건수 (a flow, summable over periods) and 조혼인율 (a rate), 2023~2025, no categories."""
    rows = [{"period": p, "item_id": "T1", "item_name": "혼인건수", "unit": "건", "value_raw": v, "value": float(v)} for p, v in (("2023", "10"), ("2024", "14"), ("2025", "12"))]
    rows += [{"period": p, "item_id": "T2", "item_name": "조혼인율", "unit": "천명당", "value_raw": v, "value": float(v)} for p, v in (("2023", "4.1"), ("2024", "4.4"), ("2025", "4.2"))]
    return clean(load_table(write_table(tmp_path / "data", "FAKE_FLOW", rows, CATALOG)))


def flow_ref(item, period):
    return f"FAKE_FLOW|{item}||{period}"


def test_avg_and_sum_over_periods(households, flow):
    avg = compute(households, CalcRequest(type="trend", op="avg", terms=[Term(cells=seoul("2015", "2020", "2023", "2024"), agg="avg")]))
    by_id = facts_by_id(avg)
    assert avg.op == "avg" and avg.main == ["f5"] and by_id["f5"].derivation == "(f1 + f2 + f3 + f4) / 4"
    assert by_id["f5"].value == 3101.5 and by_id["f5"].round == 1 and by_id["f5"].unit == "가구" and recomputes(avg)
    assert avg.terms[0].cell_ids == ["f1", "f2", "f3", "f4"] and avg.terms[0].periods == ("2015", "2024")
    total = compute(flow, CalcRequest(type="value", op="sum", terms=[Term(cells=[flow_ref("T1", "2023"), flow_ref("T1", "2024")], agg="sum")]))
    assert facts_by_id(total)["f3"].derivation == "f1 + f2" and facts_by_id(total)["f3"].value == 24 and total.main == ["f1", "f2", "f3"]
    assert facts_by_id(total)["f3"].round == 0 and "합계" in facts_by_id(total)["f3"].note


def test_two_period_groups_compared(households, flow):
    early, late = Term(cells=seoul("2015", "2020"), agg="avg"), Term(cells=seoul("2023", "2024"), agg="avg")
    pavg = compute(households, CalcRequest(type="trend", op="pavg", terms=[early, late]))
    by_id = facts_by_id(pavg)
    assert [by_id[i].value for i in pavg.main] == [3100.5, 3102.5, 2.0, 0.1] and by_id["f7"].derivation == "f6 - f3"
    assert by_id["f8"].derivation == "(f6 / f3 - 1) * 100" and by_id["f8"].unit == "%" and recomputes(pavg)
    terms = [Term(cells=[flow_ref("T1", "2023"), flow_ref("T1", "2024")], agg="sum"), Term(cells=[flow_ref("T1", "2025")])]
    with pytest.raises(ValidationError):  # psum compares two sums, not a sum and a cell
        CalcRequest(type="trend", op="psum", terms=terms)


def test_value_against_an_average(households):
    vsa = compute(households, CalcRequest(type="trend", op="vsa", terms=[Term(cells=seoul("2015", "2020", "2023"), agg="avg"), Term(cells=seoul("2024"))]))
    by_id = facts_by_id(vsa)
    assert vsa.main == ["f4", "f5", "f6", "f7"] and by_id["f4"].value == 3101.0 and by_id["f5"].source_ref == ref("T1", "11", "2024")
    assert (by_id["f6"].derivation, by_id["f6"].value) == ("f5 - f4", 2.0) and by_id["f7"].derivation == "(f5 / f4 - 1) * 100"
    assert by_id["f8"].derivation == "f5 / f4 * 100" and by_id["f8"].value == 100.1 and "수준" in by_id["f8"].note and recomputes(vsa)


def test_share_change_in_points(households):
    terms = [Term(cells=[ref("T1", "00", "2015")], total=ref("T0", "00", "2015")), Term(cells=[ref("T1", "00", "2024")], total=ref("T0", "00", "2024"))]
    shr = compute(households, CalcRequest(type="share", op="shr", terms=terms))
    by_id = facts_by_id(shr)
    assert shr.main == ["f3", "f6", "f7"] and by_id["f3"].derivation == "f1 / f2 * 100" and by_id["f6"].derivation == "f4 / f5 * 100"
    assert by_id["f7"].derivation == "f6 - f3" and by_id["f7"].unit == "%p" and recomputes(shr)


def test_summing_a_stock_or_a_rate_over_periods_is_refused(households, flow):
    with pytest.raises(CalcRefused) as err:  # 1인 (가구) is a stock: the same households would be counted again
        compute(households, CalcRequest(type="trend", op="sum", terms=[Term(cells=seoul("2015", "2020"), agg="sum")]))
    assert err.value.reason == "not_aggregatable"
    avg = compute(households, CalcRequest(type="trend", op="avg", terms=[Term(cells=seoul("2015", "2020"), agg="avg")]))
    assert avg.main == ["f1", "f2", "f3"]  # an average of a stock is fine
    with pytest.raises(CalcRefused) as err:  # a rate summed over periods
        compute(flow, CalcRequest(type="trend", op="sum", terms=[Term(cells=[flow_ref("T2", "2023"), flow_ref("T2", "2024")], agg="sum")]))
    assert err.value.reason == "not_aggregatable"
    psum = compute(flow, CalcRequest(type="trend", op="psum", terms=[Term(cells=[flow_ref("T1", "2023"), flow_ref("T1", "2024")], agg="sum"), Term(cells=[flow_ref("T1", "2024"), flow_ref("T1", "2025")], agg="sum")]))
    assert [facts_by_id(psum)[i].value for i in psum.main] == [24, 26, 2, 8.3] and recomputes(psum)


@pytest.mark.parametrize("payload", [
    {"type": "trend", "op": "avg", "terms": [{"cells": ["a"], "agg": "avg"}]},  # an average of one cell
    {"type": "trend", "op": "avg", "terms": [{"cells": ["a", "b"], "agg": "sum"}]},  # wrong aggregate
    {"type": "trend", "op": "pavg", "terms": [{"cells": ["a", "b"], "agg": "avg"}]},  # one group for two
    {"type": "trend", "op": "vsa", "terms": [{"cells": ["a"]}, {"cells": ["b", "c"], "agg": "avg"}]},  # the average comes first
    {"type": "value", "op": "shr", "terms": [{"cells": ["a"], "total": "t"}, {"cells": ["b"], "total": "u"}]},  # shr is a share question
    {"type": "share", "op": "shr", "terms": [{"cells": ["a"], "total": "t"}, {"cells": ["b", "c"], "total": "u"}]},  # different parts
    {"type": "trend", "op": "sum", "cells": ["a", "b"], "terms": [{"cells": ["a", "b"], "agg": "sum"}]},  # cells and terms together
    {"type": "trend", "terms": [{"cells": ["a", "b"], "agg": "sum"}]},  # terms without an op
    {"type": "trend", "op": "avg", "terms": [{"cells": ["a", "b"], "agg": "avg", "total": "t"}]},  # aggregate and share at once
])
def test_op_request_shape_is_checked(payload):
    with pytest.raises(ValidationError):
        CalcRequest.model_validate(payload)


def test_oracle_request_builds_the_op_from_the_golden_derivations(household_path):
    item = golden(household_path, id="S-CX-PAVG-09", type="trend", required_facts=[
        {"id": "f1", "value": value("T1", "11", "2015"), "source_ref": ref("T1", "11", "2015")},
        {"id": "f2", "value": value("T1", "11", "2020"), "source_ref": ref("T1", "11", "2020")},
        {"id": "f3", "value": 3100.5, "derivation": "(f1 + f2) / 2", "round": 1},
        {"id": "f4", "value": value("T1", "11", "2023"), "source_ref": ref("T1", "11", "2023")},
        {"id": "f5", "value": value("T1", "11", "2024"), "source_ref": ref("T1", "11", "2024")},
        {"id": "f6", "value": 3102.5, "derivation": "(f4 + f5) / 2", "round": 1},
        {"id": "f7", "value": 2.0, "derivation": "f6 - f3", "round": 1}])
    request = oracle_request(item, clean(load_table(household_path)))
    assert request.op == "pavg" and [t.cells for t in request.terms] == [seoul("2015", "2020"), seoul("2023", "2024")]
    assert check(item)["verdict"] == "PASS", check(item)["details"]
