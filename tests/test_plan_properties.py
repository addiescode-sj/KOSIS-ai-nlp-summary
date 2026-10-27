"""Property tests for v3 plans (T0-04, hypothesis): random small tables and plans, rules that must always hold.

1. Every derived fact recomputes (eval/grade.evaluate) to its own value, and blocks equal a direct computation.
2. The gap and growth plans the rules build (S5) equal the same numbers worked out by hand.
3. A random plan is either rejected by the schema or runs to facts or a CalcRefused, never another error,
   and what runs keeps the meaning checks (semantics.py): the made-up table is a stock, so no sum over periods.
4. Periods named by meaning resolve to table periods only, ascending and once each, or to a refusal / ask-back.
"""

import tempfile
from pathlib import Path

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from pydantic import ValidationError

from conftest import write_table
from eval.grade import evaluate
from kosis_chat.compute import CalcRefused, build, compute
from kosis_chat.compute.plan_engine import PlanUnsupported, run_plan
from kosis_chat.compute.resolve import NeedsClarify, resolve_select
from kosis_chat.data.loader import load_table
from kosis_chat.data.preprocessing import clean

from kosis_ai.schemas import PlanV3, SelectStep  # isort: skip
from test_compute import CATALOG

PERIODS = ["2015", "2016", "2017", "2018", "2019"]
REGIONS = (("00", "전국"), ("11", "서울특별시"), ("21", "부산광역시"))
CHECK = settings(max_examples=40, deadline=None, suppress_health_check=[HealthCheck.too_slow])

# A value as the table writes it: a whole number, or one or two decimals; zero and negatives included on purpose.
values = st.one_of(st.integers(-500, 50_000).map(str),
                   st.decimals(-50, 500, places=1, allow_nan=False, allow_infinity=False).map(str),
                   st.decimals(0, 100, places=2, allow_nan=False, allow_infinity=False).map(str))
grids = st.lists(values, min_size=len(PERIODS) * len(REGIONS), max_size=len(PERIODS) * len(REGIONS))


def table_of(grid: list[str], unit: str = "명"):
    rows, n = [], 0
    for code, label in REGIONS:
        for period in PERIODS:
            raw = grid[n]
            n += 1
            rows.append({"period": period, "item_id": "T1", "item_name": "가상 인구", "unit": unit, "dim1_name": "행정구역별",
                         "dim1_code": code, "dim1_label": label, "value_raw": raw, "value": float(raw)})
    with tempfile.TemporaryDirectory() as tmp:
        return clean(load_table(write_table(Path(tmp) / "data", "FAKE_PROP", rows, CATALOG)))


def ref(region: str, period: str) -> str:
    return f"FAKE_PROP|T1|{region}|{period}"


def recomputes(facts) -> None:
    known = {f.id: f.value for f in facts if f.value is not None}
    for f in facts:
        if f.derivation:
            assert abs(round(evaluate(f.derivation, known), f.round) - f.value) < 1e-9, f


@CHECK
@given(grid=grids, region=st.sampled_from([c for c, _ in REGIONS]),
       fn=st.sampled_from(["diff", "ratio", "pct_change", "index", "cagr"]))
def test_blocks_equal_direct_computation_and_recompute(grid, region, fn):
    table = table_of(grid)
    steps = [{"id": "s", "op": "select", "cells": [ref(region, p) for p in PERIODS]},
             {"id": "m", "op": "aggregate", "of": "s", "fn": "avg"},
             {"id": "a", "op": "aggregate", "of": "s", "fn": "first"},
             {"id": "b", "op": "aggregate", "of": "s", "fn": "last"},
             {"id": "c", "op": "combine", "a": "b", "b": "a", "fn": fn, **({"years": 4} if fn == "cagr" else {})}]
    try:
        run = run_plan(table, PlanV3(steps=steps, show=["m", "c"], present="trend"))
    except CalcRefused:
        return  # a zero base or a non-positive value for a growth rate: refused, which is allowed
    recomputes(run.facts)
    nums = [float(v) for v in grid[[c for c, _ in REGIONS].index(region) * len(PERIODS):][: len(PERIODS)]]
    by_id = {f.id: f for f in run.facts}
    mean = by_id[run.main[0]]
    assert abs(mean.value - round(sum(nums) / len(nums), mean.round)) < 1e-9
    last, first = nums[-1], nums[0]
    direct = {"diff": last - first, "ratio": last / first if first else None,
              "pct_change": (last / first - 1) * 100 if first else None, "index": last / first * 100 if first else None,
              "cagr": ((last / first) ** (1 / 4) - 1) * 100 if first > 0 and last > 0 else None}[fn]
    got = by_id[run.main[1]]
    assert direct is not None and abs(got.value - round(direct, got.round)) < 1e-9



@CHECK
@given(grid=grids, a=st.sampled_from(["11", "21"]), early=st.sampled_from(PERIODS[:2]), late=st.sampled_from(PERIODS[2:]),
       amount=st.booleans(), decline=st.booleans())
def test_gap_and_growth_plans_equal_direct_computation(grid, a, early, late, amount, decline):
    """The two S5 shapes (build.gap, build.growth) against the same numbers worked out by hand."""
    table = table_of(grid)
    b = "21" if a == "11" else "11"
    num = {(r, p): float(grid[[c for c, _ in REGIONS].index(r) * len(PERIODS) + PERIODS.index(p)]) for r in (a, b) for p in (early, late)}
    for built in (build.gap((ref(a, early), ref(b, early)), (ref(a, late), ref(b, late))),
                  build.growth((ref(a, early), ref(a, late)), (ref(b, early), ref(b, late)), amount=amount, decline=decline)):
        try:
            computed = compute(table, built)
        except CalcRefused:
            continue  # a zero base for a growth rate
        recomputes(computed.result.facts)
        got = {f.id: f for f in computed.result.facts}[computed.main[-1]]
        if built.pattern == "gap":
            direct = abs(num[a, late] - num[b, late]) - abs(num[a, early] - num[b, early])
        else:
            change = [(num[r, late] - num[r, early]) if amount else (num[r, late] / num[r, early] - 1) * 100 for r in (a, b)]
            direct = min(change) - max(change) if decline else max(change) - min(change)
        # the gap of rounded growth rates may differ from the exact one by one rounding step
        assert abs(got.value - round(direct, got.round)) <= (0 if amount or built.pattern == "gap" else 1.5) * 10 ** -got.round + 1e-9


step_ids = st.sampled_from(["s1", "s2", "x", "y", "z"])
random_steps = st.lists(st.one_of(
    st.builds(lambda i, cells: {"id": i, "op": "select", "cells": cells}, step_ids,
              st.lists(st.sampled_from([ref(r, p) for r, _ in REGIONS for p in PERIODS]), min_size=1, max_size=3)),
    st.builds(lambda i, of, fn: {"id": i, "op": "aggregate", "of": of, "fn": fn}, step_ids, step_ids,
              st.sampled_from(["sum", "avg", "min", "max", "first", "last"])),
    st.builds(lambda i, a, b, fn, opt: {"id": i, "op": "combine", "a": a, "b": b, "fn": fn, "optional": opt,
                                        **({"years": 3} if fn == "cagr" else {}), **({"scale": 1000} if fn == "per" else {})},
              step_ids, step_ids, step_ids,
              st.sampled_from(["diff", "ratio", "pct_change", "pp_diff", "share", "per", "cagr", "index"]), st.booleans()),
    st.builds(lambda i, of, v: {"id": i, "op": "find", "of": of, "cmp": ">", "value": v}, step_ids, step_ids,
              st.integers(-100, 1000)),
), min_size=1, max_size=6)


@CHECK
@given(grid=grids, steps=random_steps, show=st.lists(step_ids, min_size=1, max_size=3, unique=True))
def test_random_plans_are_rejected_or_run_cleanly(grid, steps, show):
    try:
        plan = PlanV3(steps=steps, show=show, present="value")
    except ValidationError:
        return
    try:
        run = run_plan(table_of(grid), plan)
    except (CalcRefused, PlanUnsupported):
        return
    recomputes(run.facts)
    ids = {f.id for f in run.facts}
    assert set(run.main) <= ids and len(ids) == len(run.facts)
    sums = [f for f in run.facts if f.derivation and "+" in f.derivation and "/" not in f.derivation]
    assert not sums, sums  # a population (count_stock) is never summed over periods


years = st.sampled_from(["2010", "2014", "2015", "2017", "2019", "2021"])
period_specs = st.one_of(
    st.builds(lambda ps: {"kind": "list", "periods": ps}, st.lists(years, min_size=1, max_size=4)),
    st.builds(lambda a, b: {"kind": "range", "start": min(a, b), "end": max(a, b)}, years, years),
    st.builds(lambda n: {"kind": "recent", "n": n}, st.integers(1, 8)),
    st.builds(lambda p, o, latest: {"kind": "anchor", "period": p, "offset": o, "to_latest": latest},
              years, st.integers(-3, 3), st.booleans()),
    st.builds(lambda e, o: {"kind": "anchor", "event": e, "offset": o}, st.sampled_from(["코로나", "IMF", "금융위기"]),
              st.integers(-12, 12)),
)


@CHECK
@given(grid=grids, spec=period_specs, region=st.sampled_from([c for c, _ in REGIONS]))
def test_periods_by_meaning_resolve_to_table_periods(grid, spec, region):
    table = table_of(grid)
    step = SelectStep(id="s", item={"codes": ["T1"]}, dims={"행정구역별": {"codes": [region]}}, periods=spec)
    try:
        got = resolve_select(table, step)
    except (CalcRefused, NeedsClarify):
        return
    periods = [r.rsplit("|", 1)[-1] for r in got.cells]
    assert periods == sorted(set(periods)) and set(periods) <= set(PERIODS)
    if spec["kind"] == "recent":
        assert len(periods) == min(spec["n"], len(PERIODS))
