"""The callers of the cell store that T1-31 S2 reshaped, and how many store requests a question makes.

A file store (T1-31 S3) answers each request with one filtered read of the Parquet file, so a question must make a
small number of requests that does not grow with the cells it touches. Counting wraps the memory store and counts
the requests per method; the tests below pin the paths a mutation run found unguarded (the multi-table caveats, the
value series, caveats beyond the main facts, the item-named flag, the numeric periods of the rules' lookup).
"""

from __future__ import annotations

from collections import Counter
from dataclasses import replace
from types import SimpleNamespace

import pytest

from kosis_chat import calc
from kosis_chat.analyze import blocks
from kosis_chat.analyze.rules import Unresolved, _lookup, _one_period, build_plan, rule_plan
from kosis_chat.analyze.schemas import Plan
from kosis_chat.analyze.write import place
from kosis_chat.compose.caveats import caveats_of
from kosis_chat.compute import engine
from kosis_chat.compute.context import scenario_alternatives
from kosis_chat.compute.plan_engine import run_plan
from kosis_chat.compute.schemas import Computed
from kosis_chat.graph.multi import _Cells, caveats_node
from kosis_chat.judge.rules import judge
from kosis_chat.prepare.clean import clean_cell
from kosis_chat.prepare.index import ITEM_AXIS, index_of
from kosis_chat.prepare.schemas import Cell, TableInput, TableMeta
from kosis_chat.prepare.store import _STORE_CACHE, CellStore, MemoryStore
from kosis_chat.schemas.resources import MultiResources, load_resources

from kosis_ai.schemas import (CombineStep, Fact, MapStep, PeriodSpec, PlanV3, RankStep, SelectStep, StatResult,  # isort: skip
                              TargetSpec)

REGIONS = [("00", "전국"), ("11", "서울특별시"), ("21", "부산광역시"), ("22", "대구광역시"), ("23", "인천광역시"),
           ("24", "광주광역시"), ("25", "대전광역시"), ("26", "울산광역시"), ("31", "경기도"), ("32", "강원도"),
           ("33", "충청북도"), ("34", "충청남도"), ("35", "전라북도"), ("36", "전라남도"), ("37", "경상북도"),
           ("38", "경상남도"), ("39", "제주특별자치도")]
SEXES = [("0", "계"), ("1", "남자"), ("2", "여자")]
SYMBOL_KINDS = {"-": "dash", "x": "x_secret", "560p": "other_text"}


def _meta(tbl_id: str) -> TableMeta:
    return TableMeta(tbl_id=tbl_id, tbl_nm="가상 인구 표", survey_nm="가상 조사", org_nm="가상 기관", period_type="Y",
                     source_url="", fetched_at="2026-11-20", input_format="parquet", file_sha256="x")


def _cell(tbl_id: str, item: tuple[str, str], dims: dict[str, tuple[str, str]], period: str, raw: str,
          unit: str = "명") -> Cell:
    kind = SYMBOL_KINDS.get(raw) or ("other_text" if raw.endswith("p") else "ok")  # "1100p": provisional
    codes = "|".join(code for code, _ in dims.values())
    return Cell(item=item[1], dims={n: label for n, (_, label) in dims.items()}, period_raw=period, value_raw=raw,
                unit=unit, source_ref=f"{tbl_id}|{item[0]}|{codes}|{period}",
                dim_codes={n: code for n, (code, _) in dims.items()}, item_code=item[0], value_kind=kind,
                source_value=float(raw.replace(",", "")) if kind == "ok" else None, unit_source="given")


def _population(years: int, tbl_id: str = "FAKE_S2", regions: list[tuple[str, str]] = REGIONS) -> TableInput:
    """가상 인구 by 행정구역 (전국 and 16 regions) and 성별, the last `years` years up to 2024; every value a number."""
    cells = []
    for year in range(2025 - years, 2025):
        for region in regions:
            for sex in SEXES:
                value = 1000 + year + int(region[0]) * 7 + int(sex[0])
                cells.append(_cell(tbl_id, ("T1", "가상 인구"), {"행정구역별": region, "성별": sex}, str(year), str(value)))
    return TableInput(meta=_meta(tbl_id), cells=tuple(cells))


class Counting:
    """A CellStore that answers from a MemoryStore and counts the requests it gets, per method."""

    METHODS = frozenset(n for n in vars(CellStore) if not n.startswith("_"))

    def __init__(self, table: TableInput) -> None:
        self.inner = MemoryStore(table.cells)
        self.calls: Counter[str] = Counter()

    def __getattr__(self, name: str):
        found = getattr(self.inner, name)
        if name not in self.METHODS:
            return found

        def counted(*args, **kwargs):
            self.calls[name] += 1
            return found(*args, **kwargs)
        return counted

    @property
    def total(self) -> int:
        return sum(self.calls.values())


def counting(table: TableInput) -> Counting:
    """Make a counting store the table's store (store_of keeps one per TableInput, in its caches)."""
    store = Counting(table)
    assert table.caches.get(_STORE_CACHE, lambda: store) is store, "the table's store was asked for before"
    return store


# ---------------------------------------------------------------- request counts


QUESTIONS = {
    "value": "2023년 서울특별시 가상 인구는?",  # reads one cell, and its series of every year for the chart
    "value_default": "서울특별시 가상 인구는?",  # the latest year with a number
    "trend": "서울특별시 가상 인구 추이는?",
    "extreme_region": "2023년 가상 인구가 가장 많은 곳은?",  # ranks 16 regions
    "extreme_year": "서울특별시 가상 인구가 가장 많은 해는?",  # ranks every year
    "compare": "2023년 서울특별시와 부산광역시 가상 인구 차이는?",
    "average": "2020년부터 2023년까지 서울특별시 가상 인구 평균은?",
}
MAX_REQUESTS = 7  # measured (T1-31 S2): 2 to 7 per question in calc._one, up to 8 through the whole graph (no
# empty requests); before S2 one request per cell read (up to 103)


@pytest.mark.parametrize("kind", QUESTIONS)
def test_a_question_makes_a_few_store_requests_whatever_the_table_size(kind):
    totals = []
    for years in (5, 25):
        res = load_resources(_population(years))
        store = counting(res.table)
        result = calc._one(QUESTIONS[kind], res, None)
        assert result.status == "ok", result
        totals.append(store.total)
        assert store.total <= MAX_REQUESTS, (kind, years, dict(store.calls))
    assert totals[0] == totals[1], (kind, totals)  # five times the years, as many requests


def test_a_ranking_reads_its_candidates_in_one_request():
    res = load_resources(_population(5))
    store = counting(res.table)
    result = calc._one(QUESTIONS["extreme_region"], res, None)
    assert len(result.computed.ranking) == len(REGIONS) - 1  # every region but 전국
    assert store.calls == {"cells_at": 1, "by_refs": 2}  # the rules' lookup; the rank; the context cell (전국)


def test_the_judge_asks_for_the_target_summary_at_most_once_and_only_when_a_rule_needs_it():
    res = load_resources(_population(5))
    store = counting(res.table)
    assert judge("2023년 서울특별시 가상 인구는?", res.table, res.index).outcome == "answered"
    assert store.calls == {"summary": 1}  # _no_cell and _symbol both look: one request
    store.calls.clear()
    judge("서울특별시 가상 인구가 가장 많은 해는?", res.table, res.index)  # asks for a period: no default to set
    assert store.calls == {}
    store.calls.clear()
    applied = judge("서울특별시 가상 인구는?", res.table, res.index).default_applied
    assert store.calls == {"summary": 1} and applied == {"period": "latest (2024)"}


def test_calc_does_not_summarize_a_question_that_names_its_period():
    res = load_resources(_population(5))
    store = counting(res.table)
    calc._one(QUESTIONS["value"], res, None)
    assert "summary" not in store.calls


# ---------------------------------------------------------------- the callers a mutation run left unguarded


def test_value_series_is_every_number_of_the_cell_series_by_period():
    seoul = {"행정구역별": REGIONS[1], "성별": SEXES[0]}
    busan = {"행정구역별": REGIONS[2], "성별": SEXES[0]}
    cells = [_cell("FAKE_S2", ("T1", "가상 인구"), seoul, p, raw) for p, raw in
             (("2024", "1,300"), ("2021", "1,000"), ("2023", "-"), ("2022", "1,100.5"))]
    cells += [_cell("FAKE_S2", ("T1", "가상 인구"), busan, p, "900") for p in ("2021", "2024")]
    table = TableInput(meta=_meta("FAKE_S2"), cells=tuple(cells))  # raw: not cleaned, periods out of order
    expected = [{"period": "2021", "value": 1000.0}, {"period": "2022", "value": 1100.5},
                {"period": "2024", "value": 1300.0}]  # 2023 holds a symbol
    assert engine._series(table, cells[0]) == expected
    computed = engine.compute(table, blocks.value(cells[0].source_ref, []))
    assert computed.series == expected
    assert engine._series(table, cells[-1]) == [{"period": p, "value": 900.0} for p in ("2021", "2024")]


def _computed(facts: list[Fact], main: list[str]) -> Computed:
    return Computed(StatResult(facts=facts), main)


def _fact(fact_id: str, cell: Cell) -> Fact:
    c = clean_cell(cell)
    return Fact(id=fact_id, value=c.value, unit=c.unit, period=c.period_raw, dims=dict(c.dims), source_ref=c.source_ref)


def test_caveats_look_up_every_shown_fact_not_only_the_main_ones():
    seoul = {"행정구역별": REGIONS[1], "성별": SEXES[0]}
    cells = [_cell("FAKE_S2", ("T1", "가상 인구"), seoul, p, raw) for p, raw in
             (("2022", "1000"), ("2023", "1100p"), ("2024", "1200p"))]
    table = load_resources(TableInput(meta=_meta("FAKE_S2"), cells=tuple(cells))).table
    computed = _computed([_fact(f"f{n}", c) for n, c in enumerate(cells, 1)], ["f1"])
    found = caveats_of(computed, table, [], ["f2"])  # f2 is shown as context, f3 is not shown
    assert [(c.kind, c.fact_ids) for c in found] == [("provisional", ["f2"])]


def test_multi_table_caveats_ask_every_table_and_the_later_table_wins():
    sex = {"성별": SEXES[0]}
    item = ("T1", "가상 인구")
    first = [_cell("FAKE_A", item, sex, "2022", "1000"), _cell("FAKE_A", item, sex, "2023", "1100p")]
    second = [_cell("FAKE_B", item, sex, "2024", "1200p"), _cell("FAKE_B", item, sex, "2021", "900")]
    # a ref the two tables share: 900 (a number) in the first table, 1050p (provisional) in the second
    shared = [_cell("FAKE_A", item, sex, "2021", "900"), _cell("FAKE_A", item, sex, "2021", "1050p")]
    a = TableInput(meta=_meta("FAKE_A"), cells=(*first, shared[0]))
    b = TableInput(meta=_meta("FAKE_B"), cells=(*second, shared[1]))
    res = MultiResources((load_resources(a), load_resources(b)))
    facts = [_fact("f1", first[1]), _fact("f2", second[0]), _fact("f3", shared[0]), _fact("f4", first[0])]
    out = caveats_node(res)(SimpleNamespace(computed=_computed(facts, ["f1", "f2", "f3", "f4"])))
    provisional = [c.fact_ids for c in out["caveats_out"] if c.kind == "provisional"]
    # f1 only in the first table, f2 only in the second, f3 shared (the second table's cell), f4 a number
    assert provisional == [["f1"], ["f2"], ["f3"]]
    refs = (f.source_ref for f in facts)  # any iterable, read once: every table is asked for every ref
    assert {r: c.value_raw for r, c in _Cells(res).by_refs(refs).items()} == {
        "FAKE_A|T1|0|2023": "1100p", "FAKE_B|T1|0|2024": "1200p", "FAKE_A|T1|0|2021": "1050p",
        "FAKE_A|T1|0|2022": "1000"}


def test_facts_leave_out_an_item_that_only_wraps_a_basic_item_category():
    basic = [("A", "가상 출생아수"), ("B", "가상 사망자수")]
    cells = [_cell("FAKE_S2", ("T10", "가상 통계"), {"기본항목별": b, "성별": s}, "2023", "100") for b in basic for s in SEXES]
    table = TableInput(meta=_meta("FAKE_S2"), cells=tuple(cells))
    assert engine._item_named(table) is False and index_of(table).item_named is False
    fact = engine.compute(table, blocks.value(cells[0].source_ref, [])).result.facts[0]
    assert fact.dims == {"기본항목별": "가상 출생아수", "성별": "계"}
    other = _cell("FAKE_S2", ("T11", "가상 기타"), {"기본항목별": basic[0], "성별": SEXES[0]}, "2023", "5")
    two = TableInput(meta=_meta("FAKE_S2"), cells=(*cells, other))
    assert engine._item_named(two) is True
    fact = engine.compute(two, blocks.value(cells[0].source_ref, [])).result.facts[0]
    assert fact.dims == {"기본항목별": "가상 출생아수", "성별": "계", "항목": "가상 통계"}


def test_lookup_numeric_periods_skip_symbols_and_the_rules_read_the_latest_number():
    seoul = {"행정구역별": REGIONS[1], "성별": SEXES[0]}
    busan = {"행정구역별": REGIONS[2], "성별": SEXES[0]}
    cells = [_cell("FAKE_S2", ("T1", "가상 인구"), seoul, p, raw) for p, raw in
             (("2021", "1000"), ("2022", "x"), ("2023", "1100"), ("2024", "-"))]
    cells += [_cell("FAKE_S2", ("T1", "가상 인구"), busan, p, raw) for p, raw in
              (("2021", "900"), ("2022", "950"), ("2023", "-"), ("2024", "990"))]
    table = load_resources(TableInput(meta=_meta("FAKE_S2"), cells=tuple(cells))).table
    lookup = _lookup(table)
    fixed = {ITEM_AXIS: "T1", "행정구역별": "11", "성별": "0"}
    other = {**fixed, "행정구역별": "21"}
    assert lookup.numeric_periods(fixed) == ["2021", "2023"]
    assert lookup.all_periods(fixed) == ["2021", "2022", "2023", "2024"]
    assert lookup.numeric_periods_of([fixed, other, {**fixed, "행정구역별": "99"}]) == [
        ["2021", "2023"], ["2021", "2022", "2024"], []]
    plan = SimpleNamespace(periods=[])
    assert _one_period(plan, lookup, [fixed]) == "2023"
    assert _one_period(plan, lookup, [fixed, other]) == "2021"  # the latest year both have a number for
    index = index_of(table)
    built = build_plan(rule_plan("서울특별시 가상 인구는?", index), index, table)
    assert built.plan.steps[0].cells == ["FAKE_S2|T1|11|0|2023"]


def _map_plan(regions: list[tuple[str, str]]) -> PlanV3:
    """What the model writes for '2024년 가상 인구가 가장 많은 지역은?': a map over every region, then a rank."""
    body = SelectStep(id="s", item=TargetSpec(codes=["T1"]), dims={"행정구역별": TargetSpec(codes=["*"]),
                                                                  "성별": TargetSpec(codes=["0"])},
                      periods=PeriodSpec(kind="list", periods=["2024"]))
    codes = [code for code, _ in regions if code != "00"]
    return PlanV3(steps=[MapStep(id="m", axis="행정구역별", codes=codes, body=[body], result="s"),
                         RankStep(id="k", of="m", pick="max", candidates=[], totals=[])], show=["k"], present="extreme")


def test_a_map_makes_as_many_store_requests_for_any_number_of_codes():
    for regions in (REGIONS[:7], REGIONS):
        res = load_resources(_population(5, regions=regions))
        store = counting(res.table)
        placed = place("q", _map_plan(regions), res.table, res.index)
        assert placed.computed is not None, placed.problems
        facts = placed.computed.result.facts  # one per region, in the map's order; the last region is the largest
        assert [(f.source_ref, f.value) for f in facts] == [
            (f"FAKE_S2|T1|{code}|0|2024", 1000 + 2024 + int(code) * 7) for code, _ in regions[1:]]
        assert facts[-1].note == f"{regions[-1][1]}: {len(regions) - 1}개 중 가장 높음"
        # the map's selects in one request (every region at once), the facts of the query in one more
        assert store.calls == {"cells_at": 1, "by_refs": 1}, (len(regions), dict(store.calls))


def _households(shares: dict[tuple[str, str], tuple[int, int]]) -> TableInput:
    """가구 by 행정구역 and item: 일반가구_계 (the item total) and 1인, from {region: (total, one person)}."""
    cells = []
    for region, (total, one) in shares.items():
        for item, value in ((("T0", "일반가구_계"), total), (("T1", "1인"), one), (("T2", "2인이상"), total - one)):
            cells.append(_cell("FAKE_HH", item, {"행정구역별": region}, "2023", str(value), unit="가구"))
    return TableInput(meta=_meta("FAKE_HH"), cells=tuple(cells))


def test_a_rule_ranking_of_shares_divides_each_candidate_by_its_own_item_total():
    # the order by share is neither the order by the part nor the order over the first region's total
    table = _households({REGIONS[0]: (1900, 600), REGIONS[1]: (1000, 300), REGIONS[2]: (400, 200),
                         REGIONS[3]: (500, 100)})
    result = calc._one("2023년 1인 가구 비중이 가장 높은 지역은?", load_resources(table), None)
    assert result.status == "ok", result
    assert [(r["dims"]["행정구역별"], r["value"]) for r in result.computed.ranking] == [
        ("부산광역시", 50.0), ("서울특별시", 30.0), ("대구광역시", 20.0)]


def _sex_table() -> TableInput:
    cells = [_cell("FAKE_S2", ("T1", "가상 인구"), {"성별": sex}, period, "100" if sex[0] == "0" else "40")
             for period in ("2020", "2021", "2022") for sex in SEXES if (period, sex[0]) != ("2020", "1")]
    return load_resources(TableInput(meta=_meta("FAKE_S2"), cells=tuple(cells))).table


@pytest.mark.parametrize("groups, raised", [
    ([["2020"], ["1999"]], "칸 없음 {'항목': 'T1', '성별': '1'} 2020"),  # the first group's problems only
    ([["2020"], []], "칸 없음 {'항목': 'T1', '성별': '1'} 2020"),  # the first group stops it before the empty one
    ([["2021"], []], IndexError),  # an empty group fails at its turn (group[-1]), as the group-by-group reads did
    ([[], ["2020"]], IndexError),
    ([[]], IndexError),
    ([], "shr: 비중 시점 0개"),
])
def test_shr_reads_the_groups_in_order_and_stops_at_the_first_problem(groups, raised):
    table = _sex_table()
    plan = Plan(question="비중", type="share", codes={"성별": ["1"]}, periods=[], pattern="shr", groups=groups)
    if raised is IndexError:
        with pytest.raises(IndexError):
            build_plan(plan, index_of(table), table)
    else:
        with pytest.raises(Unresolved) as exc:
            build_plan(plan, index_of(table), table)
        assert exc.value.what == [raised]


def _select(cells: list[str] | None = None) -> SelectStep:
    """A select of the given cells, else of 가상 인구 계 at 2023 by meaning."""
    if cells:
        return SelectStep(id="s", cells=cells, dims={})
    return SelectStep(id="s", item=TargetSpec(codes=["T1"]), dims={"성별": TargetSpec(codes=["0"])},
                      periods=PeriodSpec(kind="list", periods=["2023"]))


def _sex_cells(tbl_id: str, values: dict[str, str], ref_table: str | None = None) -> list[Cell]:
    """One cell per 성별 code at 2023; ref_table: the tbl_id the refs name (another table's, on purpose)."""
    out = []
    for code, label in SEXES:
        if code in values:
            c = _cell(tbl_id, ("T1", "가상 인구"), {"성별": (code, label)}, "2023", values[code])
            out.append(c if ref_table is None else replace(c, source_ref=c.source_ref.replace(tbl_id, ref_table, 1)))
    return out


def _tables(*tables: TableInput) -> dict[str, TableInput]:
    return {t.meta.tbl_id: t for t in tables}


def test_a_select_by_cells_reads_each_ref_in_the_table_it_names():
    a = TableInput(meta=_meta("FAKE_A"), cells=tuple(_sex_cells("FAKE_A", {"0": "100"})))
    b = TableInput(meta=_meta("FAKE_B"), cells=tuple(_sex_cells("FAKE_B", {"0": "200"})))
    plan = PlanV3(steps=[_select(["FAKE_A|T1|0|2023", "FAKE_B|T1|0|2023"])], show=["s"], present="value")
    run = run_plan(a, plan, tables=_tables(a, b))
    assert [(f.source_ref, f.value) for f in run.facts] == [("FAKE_A|T1|0|2023", 100.0), ("FAKE_B|T1|0|2023", 200.0)]


def test_a_select_by_meaning_reads_a_ref_that_names_another_table_in_that_table():
    # the cells resolve finds in FAKE_A carry refs of FAKE_B: they are read in FAKE_B (the rule before S2)
    a = TableInput(meta=_meta("FAKE_A"), cells=tuple(_sex_cells("FAKE_A", {"0": "100"}, ref_table="FAKE_B")))
    b = TableInput(meta=_meta("FAKE_B"), cells=tuple(_sex_cells("FAKE_B", {"0": "200"})))
    run = run_plan(a, PlanV3(steps=[_select()], show=["s"], present="value"), tables=_tables(a, b))
    assert [(f.source_ref, f.value) for f in run.facts] == [("FAKE_B|T1|0|2023", 200.0)]


def test_a_select_by_meaning_reads_every_cell_it_resolves():
    table = _population(5)
    plan = PlanV3(steps=[SelectStep(id="s", item=TargetSpec(codes=["T1"]),
                                    dims={"행정구역별": TargetSpec(codes=["11", "21"]), "성별": TargetSpec(codes=["0"])},
                                    periods=PeriodSpec(kind="list", periods=["2022", "2023"]))],
                  show=["s"], present="value")
    run = run_plan(table, plan)
    assert [(f.source_ref, f.value) for f in run.facts] == [
        (f"FAKE_S2|T1|{r}|0|{p}", 1000 + int(p) + int(r) * 7) for r in ("11", "21") for p in ("2022", "2023")]


def test_a_rank_reads_each_total_in_its_candidate_s_table():
    # FAKE_B holds the candidates and their total, whose ref names FAKE_A: it is read in FAKE_B, beside the candidates
    a = TableInput(meta=_meta("FAKE_A"), cells=tuple(_sex_cells("FAKE_A", {"1": "1", "2": "1"})))
    b_cells = _sex_cells("FAKE_B", {"1": "40", "2": "60"}) + _sex_cells("FAKE_B", {"0": "100"}, ref_table="FAKE_A")
    b = TableInput(meta=_meta("FAKE_B"), cells=tuple(b_cells))
    step = RankStep(id="k", candidates=["FAKE_B|T1|1|2023", "FAKE_B|T1|2|2023"],
                    totals=["FAKE_A|T1|0|2023", "FAKE_A|T1|0|2023"], pick="max")
    run = run_plan(a, PlanV3(steps=[step], show=["k"], present="extreme"), tables=_tables(a, b))
    assert [(r["dims"]["성별"], r["value"]) for r in run.results["k"].ranking] == [("여자", 60.0), ("남자", 40.0)]


def test_scenario_alternatives_cover_every_defaulted_scenario_category():
    births, deaths = [("1", "중위"), ("2", "고위"), ("3", "저위")], [("1", "중위"), ("2", "고위")]
    cells = [_cell("FAKE_SC", ("T1", "가상 인구"), {"출산율 가정별": b, "사망률 가정별": d}, "2030", "100")
             for b in births for d in deaths]
    table = load_resources(TableInput(meta=_meta("FAKE_SC"), cells=tuple(cells))).table
    built = blocks.value("FAKE_SC|T1|1|1|2030", [])
    found = scenario_alternatives(table, index_of(table), built, {"출산율 가정별": "1", "사망률 가정별": "1"})
    assert [(a.axis, a.chosen, a.refs) for a in found] == [
        ("출산율 가정별", "중위", ["FAKE_SC|T1|2|1|2030", "FAKE_SC|T1|3|1|2030"]),
        ("사망률 가정별", "중위", ["FAKE_SC|T1|1|2|2030"])]
    assert scenario_alternatives(table, index_of(table), built, {}) == []


def test_nothing_asked_makes_no_store_request():
    cells = [_cell("FAKE_SC", ("T1", "가상 인구"), {"출산율 가정별": ("1", "중위"), "성별": sex}, "2030", "100")
             for sex in SEXES]  # a scenario category with one code: no other scenario to show
    table = load_resources(TableInput(meta=_meta("FAKE_SC"), cells=tuple(cells))).table
    store = counting(table)
    lookup = _lookup(table)
    assert lookup.cells([]) == [] and lookup.numeric_periods_of([]) == []
    computed = _computed([_fact("f1", cells[0])], [])
    assert caveats_of(computed, table) == []  # no fact shown
    assert engine._with_context(table, computed, []).result.facts == computed.result.facts  # no context cell
    assert store.calls == {}
    built = blocks.value(cells[0].source_ref, [])
    assert scenario_alternatives(table, index_of(table), built, {"출산율 가정별": "1"}) == []
    assert store.calls == {"by_ref": 1}  # the answer's own cell only


def _doubled(table: TableInput, tbl_id: str) -> TableInput:
    """The same table under another tbl_id, every value twice as large."""
    cells = [replace(c, value_raw=str(2 * int(c.value_raw)), source_value=2 * c.source_value,
                     source_ref=c.source_ref.replace(table.meta.tbl_id, tbl_id, 1)) for c in table.cells]
    return TableInput(meta=_meta(tbl_id), cells=tuple(cells))


def _region_select(step_id: str, table: str | None = None) -> SelectStep:
    return SelectStep(id=step_id, table=table, item=TargetSpec(codes=["T1"]),
                      dims={"행정구역별": TargetSpec(codes=["*"]), "성별": TargetSpec(codes=["0"])},
                      periods=PeriodSpec(kind="list", periods=["2024"]))


def test_a_map_reads_one_starred_select_in_each_of_two_tables_apart():
    # the same select in two tables of one shape: the map's cache keeps each table's cells apart
    a = _population(5, tbl_id="FAKE_A", regions=REGIONS[:5])
    b = _doubled(a, "FAKE_B")
    stores = {"FAKE_A": counting(a), "FAKE_B": counting(b)}
    codes = [code for code, _ in REGIONS[1:5]]
    body = [_region_select("a", "FAKE_A"), _region_select("b", "FAKE_B"), CombineStep(id="d", a="b", b="a", fn="diff")]
    plan = PlanV3(steps=[MapStep(id="m", axis="행정구역별", codes=codes, body=body, result="d")], show=["m"],
                  present="table")
    run = run_plan(a, plan, tables=_tables(a, b))
    assert [v.value for v in run.results["m"].values] == [1000 + 2024 + int(code) * 7 for code in codes]  # b - a = a
    assert {tbl: dict(s.calls) for tbl, s in stores.items()} == {"FAKE_A": {"cells_at": 1}, "FAKE_B": {"cells_at": 1}}


def test_a_select_by_meaning_after_a_map_reads_the_store_itself():
    table = _population(5)
    codes = [code for code, _ in REGIONS[1:4]]
    after = SelectStep(id="s", item=TargetSpec(codes=["T1"]),
                       dims={"행정구역별": TargetSpec(codes=["00"]), "성별": TargetSpec(codes=["0"])},
                       periods=PeriodSpec(kind="list", periods=["2023"]))
    plan = PlanV3(steps=[MapStep(id="m", axis="행정구역별", codes=codes, body=[_region_select("x")], result="x"), after],
                  show=["m", "s"], present="table")
    run = run_plan(table, plan)
    assert [v.value for v in run.results["m"].values] == [1000 + 2024 + int(code) * 7 for code in codes]
    assert [(f.source_ref, f.value) for f in run.facts if f.id in run.ids_of("s")] == [("FAKE_S2|T1|00|0|2023", 3023.0)]


def test_a_refused_map_leaves_neither_its_cache_nor_its_star():
    from kosis_chat.compute.plan_engine import CalcRefused, _Run

    run = _Run(_population(5))
    step = MapStep(id="m", axis="행정구역별", codes=["11", "99"], body=[_region_select("x")], result="x")  # 99: none
    with pytest.raises(CalcRefused):
        run.map(step)
    assert run.cache is None and run.star == {}
