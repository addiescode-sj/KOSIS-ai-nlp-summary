"""The cell store (T1-31 S1, kosis_chat/prepare/store.py): MemoryStore answers what a plain loop over table.cells
answers, with the dict semantics the stages had (last cell wins on a shared key or ref, series in table order within
one period), and summary() gives what judge.rules.targets() gives, summarized."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest

from kosis_chat.adapters.parquet import load_table
from kosis_chat.judge.rules import _symbol, defaults, parse, resolve, target_summary, targets
from kosis_chat.judge.schemas import Periods, Resolved
from kosis_chat.prepare.index import build_index, index_of
from kosis_chat.prepare.schemas import Cell, TableInput, TableMeta
from kosis_chat.prepare.store import ITEM_AXIS, MemoryStore, RangeSummary, cell_key, dim_key, store_of
from kosis_chat.schemas.resources import load_resources

META = TableMeta(tbl_id="FAKE_STORE", tbl_nm="가상 표", survey_nm="가상 조사", org_nm="가상 기관", period_type="Y",
                 source_url="", fetched_at="2026-11-20", input_format="parquet", file_sha256="x")


def _cell(item: str, sex: str | None, period: str, raw: str, kind: str, ref: str, unit: str = "명") -> Cell:
    dims = {"성별": {"0": "계", "1": "남자", "2": "여자"}[sex]} if sex else {}
    return Cell(item={"T1": "가상 인구", "T2": "가상 성비"}[item], dims=dims, period_raw=period, value_raw=raw,
                unit=unit, source_ref=ref, dim_codes={"성별": sex} if sex else {}, item_code=item, value_kind=kind)


# Not a valid table on purpose: a shared key (T1/1/2023 twice), a shared ref (r9 twice), a cell without 성별,
# symbols (blank, dash), two items, periods out of order, an empty unit.
CELLS = (
    _cell("T1", "0", "2024", "1,050", "ok", "r1"),
    _cell("T1", "0", "2022", "1,000", "ok", "r2"),
    _cell("T1", "1", "2023", "", "blank", "r3"),
    _cell("T1", "1", "2023", "-", "dash", "r4"),
    _cell("T1", "1", "2022", "480", "ok", "r5"),
    _cell("T1", "2", "2023", "-", "dash", "r6"),
    _cell("T1", "2", "2024", "", "blank", "r7"),
    _cell("T2", None, "2023", "90.9", "ok", "r8", unit="%"),
    _cell("T2", "0", "2023", "x", "x_secret", "r9", unit=""),
    _cell("T2", "0", "2024", "91.0", "ok", "r9", unit="%"),
)


@pytest.fixture
def table() -> TableInput:
    return TableInput(meta=META, cells=CELLS)


def test_cell_and_cells_at_match_a_dict_over_the_cells(table):
    store = store_of(table)
    by_key = {cell_key(c): c for c in table.cells}
    assert by_key[("T1", (("성별", "1"),), "2023")].source_ref == "r4"  # the last one wins
    keys = [*by_key, ("T1", (("성별", "9"),), "2023"), ("T2", (), "2024")]
    for key in keys:
        assert store.cell(key) is by_key.get(key)
    assert store.cells_at(keys) == {k: by_key[k] for k in keys if k in by_key}
    assert store.cells_at([]) == {}


def test_by_ref_and_by_refs_match_a_dict_over_the_cells(table):
    store = store_of(table)
    by_ref = {c.source_ref: c for c in table.cells}
    assert store.by_ref("r9") is CELLS[-1]  # the last one wins
    refs = [*by_ref, "nope"]
    for ref in refs:
        assert store.by_ref(ref) is by_ref.get(ref)
    assert store.by_refs(refs) == {r: by_ref[r] for r in refs if r in by_ref}


def test_series_are_sorted_by_period_and_keep_table_order_within_a_period(table):
    store = store_of(table)
    groups: dict = {}
    for c in table.cells:
        groups.setdefault((c.item_code, dim_key(c)), []).append(c)
    for (item, dims), cells in groups.items():
        assert list(store.series(item, dims)) == sorted(cells, key=lambda c: c.period_raw)
    assert [c.source_ref for c in store.series("T1", (("성별", "1"),))] == ["r5", "r3", "r4"]
    assert [c.source_ref for c in store.series("T2", ())] == ["r8"]
    assert list(store.series("T9", ())) == []


def test_items_first_and_units(table):
    store = store_of(table)
    assert store.item_codes() == frozenset(c.item_code for c in table.cells) == {"T1", "T2"}
    assert store.first() is table.cells[0]
    assert store.units() == sorted({c.unit for c in table.cells if c.unit}) == ["%", "명"]
    empty = store_of(replace(table, cells=()))
    assert empty.first() is None and empty.item_codes() == frozenset() and empty.units() == []


def test_one_store_per_table_object(table):
    assert store_of(table) is store_of(table) is MemoryStore.of(table)
    other = replace(table, cells=table.cells)
    assert store_of(other) is not store_of(table)


CODES = [
    {},
    {ITEM_AXIS: {"T1"}},
    {ITEM_AXIS: {"T2"}},
    {"성별": {"1"}},
    {"성별": {"1", "2"}},
    {"성별": {""}},  # the cell without 성별
    {ITEM_AXIS: {"T1"}, "성별": {"0"}},
    {ITEM_AXIS: {"T2"}, "성별": {"0", "1"}},
    {"지역": {"11"}},  # a category no cell has
]
PERIODS = [Periods(), Periods(singles=("2023",)), Periods(singles=("2022", "2024")),
           Periods(ranges=(("2023", "2024"),)), Periods(singles=("2030",))]


@pytest.mark.parametrize("codes", CODES)
@pytest.mark.parametrize("periods", PERIODS)
def test_summary_is_targets_summarized(table, codes, periods):
    index = build_index(table)
    parsed = SimpleNamespace(periods=periods)
    resolved = Resolved(codes=codes, ambiguous=[])
    cells = targets(index, parsed, resolved)
    expected = RangeSummary(
        count=len(cells),
        kinds=frozenset(c.value_kind for c in cells),
        numeric_periods=tuple(sorted({c.period_raw for c in cells if c.value_kind == "ok"})),
        sample=next((c for c in cells if c.value_kind != "blank"), None),
    )
    assert RangeSummary.of(cells) == expected
    assert target_summary(index, parsed, resolved) == expected
    chosen = [p for p in index.periods if periods.contains(p)]
    assert store_of(table).summary(codes, chosen) == expected


def test_summary_sample_is_the_first_non_blank_cell_in_table_order(table):
    found = store_of(table).summary({ITEM_AXIS: {"T1"}, "성별": {"1", "2"}}, ["2023"])
    assert found.count == 3 and found.kinds == {"blank", "dash"} and found.numeric_periods == ()
    assert found.sample is CELLS[3]


QUESTIONS = ["2023년 여자 가상 인구는?", "2024년 남자 가상 인구는?", "가상 인구 추이는?", "남자 가상 인구는?",
             "2030년 가상 인구는?", "작년 대비 여자 인구는?", "2022년부터 2024년까지 계 가상 인구"]


@pytest.mark.parametrize("question", QUESTIONS)
def test_rules_take_the_cells_or_their_summary(pop_path, question):
    index = index_of(load_resources(load_table(pop_path)).table)
    parsed = parse(index, question)
    resolved = resolve(index, parsed)
    cells, found = targets(index, parsed, resolved), target_summary(index, parsed, resolved)
    assert found == RangeSummary.of(cells)
    assert defaults(parsed, index, resolved, cells) == defaults(parsed, index, resolved, found)
    assert _symbol(parsed, index, resolved, cells) == _symbol(parsed, index, resolved, found)
