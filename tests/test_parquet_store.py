"""The file-backed table (T1-31 S3, kosis_chat/adapters/parquet_store.py): a table over the row threshold is read from
its Parquet file request by request, and every answer equals the in-memory route's for the same file: the meta, the
index, each cell store method, and the answers of whole questions. Tables that break an eligibility condition go the
memory route (or are refused over ROW_LIMIT, naming the condition). The scan is saved and reused by file stamp.

Synthetic tables only (made-up values), written with tiny row groups so that pruning and multi-group reads run.
The last test compares the two routes on the provided tables when the workspace has them (KOSIS_WORKSPACE)."""

from __future__ import annotations

import collections
import csv
import dataclasses
import itertools
import json
import logging
import os
import random
import re
import sys
import threading
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from kosis_chat import calc
from kosis_chat.adapters import parquet, parquet_store
from kosis_chat.llm.model import MockClient
from kosis_chat.pipeline import StatPipeline
from kosis_chat.prepare.convert import rows_of, tidy_rows
from kosis_chat.prepare.index import index_of
from kosis_chat.prepare.schemas import FileBackedError, SchemaError, validate
from kosis_chat.prepare.store import ITEM_AXIS, MemoryStore, cell_key, dim_key, store_of
from kosis_chat.schemas.resources import load_resources
from tests.conftest import CATALOG_COLUMNS, PARQUET_COLUMNS

from kosis_ai.paths import workspace_root  # isort: skip

KINDS = {"": "blank", "-": "dash", "x": "x_secret", "*": "asterisk", "560p": "other_text"}
CATALOG = {"survey_name": "가상 조사", "organization": "가상 기관", "table_name": "가상 표", "period": "년 (2020 ~ 2024)",
           "url": "https://example.invalid/t"}


@pytest.fixture(autouse=True)
def index_dir(tmp_path, monkeypatch):
    """Saved scans go to a folder of the test (the default, <data dir>/index, is tested on its own)."""
    folder = tmp_path / "saved_index"
    monkeypatch.setattr(parquet_store, "INDEX_DIR", folder)
    return folder


def _row(period, item, levels, raw, *, prd_se="Y", unit="명", item_name=None) -> dict:
    """levels: (name, code, label) per level, from dim1."""
    row = {"org_id": "101", "source": "api", "prd_se": prd_se, "period": period, "item_id": item,
           "item_name": item_name or {"T1": "가상 인구", "T2": "가상 성비", "T3": "가상 출생아수 (명)"}.get(item, f"가상 {item}"),
           "unit": unit, "value_raw": raw, "value_kind": KINDS.get(raw, "ok") if raw is not None else None,
           "value": float(raw.replace(",", "")) if raw is not None and raw not in KINDS else None}
    for level, (name, code, label) in enumerate(levels, 1):
        row |= {f"dim{level}_name": name, f"dim{level}_code": code, f"dim{level}_label": label}
    return row


def write(folder: Path, tbl_id: str, rows: list[dict], group_rows: int = 3, *, key: str | None = None,
          types: dict | None = None) -> Path:
    """values/<key>.parquet (key: 101_<tbl_id> unless given) in row groups of group_rows rows, and its tables.csv row.
    types: Arrow types of some columns (else string, and float64 for value)."""
    key = key or f"101_{tbl_id}"
    values = folder / "values"
    values.mkdir(parents=True, exist_ok=True)
    full = [{"tbl_id": tbl_id} | r for r in rows]
    types = types or {}
    columns = {}
    for c in PARQUET_COLUMNS:
        kind = types.get(c, pa.float64() if c == "value" else pa.string())
        plain = kind.value_type if pa.types.is_dictionary(kind) else kind
        column = pa.array([r.get(c) for r in full], type=plain)
        columns[c] = column.dictionary_encode() if pa.types.is_dictionary(kind) else column
    table = pa.table(columns)
    path = values / f"{key}.parquet"
    pq.write_table(table, path, row_group_size=max(1, group_rows))
    catalog = folder / "tables.csv"
    new = not catalog.exists()
    with catalog.open("a", encoding="utf-8-sig" if new else "utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CATALOG_COLUMNS)
        if new:
            writer.writeheader()
        writer.writerow({"key": key, "org_id": "101", "tbl_id": tbl_id, "source": "api",
                         "parquet": f"values/{key}.parquet", **CATALOG})
    return path


REGIONS = [("00", "전국"), ("11", "서울특별시"), ("21", "부산광역시"), ("11010", "종로구")]
SEXES = [("0", "계"), ("1", "남자"), ("2", "여자")]
SYMBOLS = ["-", "x", "*", "560p", ""]


def rich_rows(seed: int = 0, scenario: bool = True) -> list[dict]:
    """Three items and units (one only in the item name), two or three named levels, a level with null codes, a
    scenario category, every symbol, rule-1 rows, annual and monthly rows that share no key, unsorted."""
    rows = []
    for year in range(2020, 2025):
        for r, (rc, rl) in enumerate(REGIONS):
            for s, (sc, sl) in enumerate(SEXES):
                raw = SYMBOLS[(year + r + s) % 5] if (year + r * 3 + s) % 7 == 0 else f"{1000 + year + r * 10 + s:,}"
                rows.append(_row(str(year), "T1", [("행정구역별", rc, rl), ("성별", sc, sl)], raw))
            # 가상 성비 has the 성별 level named but no code on it
            rows.append(_row(str(year), "T2", [("행정구역별", rc, rl), ("성별", None, "계")], f"9{r}.{year % 10}", unit="%"))
        for code, label in (("1", "중위"), ("2", "고위")) if scenario else ():  # a third level only on these rows
            rows.append(_row(str(year), "T3", [("행정구역별", "00", "전국"), ("성별", "0", "계"), ("출산율 가정별", code, label)],
                             str(300 + year % 10 + int(code)), unit=None))
    for month in ("202401", "202402", "202403"):
        rows.append(_row(month, "T1", [("행정구역별", "00", "전국"), ("성별", "0", "계")], "77", prd_se="M"))
    rows += [_row("", "T1", [("행정구역별", "00", "전국"), ("성별", "0", "계")], ""),
             _row(None, "T2", [("행정구역별", "11", "서울특별시"), ("성별", None, "계")], None)]
    random.Random(seed).shuffle(rows)
    return rows


def cycle_rows(collide: bool = True) -> list[dict]:
    """Year blocks (by year, then region, months interleaved): months, quarters writing the same period strings
    (collide) or only for another item (not collide), and years."""
    rows = []
    for year in (2020, 2021):
        for rc, rl in REGIONS[:3]:
            level = [("행정구역별", rc, rl)]
            for month in range(1, 7):
                rows.append(_row(f"{year}{month:02d}", "T1", level, str(year + month), prd_se="M"))
                if month <= 4:
                    rows.append(_row(f"{year}{month:02d}", "T1" if collide else "T9", level, str(month * 1000), prd_se="Q"))
            rows.append(_row(str(year), "T1", level, str(year * 10), prd_se="Y"))
    return rows


def load_both(path: Path):
    """(memory route, file route) of one file, cleaned and indexed as the pipeline holds them."""
    memory = load_resources(parquet.load_table(path, big_rows=10**9))
    file = load_resources(parquet.load_table(path, big_rows=0))
    assert not memory.table.file_backed and file.table.file_backed
    return memory, file


def index_view(index) -> tuple:
    """Every field of a TableIndex but the table, dict orders included (measures' sets sorted)."""
    return ([(name, list(a.labels.items()), list(a.parents.items())) for name, a in index.axes.items()], index.periods,
            [(key, sorted(found, key=repr)) for key, found in index.measures.items()], index.item_named, index.units)


def absent_keys(keys: list) -> list:
    item, dims, period = keys[0]
    return [(item, dims, "1999"), ("NOPE", dims, period), (item, (*dims, ("없는 분류", "1")), period), (item, (), period),
            (item, tuple((n, "zz") for n, _ in dims), period), ("", dims, period)]


def same(a, b) -> bool:
    """a == b, or equal in repr: a cell whose value is NaN (a row that rule 3 keeps alone) equals no other cell, not
    even its own copy, since NaN != NaN."""
    return a == b or repr(a) == repr(b)


def same_summary(a, b) -> bool:
    return (a.count, a.kinds, a.numeric_periods) == (b.count, b.kinds, b.numeric_periods) and same(a.sample, b.sample)


def compare_stores(memory, file, *, summaries: int = 120, seed: int = 0, sample: int | None = None) -> None:
    """Every CellStore method of the file store against MemoryStore over the memory route's cells."""
    mem, fil = store_of(memory.table), store_of(file.table)
    assert isinstance(mem, MemoryStore) and fil is file.table.store
    rng = random.Random(seed)
    cells = list(memory.table.cells)
    chosen = cells if sample is None or len(cells) <= sample else rng.sample(cells, sample)
    keys = [cell_key(c) for c in chosen] + absent_keys([cell_key(c) for c in chosen])
    for key in keys[:40]:
        assert same(fil.cell(key), mem.cell(key)), key
    got = fil.cells_at(keys)
    assert same(list(got.items()), list(mem.cells_at(keys).items()))
    refs = [c.source_ref for c in chosen]
    refs += [f"OTHER{refs[0]}", "", "garbage", refs[0] + "|x", refs[0].rsplit("|", 1)[0] + "|1999",
             f"{memory.table.meta.tbl_id}|None||2020", f"{memory.table.meta.tbl_id}|T1||2020", "|".join(["a"] * 6)]
    for ref in refs[:40]:
        assert same(fil.by_ref(ref), mem.by_ref(ref)), ref
    assert same(list(fil.by_refs(refs).items()), list(mem.by_refs(refs).items()))
    series = list(dict.fromkeys((c.item_code, dim_key(c)) for c in chosen))
    series += [("NOPE", series[0][1]), (series[0][0], (("없는 분류", "1"),)), (series[0][0], ())]
    for key in series[:20]:
        assert same(tuple(fil.series(*key)), tuple(mem.series(*key))), key
    assert same([(k, tuple(v)) for k, v in fil.series_at(series).items()],
                [(k, tuple(v)) for k, v in mem.series_at(series).items()])
    assert list(fil.series_at(series)) == list(mem.series_at(series))
    assert fil.cells_at([]) == fil.by_refs([]) == fil.series_at([]) == {}
    index = memory.index
    axes = {name: list(axis.labels) for name, axis in index.axes.items()} | {"없는 분류": ["1"]}
    for _ in range(summaries):
        codes = {}
        for name in rng.sample(sorted(axes), rng.randint(0, min(3, len(axes)))):
            pool = axes[name] + [""]
            codes[name] = set(rng.sample(pool, rng.randint(1, min(3, len(pool)))))
        periods = rng.choice([[], index.periods, rng.sample(index.periods, min(2, len(index.periods))), ["1999"],
                              [index.periods[-1], "1999"]])
        assert same_summary(fil.summary(codes, periods), mem.summary(codes, periods)), (codes, periods)


def assert_same(path: Path, **kw) -> tuple:
    memory, file = load_both(path)
    assert file.table.meta == memory.table.meta
    assert index_view(file.index) == index_view(memory.index)
    compare_stores(memory, file, **kw)
    return memory, file


# ---------------------------------------------------------------- parity on made-up files


@pytest.mark.parametrize("group_rows", [1, 4, 1000])
@pytest.mark.parametrize("seed", [0, 1])
def test_a_rich_table_answers_alike_both_ways(tmp_path, group_rows, seed):
    memory, file = assert_same(write(tmp_path / "data", "FAKE_RICH", rich_rows(seed), group_rows), seed=seed)
    assert "시점·값이 빈 행 2 제외" in file.table.meta.note and file.table.meta.period_type == "M"
    assert len(file.table.store.facts.layouts) == 2  # 2 and 3 named levels


def test_a_request_larger_than_the_memo_is_answered_whole(tmp_path, monkeypatch):
    for name in ("MEMO_CELLS", "MEMO_SERIES_CELLS", "MEMO_SUMMARIES"):
        monkeypatch.setattr(parquet_store, name, 2)
    assert_same(write(tmp_path / "data", "FAKE_RICH", rich_rows(), 4), summaries=30)


@pytest.mark.parametrize("collide", [True, False])
def test_two_period_codes_are_settled_as_tidy_rows_does(tmp_path, collide):
    memory, file = assert_same(write(tmp_path / "data", "FAKE_CYCLE", cycle_rows(collide), 5))
    facts = file.table.store.facts
    if collide:  # every quarter row goes, as in the memory route
        assert facts.losing == ["Q"] and facts.cycle_rows == 24 and "다른 주기 행 24 제외" in file.table.meta.note
        assert all(c.value != 1000.0 for c in memory.table.cells)
    else:
        assert facts.losing == [] and "행 정리" not in file.table.meta.note


def merge_rows(seed: int = 0) -> list[dict]:
    """Rows that tidy rule 3 settles, of every kind DUPLICATES makes but "cycle" (which would make every year row lose
    to the months below) (same and conflicting values, NaN, None, 0.0 and
    -0.0, labels, names and level names that differ on a dropped row), keys of three rows, a label first seen on a
    dropped row, and keys whose first row rule 2 drops (a quarter row first, then months); shuffled by seed, the
    first row of each key ahead of its later rows."""
    sex = [("성별", "0", "계"), ("성별", "1", "남자"), ("성별", "2", "여자")]
    first, later = [], []
    for year in (2020, 2021, 2022):
        for n, level in enumerate(sex):
            for item in ("T1", "T2"):
                row = _row(str(year), item, [level], str(year + n))
                first.append(row)
                later.append(duplicate(row, DUPLICATES[(year + 3 * n + (item == "T2")) % (len(DUPLICATES) - 1)]))
                if (year + n) % 3 == 0:  # a third row: equal to the first, or another value
                    later.append(duplicate(row, "same" if item == "T1" else "value"))
    # a label first seen on a dropped row, then on a kept row of another period
    first.append(_row("2023", "T3", [("성별", "1", "남자")], "1"))
    later += [_row("2023", "T3", [("성별", "1", "남성")], "1"), _row("2024", "T3", [("성별", "1", "남성")], "2")]
    # rule 2 drops the first row of these keys: a quarter row, then two equal month rows (months win: "M" < "Q")
    for month in ("202401", "202402"):
        first.append(_row(month, "T4", [("성별", "0", "계")], "1", prd_se="Q"))
        later += [_row(month, "T4", [("성별", "0", "계")], "2", prd_se="M"),
                  _row(month, "T4", [("성별", "0", "합계")], "2", prd_se="M")]
    rng = random.Random(seed)
    rng.shuffle(first)
    rng.shuffle(later)
    return first + later


def _shared_rows(rows: list[dict]) -> int:
    """The rows of merge_rows() in keys of two rows or more after rules 1 and 2 (rule 2 drops its quarter rows)."""
    from kosis_chat.prepare.convert import _cell_key

    sizes = collections.Counter(_cell_key({c: r.get(c) for c in PARQUET_COLUMNS}) for r in rows if r["prd_se"] != "Q")
    return sum(n for n in sizes.values() if n > 1)


@pytest.mark.parametrize("group_rows", [1, 4, 1000])
@pytest.mark.parametrize("seed", [0, 1])
def test_rows_that_share_a_key_are_settled_as_tidy_rows_does(tmp_path, group_rows, seed):
    rows = merge_rows(seed)
    memory, file = assert_same(write(tmp_path / "data", "FAKE_MERGE", rows, group_rows), seed=seed)
    _, counts = tidy_rows(rows_of(pq.read_table(file.table.store.path)))
    facts = file.table.store.facts
    assert counts["same_rows"] and counts["conflict_cells"] and counts["cycle_rows"]
    assert (facts.same_rows, facts.conflict_cells, facts.cycle_rows) == (counts["same_rows"], counts["conflict_cells"],
                                                                         counts["cycle_rows"])
    assert f"같은 행 {counts['same_rows']} 제외" in file.table.meta.note
    assert f"값이 서로 다른 겹친 셀 {counts['conflict_cells']}개는 값 없음으로 처리" in file.table.meta.note
    # every cell in order, the blank ones included; the rows rule 3 drops never answer
    cells = list(memory.table.cells)
    store = file.table.store
    assert list(store.cells_at([cell_key(c) for c in cells]).values()) == cells
    assert list(store.by_refs([c.source_ref for c in cells]).values()) == cells
    blank = [c for c in cells if c.value_kind == "blank" and c.value_raw == "" and c.source_value is None]
    assert len(blank) >= counts["conflict_cells"] and len(cells) == len(rows) - counts["cycle_rows"] - len(facts.drop)
    assert store.summary({}, store.periods).count == len(cells)
    assert "다른 분류" not in [name for name, *_ in index_view(file.index)[0]]  # a level name of dropped rows only


def test_a_conflict_is_blank_in_every_answer(tmp_path):
    rows = [_row(str(y), "T1", [("성별", sc, sl)], str(y + n)) for y in (2020, 2021) for n, (sc, sl) in enumerate(SEXES)]
    rows += [duplicate(rows[1], "value"), duplicate(rows[2], "same"), duplicate(rows[3], "nan_both")]
    memory, file = assert_same(write(tmp_path / "data", "FAKE_CONFLICT", rows, 2))
    store = file.table.store
    assert (store.facts.same_rows, store.facts.conflict_cells) == (1, 2)
    assert list(store.facts.drop) == [6, 7, 8] and list(store.facts.conflict) == [1, 3]
    blank = store.cell(("T1", (("성별", "1"),), "2020"))
    assert (blank.value_raw, blank.source_value, blank.value_kind, blank.value_flag) == ("", None, "blank", "missing")
    assert store.by_ref("FAKE_CONFLICT|T1|0|2021") == store_of(memory.table).by_ref("FAKE_CONFLICT|T1|0|2021")
    assert [c.value_kind for c in store.series("T1", (("성별", "1"),))] == ["blank", "ok"]
    found = store.summary({"성별": {"1", "2"}}, ["2020"])
    assert (found.count, found.kinds, found.numeric_periods) == (2, frozenset({"blank", "ok"}), ("2020",))
    found = store.summary({"성별": {"1"}}, ["2020"])
    assert (found.count, found.kinds, found.numeric_periods, found.sample) == (1, frozenset({"blank"}), (), None)


def test_a_table_without_shared_keys_makes_no_second_pass_and_holds_nothing(tmp_path, index_dir, monkeypatch):
    monkeypatch.setattr(parquet_store, "_merge_pass", lambda *a: pytest.fail("a second pass"))
    path = write(tmp_path / "data", "FAKE_RICH", rich_rows(), 4)
    index_dir.mkdir(parents=True)
    (index_dir / "101_FAKE_RICH.merge.parquet").write_bytes(b"left by an earlier version of the file")
    table = parquet.load_table(path, big_rows=0)
    facts = table.store.facts
    assert facts.drop is None and facts.conflict is None and facts.merge_rows == 0 and "같은 행" not in table.meta.note
    assert not (index_dir / "101_FAKE_RICH.merge.parquet").exists()


def test_the_rule_3_rows_are_saved_beside_the_scan_and_reused(tmp_path, index_dir, monkeypatch):
    path = write(tmp_path / "data", "FAKE_MERGE", merge_rows(), 4)
    first = parquet.load_table(path, big_rows=0)
    saved = index_dir / "101_FAKE_MERGE.merge.parquet"
    table = pq.read_table(saved)
    facts = first.store.facts
    assert table.schema.metadata[b"kosis_source_sha256"] == facts.sha256.encode("ascii")
    assert table.column("row").to_pylist() == sorted([*facts.drop.tolist(), *facts.conflict.tolist()])
    assert table.column("conflict").to_pylist() == [r in set(facts.conflict.tolist()) for r in table.column("row").to_pylist()]
    assert facts.merge_rows == table.num_rows == len(facts.drop) + len(facts.conflict)
    real = parquet_store.scan
    monkeypatch.setattr(parquet_store, "scan", lambda *a: pytest.fail("scanned again"))
    again = load_resources(parquet.load_table(path, big_rows=0))
    assert list(again.table.store.facts.drop) == list(facts.drop)
    assert list(again.table.store.facts.conflict) == list(facts.conflict)
    compare_stores(load_resources(parquet.load_table(path, big_rows=10**9)), again, summaries=20)
    # another file's rule-3 rows (as many rows), a missing file, a broken one: scanned again, never used
    monkeypatch.setattr(parquet_store, "scan", real)
    other = write(tmp_path / "other", "FAKE_MERGE", merge_rows(1), 4)
    parquet_store.load_facts(other, tmp_path / "other_index", parquet.sha256_of)
    for bad in ((tmp_path / "other_index" / saved.name).read_bytes(), None, b"not parquet"):
        if bad is None:
            saved.unlink()
        else:
            saved.write_bytes(bad)
        scans = []
        monkeypatch.setattr(parquet_store, "scan", lambda *a, scans=scans: scans.append(1) or real(*a))
        assert_same(path, summaries=10)
        assert scans and saved.exists()


def test_more_shared_rows_than_the_bound_are_ineligible(tmp_path, monkeypatch):
    rows = merge_rows()
    path = write(tmp_path / "data", "FAKE_MERGE", rows, 4)
    monkeypatch.setattr(parquet_store, "MAX_MERGE_ROWS", _shared_rows(rows))
    assert parquet_store.scan(path, parquet.sha256_of).failure is None
    monkeypatch.setattr(parquet_store, "MAX_MERGE_ROWS", _shared_rows(rows) - 1)
    assert parquet_store.scan(path, parquet.sha256_of).failure == "merge_rows"
    expected = parquet.load_table(path, big_rows=10**9)
    found = parquet.load_table(path, big_rows=0)
    assert not found.file_backed and found.cells == expected.cells and found.meta == expected.meta
    monkeypatch.setattr(parquet, "ROW_LIMIT", 3)
    with pytest.raises(SchemaError, match=re.escape(parquet_store.ELIGIBILITY["merge_rows"])):
        parquet.load_table(path, big_rows=0)


def _shared_keys(rows: list[dict]) -> int:
    """The tidy keys of merge_rows() with two rows or more after rules 1 and 2."""
    from kosis_chat.prepare.convert import _cell_key

    sizes = collections.Counter(_cell_key({c: r.get(c) for c in PARQUET_COLUMNS}) for r in rows if r["prd_se"] != "Q")
    return sum(1 for n in sizes.values() if n > 1)


def test_too_many_shared_keys_are_refused_before_the_second_pass(tmp_path, monkeypatch):
    rows = merge_rows()
    path = write(tmp_path / "data", "FAKE_MERGE", rows, 4)
    keys = _shared_keys(rows)
    passes = []
    real = parquet_store._merge_pass
    monkeypatch.setattr(parquet_store, "_merge_pass", lambda *a: passes.append(1) or real(*a))
    monkeypatch.setattr(parquet_store, "MAX_MERGE_ROWS", 2 * keys - 1)  # two rows a key at least: over the bound
    assert parquet_store.scan(path, parquet.sha256_of).failure == "merge_rows" and not passes
    monkeypatch.setattr(parquet_store, "MAX_MERGE_ROWS", 2 * keys)  # the pass runs, and meets more rows than that
    assert _shared_rows(rows) > 2 * keys
    assert parquet_store.scan(path, parquet.sha256_of).failure == "merge_rows" and passes


def test_the_raw_values_the_second_pass_holds_are_bounded(tmp_path, monkeypatch):
    rows = merge_rows()
    path = write(tmp_path / "data", "FAKE_MERGE", rows, 4)
    held = []
    real = parquet_store._Raws.add

    def add(self, values):
        real(self, values)
        held.append(self.nbytes)

    monkeypatch.setattr(parquet_store._Raws, "add", add)
    assert parquet_store.scan(path, parquet.sha256_of).failure is None
    monkeypatch.setattr(parquet_store, "MAX_MERGE_BYTES", held[-1])
    assert parquet_store.scan(path, parquet.sha256_of).failure is None
    monkeypatch.setattr(parquet_store, "MAX_MERGE_BYTES", held[-1] - 1)
    assert parquet_store.scan(path, parquet.sha256_of).failure == "merge_rows"
    message = parquet_store.ELIGIBILITY["merge_rows"]
    assert f"{parquet_store.MAX_MERGE_ROWS:,}" in message and "MB of their values" in message


def test_the_raw_values_are_compared_chunk_by_chunk_without_joining_them(monkeypatch):
    monkeypatch.setattr(parquet_store._Raws, "CHUNK", 200)
    rng = random.Random(5)
    words = [None, "", "1", "1 ", "가", "가가가"]
    raws, expected = parquet_store._Raws(), []
    joined = []
    monkeypatch.setattr(pa, "chunked_array", lambda *a, **k: joined.append(1) or pytest.fail("joined"))
    for n in range(300):
        held = [rng.choice(words) or rng.choice([None, "가" * rng.randint(1, 30)]) for _ in range(rng.randint(0, 7))]
        raws.add(pa.array(held, type=pa.large_string()))
        expected += held
        # chunks of CHUNK / 2 bytes or more, and below that a few of halving sizes
        assert raws.count == len(expected) and len(raws.chunks) <= 2 * raws.nbytes / 200 + 10
        assert all(chunk.nbytes <= 200 or len(chunk) <= 7 for chunk in raws.chunks)  # over CHUNK: one add alone
        if not expected:
            continue
        values = pa.array([rng.choice(words + expected[-3:]) for _ in range(9)], type=rng.choice([pa.string(),
                                                                                                    pa.large_string()]))
        at = np.array([rng.randrange(9) for _ in range(rng.randint(0, 12))], dtype=np.int64)
        positions = np.array([rng.randrange(len(expected)) for _ in at], dtype=np.int64)
        assert raws.equal(values, at, positions).tolist() == [values[a].as_py() == expected[p]
                                                                for a, p in zip(at, positions)]
    assert raws.nbytes >= sum(len((v or "").encode("utf-8")) + 8 for v in expected) and not joined


@pytest.mark.parametrize("group_rows", [1, 4, 1000])
def test_rule_3_answers_alike_with_tiny_raw_chunks_and_batches(tmp_path, monkeypatch, group_rows):
    monkeypatch.setattr(parquet_store._Raws, "CHUNK", 64)
    monkeypatch.setattr(parquet_store, "MERGE_BATCH_ROWS", 3)
    assert_same(write(tmp_path / "data", "FAKE_MERGE", merge_rows(1), group_rows), summaries=20)


def _bad_merge_files(good: pa.Table, rows: int) -> list[tuple[str, pa.Table]]:
    """Merge files with the sha256 of the good one that do not hold what its facts say."""
    row, flag = good.column("row").to_pylist(), good.column("conflict").to_pylist()
    free = next(r for r in range(rows) if r not in set(row))
    at = sum(r < free for r in row)

    def table(row_values, flags, kind=pa.int64()):
        return pa.table([pa.array(row_values, kind), pa.array(flags, pa.bool_())], names=["row", "conflict"])

    first_conflict = flag.index(True)
    return [("unsorted", table(row[::-1], flag[::-1])),
            ("a row twice", table(row[:1] + row[:1] + row[2:], flag)),
            ("negative", table([-1] + row[1:], flag)),
            ("past the end", table(row[:-1] + [rows], flag)),
            ("a conflict read as a drop", table(row, flag[:first_conflict] + [False] + flag[first_conflict + 1:])),
            ("one row more", table(row[:at] + [free] + row[at:], flag[:at] + [False] + flag[at:])),
            ("int32 rows", table(row, flag, pa.int32())),
            ("a null row", table(row[:-1] + [None], flag))]


@pytest.mark.parametrize("case", range(8))
def test_a_merge_file_that_does_not_hold_what_its_facts_say_is_scanned_again(tmp_path, index_dir, monkeypatch, case):
    path = write(tmp_path / "data", "FAKE_MERGE", merge_rows(), 4)
    facts = parquet.load_table(path, big_rows=0).store.facts
    saved = index_dir / "101_FAKE_MERGE.merge.parquet"
    good = pq.read_table(saved)
    name, bad = _bad_merge_files(good, facts.rows)[case]
    pq.write_table(bad.replace_schema_metadata(good.schema.metadata), saved)
    scans = []
    real = parquet_store.scan
    monkeypatch.setattr(parquet_store, "scan", lambda *a: scans.append(1) or real(*a))
    assert_same(path, summaries=20)
    assert scans, name
    assert pq.read_table(saved).equals(good), name  # saved again, as it was


def test_the_stale_merge_file_of_the_review_is_scanned_again(tmp_path, index_dir, monkeypatch):
    """The reviewer's case: rows 3, 1, 0 (unsorted) with the right sha256, count and schema were taken as they were,
    and a request then failed with an IndexError."""
    sex = lambda code, label="남자": [("성별", code, label)]  # noqa: E731
    rows = [_row("2020", "T1", sex("1"), "1"), _row("2020", "T1", sex("1"), "2"), _row("2021", "T1", sex("1"), "3"),
            _row("2021", "T1", sex("1"), "3"), _row("2022", "T1", sex("2", "여자"), "4"),
            _row("2023", "T1", sex("2", "여자"), "5")]
    path = write(tmp_path / "data", "FAKE_STALE", rows, 2)
    facts = parquet.load_table(path, big_rows=0).store.facts
    assert (list(facts.drop), list(facts.conflict)) == ([1, 3], [0])
    saved = index_dir / "101_FAKE_STALE.merge.parquet"
    good = pq.read_table(saved)
    pq.write_table(pa.table([pa.array([3, 1, 0], pa.int64()), pa.array([False, False, True])],
                            schema=good.schema), saved)
    table = parquet.load_table(path, big_rows=0)
    assert (list(table.store.facts.drop), list(table.store.facts.conflict)) == ([1, 3], [0])
    assert table.store.summary({}, table.store.periods).count == 4
    assert_same(path, summaries=20)


def test_the_facts_are_not_saved_when_the_merge_file_cannot_be_written(tmp_path, index_dir, monkeypatch):
    """The facts name the merge file's rows: written last, so that a stop half-way leaves no facts without it."""
    path = write(tmp_path / "data", "FAKE_MERGE", merge_rows(), 4)
    real = parquet_store._atomic

    def atomic(target, write, **kw):
        if target.name.endswith(".merge.parquet"):
            raise OSError("disk full")
        return real(target, write, **kw)

    monkeypatch.setattr(parquet_store, "_atomic", atomic)
    table = parquet.load_table(path, big_rows=0)
    assert table.store.facts.merge_rows and table.store.facts.descriptors is not None
    assert not (index_dir / "101_FAKE_MERGE.json").exists()


def test_a_saved_fact_that_disagrees_with_the_merge_file_is_scanned_again(tmp_path, index_dir, monkeypatch):
    path = write(tmp_path / "data", "FAKE_MERGE", merge_rows(), 4)
    facts = parquet.load_table(path, big_rows=0).store.facts
    meta = index_dir / "101_FAKE_MERGE.json"
    saved = json.loads(meta.read_text(encoding="utf-8"))
    for name, value in (("same_rows", len(facts.drop)), ("conflict_cells", facts.conflict_cells + 1), ("rows", 3)):
        meta.write_text(json.dumps(saved | {"facts": saved["facts"] | {name: value}}), encoding="utf-8")
        scans = []
        real = parquet_store.scan
        monkeypatch.setattr(parquet_store, "scan", lambda *a, scans=scans: scans.append(1) or real(*a))
        assert parquet.load_table(path, big_rows=0).store.facts.to_json() == facts.to_json() and scans, name
        monkeypatch.setattr(parquet_store, "scan", real)


def _case_row(period, item, levels, raw, *, prd_se="Y", unit="명", item_name="가상 인구", kind="auto",
              value="auto") -> dict:
    """The reviewer's row: levels as (name, code, label) or None (a level left empty); kind and value given or read
    from raw as _row does (a raw that does not parse: no value)."""
    row = {"org_id": "101", "source": "api", "prd_se": prd_se, "period": period, "item_id": item,
           "item_name": item_name, "unit": unit, "value_raw": raw}
    row["value_kind"] = (KINDS.get(raw, "ok") if raw is not None else None) if kind == "auto" else kind
    if value == "auto":
        try:
            value = float(raw.replace(",", "")) if raw is not None and raw not in KINDS else None
        except ValueError:
            value = None
    row["value"] = value
    for level, found in enumerate(levels, 1):
        if found is not None:
            row |= {f"dim{level}_name": found[0], f"dim{level}_code": found[1], f"dim{level}_label": found[2]}
    return row


def _reviewer_cases() -> list[tuple[str, list[dict], dict]]:
    """The reviewer's 57 hand-written rule-3 cases (review of S3, cases.py): (name, rows, column types)."""
    nan = float("nan")
    row = _case_row
    cases: list[tuple[str, list[dict], dict]] = []

    def run(name: str, rows: list[dict], **types) -> None:
        cases.append((name, rows, types))

    def sex(code, label="남자", name="성별"):
        return [(name, code, label)]

    base = [row("2020", "T1", sex("0", "계"), "10"), row("2021", "T1", sex("0", "계"), "11"),
            row("2020", "T1", sex("1"), "5")]
    run("rule2 drops first, months conflict", [
        row("202001", "T1", sex("1"), "1", prd_se="Q"), row("202001", "T1", sex("1"), "2", prd_se="M"),
        row("202001", "T1", sex("1"), "3", prd_se="M"), row("202002", "T1", sex("1"), "3", prd_se="M"),
        row("202003", "T1", sex("0", "계"), "3", prd_se="Q")])
    run("rule2: Q first equal to later M rows", [
        row("202001", "T1", sex("1"), "2", prd_se="Q"), row("202001", "T1", sex("1"), "2", prd_se="M"),
        row("202001", "T1", sex("1"), "2", prd_se="M"), row("202002", "T1", sex("1", "남성"), "3", prd_se="Q"),
        row("202002", "T1", sex("1"), "3", prd_se="M")])
    run("Y and None share a key", base + [row("2020", "T1", sex("1"), "5", prd_se=None),
                                          row("2021", "T1", sex("1"), "5", prd_se=""),
                                          row("2021", "T1", sex("1"), "6", prd_se="Y")])
    run("None first then Y", [row("2020", "T1", sex("1"), "5", prd_se=None),
                              row("2020", "T1", sex("1"), "6", prd_se="Y"),
                              row("2021", "T1", sex("1"), "6", prd_se="")])
    run("M,Q,None with None losing", [
        row("202001", "T1", sex("1"), "5", prd_se=None), row("202001", "T1", sex("1"), "6", prd_se="M"),
        row("202001", "T1", sex("1"), "6", prd_se="M"), row("202002", "T1", sex("1"), "6", prd_se="Q"),
        row("202003", "T1", sex("1"), "7", prd_se="Q"), row("202003", "T1", sex("1"), "7", prd_se="Q")])

    def pair(a: dict, b: dict) -> list[dict]:
        return base + [row("2022", "T1", sex("1"), **a), row("2022", "T1", sex("1"), **b)]

    run("NaN both", pair(dict(raw="1", value=nan, kind="ok"), dict(raw="1", value=nan, kind="ok")))
    run("NaN later", pair(dict(raw="1"), dict(raw="1", value=nan, kind="ok")))
    run("NaN alone", base + [row("2022", "T1", sex("1"), "1", value=nan, kind="ok")])
    run("None vs '' raw", pair(dict(raw=None, value=None, kind="blank"), dict(raw="", value=None, kind="blank")))
    run("None raw both", pair(dict(raw=None, value=None, kind=None), dict(raw=None, value=None, kind=None)))
    run("0.0 vs -0.0", pair(dict(raw="0", value=0.0, kind="ok"), dict(raw="0", value=-0.0, kind="ok")))
    run("raw equal value differs", pair(dict(raw="1", value=1.0, kind="ok"), dict(raw="1", value=2.0, kind="ok")))
    run("value equal raw differs", pair(dict(raw="1", value=1.0, kind="ok"), dict(raw="1.0", value=1.0, kind="ok")))
    run("kind differs", pair(dict(raw="1", value=1.0, kind="ok"), dict(raw="1", value=1.0, kind="other_text")))
    run("kind None vs ''", pair(dict(raw="1", value=1.0, kind=None), dict(raw="1", value=1.0, kind="")))
    run("value None vs 0.0", pair(dict(raw="0", value=None, kind="ok"), dict(raw="0", value=0.0, kind="ok")))
    run("2^53 vs 2^53+1", pair(dict(raw="9007199254740992", value=9007199254740992.0, kind="ok"),
                               dict(raw="9007199254740992", value=9007199254740993.0, kind="ok")))
    run("float32 near values", pair(dict(raw="0.1", value=0.1, kind="ok"),
                                    dict(raw="0.1", value=0.10000000149011612, kind="ok")), value=pa.float32())
    run("inf vs -inf", pair(dict(raw="1", value=float("inf"), kind="ok"),
                            dict(raw="1", value=float("-inf"), kind="ok")))
    run("raw unicode forms", pair(dict(raw="é", value=None, kind="other_text"),
                                  dict(raw="é", value=None, kind="other_text")))
    run("raw trailing space", pair(dict(raw="1", value=1.0, kind="ok"), dict(raw="1 ", value=1.0, kind="ok")))
    for combo in itertools.product("ab", repeat=3):
        run("triple " + "".join(combo), base + [row("2022", "T1", sex("1"), "1" if c == "a" else "2") for c in combo])
    run("NaN, NaN, 1", base + [row("2022", "T1", sex("1"), "1", value=nan, kind="ok"),
                               row("2022", "T1", sex("1"), "1", value=nan, kind="ok"),
                               row("2022", "T1", sex("1"), "1")])
    run("unit differs", pair(dict(raw="1", unit="명"), dict(raw="1", unit="천명")))
    run("unit differs first None", pair(dict(raw="1", unit=None), dict(raw="1", unit="%")))
    run("label differs", base + [row("2022", "T1", sex("1", "남자"), "1"), row("2022", "T1", sex("1", "남성"), "1"),
                                 row("2023", "T1", sex("1", "남성"), "1")])
    run("label of a dropped row seen first elsewhere later", [
        row("2022", "T1", sex("1", "남자"), "1"), row("2022", "T1", sex("1", "남성"), "1"),
        row("2023", "T1", sex("1", "남성"), "1"), row("2024", "T1", sex("1", "남자"), "1")])
    run("item name differs", pair(dict(raw="1", item_name="가상 인구"), dict(raw="1", item_name="가상 비율 (%)")))
    run("first item name empty", pair(dict(raw="1", item_name=" "), dict(raw="1", item_name="가상 인구")))
    run("later item name empty", pair(dict(raw="1", item_name="가상 인구"), dict(raw="1", item_name=None)))
    run("level name differs", base + [row("2022", "T1", sex("1"), "1"), row("2022", "T1", sex("1", name="다른 분류"), "1")])
    run("level name differs, first is the odd one", [
        row("2022", "T1", sex("1", name="다른 분류"), "1"), row("2022", "T1", sex("1"), "1"),
        row("2023", "T1", sex("1"), "1")])

    def two(period, value, first, second):
        return row(period, "T1", [first, second], value)

    run("ghost swaps names over levels", [two("2020", "1", ("X", "1", "a"), ("Y", "2", "b")),
                                          two("2020", "1", ("Y", "1", "a"), ("X", "2", "b")),
                                          two("2021", "1", ("X", "1", "a"), ("Y", "2", "b"))])
    run("ghost has one name on two levels", [two("2020", "1", ("X", "1", "a"), ("Y", "2", "b")),
                                             two("2020", "2", ("X", "1", "a"), ("X", "2", "b"))])
    run("ghost cell_key clash", [two("2020", "1", ("A", "1", "a"), ("B", "2", "b")),
                                 two("2020", "1", ("C", "2", "a"), ("D", "1", "b")),
                                 two("2020", "1", ("B", "2", "a"), ("A", "1", "b"))])
    run("ghost pipe label and name", [row("2020", "T1", sex("1"), "1"), row("2020", "T1", sex("1", "a|b", "성|별"), "1")])
    run("first has a pipe name", [row("2020", "T1", sex("1", "a|b", "성|별"), "1"), row("2020", "T1", sex("1"), "1")])
    run("all rows one key same", [row("2020", "T1", sex("1"), "1") for _ in range(7)])
    run("all rows one key, conflict at the end", [row("2020", "T1", sex("1"), "1") for _ in range(6)]
        + [row("2020", "T1", sex("1"), "2")])
    run("every row twice in order", [dict(r) for r in base for _ in (0, 1)])
    run("every row twice, second block", base + [dict(r) for r in base])
    run("no levels, dup", [row("2020", "T1", [], "1"), row("2020", "T1", [], "1"), row("2021", "T1", [], "2"),
                           row("2021", "T1", [], "3")])
    run("null item dup", [row("2020", None, sex("1"), "1"), row("2020", None, sex("1"), "2"),
                          row("2021", None, sex("1"), "2")])
    run("null item vs ''", [row("2020", None, sex("1"), "1"), row("2020", "", sex("1"), "1")])
    run("null code vs '' code", [row("2020", "T1", sex(None), "1"), row("2020", "T1", sex(""), "1"),
                                 row("2020", "T1", sex(None), "1")])
    run("null code dup", [row("2020", "T1", sex(None), "1"), row("2020", "T1", sex(None), "2"),
                          row("2021", "T1", sex("1"), "2")])
    run("rule1 rows between dups", [row("2020", "T1", sex("1"), "1"), row(None, "T1", sex("1"), None),
                                    row("", "T1", sex("1"), ""), row("2020", "T1", sex("1"), "1"),
                                    row(None, "T1", sex("1"), None)])
    run("sample skips a conflict", [row("2020", "T1", sex("1"), "1"), row("2020", "T1", sex("1"), "2"),
                                    row("2020", "T1", sex("2", "여자"), "3"), row("2021", "T1", sex("1"), "4")])
    run("conflict with a text first row", [row("2020", "T1", sex("1"), "x"), row("2020", "T1", sex("1"), "2"),
                                           row("2021", "T1", sex("1"), "-"), row("2021", "T1", sex("1"), "-")])
    strings = [c for c in PARQUET_COLUMNS if c != "value"]
    run("dictionary strings", pair(dict(raw="1"), dict(raw="2")),
        **{c: pa.dictionary(pa.int32(), pa.string()) for c in strings})
    run("large strings", pair(dict(raw="1"), dict(raw="2")), **{c: pa.large_string() for c in strings})
    run("dictionary value column", pair(dict(raw="1"), dict(raw="2")), value=pa.dictionary(pa.int32(), pa.float64()))
    run("float16 value", pair(dict(raw="1"), dict(raw="2")), value=pa.float16())
    return cases


REVIEWER_CASES = _reviewer_cases()


@pytest.mark.parametrize("case", range(len(REVIEWER_CASES)), ids=[c[0] for c in REVIEWER_CASES])
def test_the_reviewers_rule_3_cases_answer_alike_fresh_and_saved(tmp_path, index_dir, case):
    """Each case once scanned, then again from the saved scan, merge file and index (row groups of 1, 2, 3 or 100)."""
    name, rows, types = REVIEWER_CASES[case]
    path = write(tmp_path / "data", "FAKE_CASE", rows, [1, 2, 3, 100][case % 4], types=types)
    try:
        memory = load_resources(parquet.load_table(path, big_rows=10**9))
    except SchemaError as exc:  # both routes refuse alike
        with pytest.raises(SchemaError) as again:
            parquet.load_table(path, big_rows=0)
        assert str(again.value) == str(exc)
        return
    file = load_resources(parquet.load_table(path, big_rows=0))
    if not file.table.file_backed:  # loaded whole: the memory route's own cells
        assert same(file.table.cells, memory.table.cells) and file.table.meta == memory.table.meta
        return
    for attempt in ("fresh", "saved"):
        assert file.table.meta == memory.table.meta and index_view(file.index) == index_view(memory.index), attempt
        compare_stores(memory, file, summaries=20, seed=case)
        if attempt == "fresh":
            file = load_resources(parquet.load_table(path, big_rows=0))
            assert file.table.store.index_state == "loaded"


def test_rows_repeated_only_on_a_code_rule_2_drops_make_no_second_pass(tmp_path, monkeypatch):
    sex = [("성별", "1", "남자")]
    rows = [_row("202001", "T1", sex, "1", prd_se="M"), _row("202002", "T1", sex, "2", prd_se="M"),
            _row("202001", "T1", sex, "5", prd_se="Y"), _row("202001", "T1", sex, "6", prd_se="Y")]
    monkeypatch.setattr(parquet_store, "_merge_pass", lambda *a: pytest.fail("a second pass"))
    memory, file = assert_same(write(tmp_path / "data", "FAKE_LOSING", rows, 1), summaries=10)
    facts = file.table.store.facts
    assert (facts.losing, facts.cycle_rows, facts.same_rows, facts.drop) == (["Y"], 2, 0, None)


def _value_row(raw, value, kind) -> dict:
    return _row("2022", "T1", [("성별", "1", "남자")], "1") | {"value_raw": raw, "value": value, "value_kind": kind}


def test_a_null_value_is_not_equal_to_zero_in_rule_3(tmp_path):
    rows = [_row(str(y), "T1", [("성별", sc, sl)], str(y)) for y in (2020, 2021) for sc, sl in SEXES]
    rows += [_value_row("0", None, "ok"), _value_row("0", 0.0, "ok")]
    memory, file = assert_same(write(tmp_path / "data", "FAKE_ZERO", rows, 2))
    facts = file.table.store.facts
    assert (facts.same_rows, facts.conflict_cells, list(facts.conflict), list(facts.drop)) == (0, 1, [6], [7])
    assert file.table.store.cell(("T1", (("성별", "1"),), "2022")).value_kind == "blank"


def test_two_period_codes_in_one_cell_are_two_rows_of_a_key(tmp_path):
    """prd_se "Y" and None on one key: rule 2 does not apply (one real code), and rule 3 sees two rows."""
    rows = [_row(str(y), "T1", [("성별", sc, sl)], str(y)) for y in (2020, 2021) for sc, sl in SEXES]
    rows += [_row("2022", "T1", [("성별", "1", "남자")], "7", prd_se="Y"),
             _row("2022", "T1", [("성별", "1", "남자")], "7", prd_se=None)]
    memory, file = assert_same(write(tmp_path / "data", "FAKE_TWOCODES", rows, 1000))
    facts = file.table.store.facts
    assert (facts.losing, facts.same_rows, facts.conflict_cells, list(facts.drop)) == ([], 1, 0, [7])


def test_item_ids_that_are_all_null_with_shared_keys(tmp_path):
    rows = [_row(str(y), None, [("성별", sc, sl)], str(y), item_name="가상 인구") for y in (2020, 2021) for sc, sl in SEXES]
    rows += [duplicate(rows[0], "same"), duplicate(rows[4], "value"), duplicate(rows[5], "name")]
    memory, file = assert_same(write(tmp_path / "data", "FAKE_NULLITEM", rows, 2))
    assert file.table.store.facts.item_null and file.table.store.facts.conflict_cells == 1


def test_quarterly_and_unsorted_periods(tmp_path):
    rows = [_row(f"{y}{q:02d}", "T1", [("성별", sc, sl)], str(y + q + int(sc)), prd_se="Q")
            for y in (2019, 2020, 2021) for q in (1, 2, 3, 4) for sc, sl in SEXES]
    random.Random(3).shuffle(rows)
    assert_same(write(tmp_path / "data", "FAKE_Q", rows, 2))


def test_a_table_without_categories(tmp_path):
    rows = [_row(str(y), item, [], str(y)) for y in (2020, 2021) for item in ("T1", "T2")]
    assert_same(write(tmp_path / "data", "FAKE_NODIM", rows, 1))


def test_item_ids_that_are_all_null(tmp_path):
    rows = [_row(str(y), None, [("성별", sc, sl)], str(y), item_name="가상 인구") for y in (2020, 2021) for sc, sl in SEXES]
    memory, file = assert_same(write(tmp_path / "data", "FAKE_NULLITEM", rows, 2))
    assert file.table.store.facts.item_null and file.table.store.by_ref("FAKE_NULLITEM|None|1|2020") is not None


DUPLICATES = ["same", "same", "value", "nan", "nan_both", "none", "zero", "raw_none", "kind", "label", "name", "level",
              "cycle"]


def duplicate(row: dict, how: str) -> dict:
    """A row with the tidy key of `row` (tidy rule 3) and, by `how`: the same values, another value, NaN (on the copy,
    or on both), no value at all, 0.0 and -0.0, None and "" as the raw value, another kind, another label, item name
    or unit, another level name on the first named level (the key reads codes, not names), another prd_se. "nan_both"
    and "zero" change `row` too."""
    copy = dict(row)
    if how == "value":
        copy |= {"value_raw": "9", "value": 9.0, "value_kind": "ok"}
    elif how == "nan":
        copy |= {"value": float("nan"), "value_kind": "ok"}
    elif how == "nan_both":
        for r in (row, copy):
            r |= {"value_raw": "NaN", "value": float("nan"), "value_kind": "ok"}
    elif how == "none":
        for r in (row, copy):
            r |= {"value_raw": None, "value": None, "value_kind": None}
    elif how == "zero":
        row |= {"value_raw": "0", "value": 0.0, "value_kind": "ok"}
        copy |= {"value_raw": "0", "value": -0.0, "value_kind": "ok"}
    elif how == "raw_none":
        row |= {"value_raw": "", "value": None, "value_kind": "blank"}
        copy |= {"value_raw": None, "value": None, "value_kind": "blank"}
    elif how == "kind":
        copy |= {"value_kind": "other_text"}
    elif how == "label":
        copy |= {k: "다른 라벨" for k in row if k.endswith("_label")}
    elif how == "name":
        copy |= {"item_name": "다른 항목", "unit": "천명"}
    elif how == "level":
        named = [k for k in row if k.endswith("_name") and k.startswith("dim") and row[k]]
        if named:
            copy[named[0]] = "다른 분류"
    elif how == "cycle":
        copy["prd_se"] = {"Y": "M", "M": "Q"}.get(row.get("prd_se"), "Y")
    return copy


_codes = st.sampled_from(["0", "1", "11", "11010", "", None])


@st.composite
def random_rows(draw) -> list[dict]:
    names = draw(st.lists(st.sampled_from(["성별", "행정구역별", "연령별", "출산율 가정별"]), max_size=3, unique=True))
    periods = draw(st.lists(st.sampled_from(["2020", "2021", "202001", "202002", "2022", "", " 2020"]), min_size=1,
                            max_size=4, unique=True))
    rows = []
    for item in draw(st.lists(st.sampled_from(["T1", "T2", "", None]), min_size=1, max_size=2, unique=True)):
        for _ in range(draw(st.integers(1, 4))):
            levels = [(name, draw(_codes), draw(st.sampled_from(["계", "남자", "서울특별시", "전국 (명)", None])))
                      for name in names if draw(st.booleans()) or len(names) < 2]
            for period in periods:
                if draw(st.integers(0, 4)):
                    rows.append(_row(period, item, levels, draw(st.sampled_from(["1", "2,000", "-", "x", "560p", "", None])),
                                     prd_se=draw(st.sampled_from(["Y", "Y", "M", "Q", None])),
                                     unit=draw(st.sampled_from(["명", None, ""]))))
    if rows and draw(st.integers(0, 2)):  # tidy rule 3: rows that share a key, equal or not
        for at, how in draw(st.lists(st.tuples(st.integers(0, len(rows) - 1), st.sampled_from(DUPLICATES)),
                                     max_size=8)):
            rows.append(duplicate(rows[at], how))
    return draw(st.permutations(rows)) if rows else [_row("2020", "T1", [], "1")]


@settings(max_examples=60, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture,
                                                                  HealthCheck.too_slow])
@given(rows=random_rows(), group_rows=st.integers(1, 6))
def test_random_tables_answer_alike_or_fall_back_to_memory(tmp_path_factory, rows, group_rows):
    path = write(tmp_path_factory.mktemp("data"), "FAKE_RANDOM", rows, group_rows)
    try:
        memory = load_resources(parquet.load_table(path, big_rows=10**9))
    except SchemaError as exc:  # the file route meets the same problem (memory fallback or the same check)
        with pytest.raises(SchemaError) as again:
            parquet.load_table(path, big_rows=0)
        assert str(again.value) == str(exc)
        return
    file = load_resources(parquet.load_table(path, big_rows=0))
    if not file.table.file_backed:  # ineligible: the memory route, unchanged
        assert same(file.table.cells, memory.table.cells) and file.table.meta == memory.table.meta
        return
    assert file.table.meta == memory.table.meta
    assert index_view(file.index) == index_view(memory.index)
    compare_stores(memory, file, summaries=30)


# ---------------------------------------------------------------- whole questions


QUESTIONS = ["2023년 서울특별시 가상 인구는?", "서울특별시 가상 인구는?", "서울특별시 가상 인구 추이는?",
             "2023년 가상 인구가 가장 많은 곳은?", "서울특별시 가상 인구가 가장 많은 해는?",
             "2023년 서울특별시와 부산광역시 가상 인구 차이는?", "2020년부터 2023년까지 서울특별시 가상 인구 평균은?",
             "2024년 여자 가상 인구는?", "2022년 남자 가상 인구는?", "2030년 가상 인구는?", "가상 성비는?"]


@pytest.fixture
def rich_path(tmp_path) -> Path:
    """The rich table without its scenario item (a scenario category the rules would default on every item)."""
    return write(tmp_path / "data", "FAKE_RICH", rich_rows(scenario=False), 4)


def test_questions_through_the_rules_give_the_same_results(rich_path):
    memory, file = load_both(rich_path)
    statuses = []
    for question in QUESTIONS:
        found = calc.calculate(question, [file])
        assert found == calc.calculate(question, [memory]), question
        statuses.append(found.status)
    assert statuses.count("ok") >= 7, statuses  # the comparison is about computed answers, not refusals


def _asked(table, question: str):
    answer = StatPipeline(table, client=MockClient("[mock] answer"), org_id="101").ask(question)
    meta = {k: v for k, v in answer.decision_metadata.items() if k != "spans"}
    return answer.stat_answer, meta, answer.chart


def test_questions_through_the_graph_give_the_same_answers(rich_path):
    memory = parquet.load_table(rich_path, big_rows=10**9)
    file = parquet.load_table(rich_path, big_rows=0)
    for question in QUESTIONS[:8]:
        found = _asked(file, question)
        assert found == _asked(memory, question), question
        assert found[0] is not None, question  # answered by the stat path, not by the fallback chatbot


def test_a_failed_plan_on_a_file_backed_table_is_refused_not_sent_to_the_fallback_chatbot(rich_path):
    question = "가상 성비는?"  # the rules leave it open (no 성별 code on 가상 성비); the mock writes no plan
    memory = StatPipeline(parquet.load_table(rich_path, big_rows=10**9), client=MockClient("[mock] answer"),
                          org_id="101").ask(question)
    assert memory.decision_metadata["route"] == "fallback_chatbot"  # the memory route keeps the fallback
    file = StatPipeline(parquet.load_table(rich_path, big_rows=0), client=MockClient("[mock] answer"),
                        org_id="101").ask(question)
    assert file.decision_metadata["route"] == "llm_plan+failed"
    assert file.stat_answer.outcome == "refused" and file.stat_answer.refusal.reason == "class_mismatch"
    assert "계산하는 방법을 정하지 못했습니다" in file.stat_answer.text


class _Down(MockClient):
    """A model that cannot be reached."""

    def generate(self, *args, **kwargs):
        raise TimeoutError("timed out")


def test_a_failed_model_call_on_a_file_backed_table_is_an_error_not_a_refusal(rich_path):
    question = "가상 성비는?"  # the rules leave it to the model's plan
    file = StatPipeline(parquet.load_table(rich_path, big_rows=0), client=_Down("x"), org_id="101")
    with pytest.raises(RuntimeError, match="model call failed: timeout"):
        file.ask(question)
    memory = StatPipeline(parquet.load_table(rich_path, big_rows=10**9), client=_Down("x"), org_id="101").ask(question)
    assert memory.stat_answer is None  # the memory route: the fallback's failed call, an error at api (RuntimeError)


def test_a_question_makes_a_few_file_reads_and_a_repeated_one_none(rich_path):
    file = load_resources(parquet.load_table(rich_path, big_rows=0))
    store = file.table.store
    for question in QUESTIONS:
        before = store.reads
        calc.calculate(question, [file])
        assert store.reads - before <= 6, question
        before = store.reads
        calc.calculate(question, [file])
        assert store.reads == before, question  # the memo answers a repeated question


# ---------------------------------------------------------------- the seam


def test_a_file_backed_table_refuses_to_be_read_as_a_sequence(rich_path):
    table = parquet.load_table(rich_path, big_rows=0)
    for use in (list, len, bool, lambda c: c[0], lambda c: "x" in c):
        with pytest.raises(FileBackedError, match="store_of"):
            use(table.cells)
    with pytest.raises(FileBackedError):
        table.period_range  # noqa: B018  (the memory-only helpers say so too)
    with pytest.raises(FileBackedError):
        MemoryStore.of(table)
    assert validate(table) == [] and "file-backed" in repr(table)
    assert load_resources(table).table is table  # clean() keeps it


def test_the_entrance_reads_the_threshold_at_call_time(rich_path, monkeypatch):
    assert not parquet.load_table(rich_path).file_backed  # BIG_TABLE_ROWS = 500_000
    rows = pq.read_metadata(rich_path).num_rows
    monkeypatch.setattr(parquet, "BIG_TABLE_ROWS", rows)  # the file route from BIG_TABLE_ROWS rows on
    assert parquet.load_table(rich_path).file_backed
    monkeypatch.setattr(parquet, "BIG_TABLE_ROWS", rows + 1)
    assert not parquet.load_table(rich_path).file_backed
    assert parquet.load_table(rich_path, big_rows=rows).file_backed
    assert not parquet.load_table(rich_path, big_rows=rows + 1).file_backed
    assert not parquet.load_table(rich_path, big_rows=sys.maxsize).file_backed  # what the legacy callers pass
    assert parquet.sha256_of(rich_path) == __import__("hashlib").sha256(rich_path.read_bytes()).hexdigest()


# ---------------------------------------------------------------- ineligible tables


def _ineligible_cases() -> list[tuple[str, list[dict]]]:
    base = [_row(str(y), "T1", [("성별", sc, sl)], str(y)) for y in (2020, 2021) for sc, sl in SEXES]
    return [
        ("merge_rows", base + [_row("2020", "T1", [("성별", "1", "남자")], "5")]),  # with MAX_MERGE_ROWS 1
        ("pipe", base + [_row("2022", "T1", [("성별", "1|2", "남자")], "5")]),
        ("period", base + [_row(" 2022", "T1", [("성별", "1", "남자")], "5")]),
        ("item_null", base + [_row("2020", None, [("성별", "1", "남자")], "5")]),
        ("level", base + [_row("2022", "T1", [("연령별", "9", "전체"), ("성별", "1", "남자")], "5")]),
        ("cell_key", base + [_row("2020", "T1", [("성별", "1", "남자"), ("연령별", None, "전체")], "5")]),
        # one ref, two keys: a null code and an empty one; a row with no level and one whose only code is empty
        ("ref", base + [_row("2022", "T1", [("성별", None, "계")], "5"), _row("2022", "T1", [("성별", "", "계")], "6")]),
        ("ref", base + [_row("2022", "T1", [], "5"), _row("2022", "T1", [("성별", "", "계")], "6")]),
        ("item_name", base + [_row("2022", "T9", [("성별", "1", "남자")], "5", item_name=" ")]),
        ("codes", [_row(str(2000 + n), "T1", [("성별", "0", "계")], "1", prd_se=code) for n, code in enumerate("ABCDEFGHI")]),
        # nine triple columns (the item and eight levels) of 256 values each: more than 63 bits a descriptor
        ("descriptors", [_row("2020", "T1", [(f"분류{level}", f"{n:03d}", f"값{n}") for level in range(8)], str(n))
                         for n in range(256)]),
    ]


@pytest.mark.parametrize("condition, rows", _ineligible_cases(), ids=[c for c, _ in _ineligible_cases()])
def test_an_ineligible_table_is_loaded_whole_under_the_limit_and_refused_over_it(tmp_path, monkeypatch, condition, rows):
    if condition == "merge_rows":  # two rows share a key: more than the bound of one
        monkeypatch.setattr(parquet_store, "MAX_MERGE_ROWS", 1)
    path = write(tmp_path / "data", "FAKE_BAD", rows, 2)
    assert parquet_store.scan(path, parquet.sha256_of).failure == condition
    try:
        expected = parquet.load_table(path, big_rows=10**9)
    except SchemaError as exc:
        with pytest.raises(SchemaError) as again:
            parquet.load_table(path, big_rows=0)
        assert str(again.value) == str(exc)
    else:
        found = parquet.load_table(path, big_rows=0)
        assert not found.file_backed and found.cells == expected.cells and found.meta == expected.meta
    monkeypatch.setattr(parquet, "ROW_LIMIT", 3)
    with pytest.raises(SchemaError, match="cannot be read from the file: " + re.escape(parquet_store.ELIGIBILITY[condition])):
        parquet.load_table(path, big_rows=0)


# ---------------------------------------------------------------- the saved scan


def test_the_scan_is_saved_and_reused_until_the_file_changes(tmp_path, index_dir, monkeypatch):
    path = write(tmp_path / "data", "FAKE_RICH", rich_rows(), 4)
    first = parquet.load_table(path, big_rows=0)
    assert (index_dir / "101_FAKE_RICH.json").exists() and (index_dir / "101_FAKE_RICH.descriptors.parquet").exists()
    real = parquet_store.scan
    monkeypatch.setattr(parquet_store, "scan", lambda *a: pytest.fail("scanned again"))
    again = parquet.load_table(path, big_rows=0)
    assert again.meta == first.meta and index_view(index_of(again)) == index_view(index_of(first))
    monkeypatch.setattr(parquet_store, "scan", real)
    rows = rich_rows() + [_row("2025", "T1", [("행정구역별", "00", "전국"), ("성별", "0", "계")], "5")]
    path.unlink()
    write(tmp_path / "data", "FAKE_RICH", rows, 4)
    changed = parquet.load_table(path, big_rows=0)
    assert "2025" in index_of(changed).periods and changed.meta.file_sha256 != first.meta.file_sha256


def test_the_default_index_folder_is_beside_tables_csv(tmp_path, monkeypatch):
    monkeypatch.setattr(parquet_store, "INDEX_DIR", None)
    path = write(tmp_path / "data", "FAKE_RICH", rich_rows(), 4)
    parquet.load_table(path, big_rows=0)
    assert (tmp_path / "data" / "index" / "101_FAKE_RICH.json").exists()


def test_an_unwritable_or_corrupt_saved_scan_is_harmless(tmp_path, monkeypatch, index_dir):
    path = write(tmp_path / "data", "FAKE_RICH", rich_rows(), 4)
    expected = load_resources(parquet.load_table(path, big_rows=10**9))
    blocker = tmp_path / "a_file"
    blocker.write_text("x")
    monkeypatch.setattr(parquet_store, "INDEX_DIR", blocker / "index")  # cannot be made: skipped (logged)
    assert index_view(load_resources(parquet.load_table(path, big_rows=0)).index) == index_view(expected.index)
    monkeypatch.setattr(parquet_store, "INDEX_DIR", index_dir)
    parquet.load_table(path, big_rows=0)
    for name, garbage in (("101_FAKE_RICH.descriptors.parquet", b"not parquet"), ("101_FAKE_RICH.json", b"{oops")):
        (index_dir / name).write_bytes(garbage)
        file = load_resources(parquet.load_table(path, big_rows=0))
        assert index_view(file.index) == index_view(expected.index)
        compare_stores(expected, file, summaries=10)
    saved = json.loads((index_dir / "101_FAKE_RICH.json").read_text(encoding="utf-8"))
    assert saved["head"] == {"code": parquet_store.SCAN_CODE} and saved["facts"]["sha256"] == expected.table.meta.file_sha256


# ---------------------------------------------------------------- review of S3: more conditions, the saved scan


def test_a_wrong_column_type_is_ineligible_and_both_routes_agree(tmp_path):
    path = write(tmp_path / "data", "FAKE_TYPES", [_row(str(y), "T1", [("성별", sc, sl)], str(y)) for y in (2020, 2021)
                                                   for sc, sl in SEXES], 2)
    table = pq.read_table(path)
    pq.write_table(table.set_column(table.schema.get_field_index("value"), "value",
                                    pa.array([str(v) for v in table.column("value").to_pylist()])), path,
                   row_group_size=2)
    assert parquet_store.scan(path, parquet.sha256_of).failure == "types"
    try:
        expected = parquet.load_table(path, big_rows=10**9)
    except SchemaError as exc:
        with pytest.raises(SchemaError) as again:
            parquet.load_table(path, big_rows=0)
        assert str(again.value) == str(exc)
    else:
        found = parquet.load_table(path, big_rows=0)
        assert not found.file_backed and found.cells == expected.cells


def test_a_pipe_in_the_tbl_id_is_ineligible(tmp_path, monkeypatch):
    rows = [_row(str(y), "T1", [("성별", sc, sl)], str(y)) for y in (2020, 2021) for sc, sl in SEXES]
    path = write(tmp_path / "data", "FAKE|PIPE", rows, 2, key="101_FAKE_PIPE")  # Windows allows no "|" in a file name
    assert parquet_store.scan(path, parquet.sha256_of).failure is None  # the rows are fine: the catalog's id is not
    assert parquet.why_not_file_backed(path) == ("tbl_id", parquet_store.ELIGIBILITY["tbl_id"])
    found = parquet.load_table(path, big_rows=0)
    assert not found.file_backed and found.cells == parquet.load_table(path, big_rows=10**9).cells
    monkeypatch.setattr(parquet, "ROW_LIMIT", 3)
    with pytest.raises(SchemaError, match=re.escape(parquet_store.ELIGIBILITY["tbl_id"])):
        parquet.load_table(path, big_rows=0)


def test_why_not_file_backed_names_the_condition_from_the_saved_scan(tmp_path, monkeypatch, index_dir):
    good = write(tmp_path / "data", "FAKE_RICH", rich_rows(), 4)
    bad = write(tmp_path / "data", "FAKE_BAD", dict(_ineligible_cases())["level"], 2)
    assert parquet.why_not_file_backed(good) is None
    assert parquet.why_not_file_backed(bad) == ("level", parquet_store.ELIGIBILITY["level"])
    assert (index_dir / "101_FAKE_BAD.json").exists()
    monkeypatch.setattr(parquet_store, "scan", lambda *a: pytest.fail("scanned again"))
    assert parquet.why_not_file_backed(str(bad))[0] == "level" and parquet.why_not_file_backed(good) is None


def _grid_rows(periods: int, descs: int) -> list[dict]:
    """Every descriptor in one or two periods, descriptors arriving as the periods go (the grid grows both ways)."""
    rows = [_row(str(1990 + p), "T1", [("분류", f"{d:05d}", f"라벨{d}")], str(d)) for p in range(periods)
            for d in range(descs) if (p + d) % 7 == 0 or p == d % periods]
    return sorted(rows, key=lambda r: (r["period"], r["dim1_code"]))


def test_the_grid_limit_is_the_number_of_combinations_not_the_room_grown(tmp_path, monkeypatch):
    path = write(tmp_path / "data", "FAKE_GRID", _grid_rows(20, 300), 50)
    combinations = (20 + 1) * 300  # the period ids (null is one) x the descriptors
    monkeypatch.setattr(parquet_store, "MAX_KEY_CELLS", combinations)
    facts = parquet_store.scan(path, parquet.sha256_of)
    assert facts.failure is None and len(facts.periods) == 20 and facts.descriptor_count == 300
    monkeypatch.setattr(parquet_store, "MAX_KEY_CELLS", combinations - 1)
    assert parquet_store.scan(path, parquet.sha256_of).failure == "grid"
    grid = parquet_store._Grid()  # growth is clamped to the limit, never below the counts
    monkeypatch.setattr(parquet_store, "MAX_KEY_CELLS", 1000)
    for periods, descs in ((3, 100), (5, 150), (6, 160), (40, 25), (1, 1000)):
        grid.fit(periods, descs, 1)
        assert grid.mask.shape[0] >= periods and grid.mask.shape[1] >= descs and grid.mask.size <= 1000
    with pytest.raises(parquet_store.Ineligible):
        grid.fit(2, 501, 1)


def test_truncated_statistics_are_kept_for_pruning():
    class Stats:
        has_min_max, min, max, has_null_count, null_count = True, "a", "b", True, 0
        is_min_value_exact = is_max_value_exact = None  # what an unset attribute returns

    class Chunk:
        path_in_schema, statistics = "period", Stats()

    class Group:
        num_columns = 1

        def column(self, n):
            return Chunk()

    assert parquet_store.ParquetStore._stats_of(Group()) == {"period": ("a", "b", 0)}


def test_only_read_errors_become_a_schema_error(rich_path, monkeypatch):
    for error, expected in ((pa.ArrowInvalid("bad footer"), SchemaError), (OSError("gone"), SchemaError),
                            (MemoryError(), MemoryError), (ValueError("a bug"), ValueError)):
        def fail(*args, error=error):
            raise error
        monkeypatch.setattr(parquet_store, "load_facts", fail)
        with pytest.raises(expected) as raised:
            parquet.load_table(rich_path, big_rows=0)
        assert type(raised.value) is expected or isinstance(raised.value, expected)
        if expected is SchemaError:
            assert "cannot read Parquet" in str(raised.value)


def _replace_keeping_mtime(path: Path, rows: list[dict], group_rows: int = 4) -> None:
    st = path.stat()
    other = write(path.parent.parent / "other", path.stem.split("_", 1)[1], rows, group_rows)
    path.write_bytes(other.read_bytes())
    os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns))


def _relabel(rows: list[dict], label: str) -> list[dict]:
    return [r | {"dim2_label": label} if r.get("dim2_code") == "1" else r for r in rows]


def test_the_saved_scan_is_used_only_for_the_same_bytes_and_code(tmp_path, index_dir, monkeypatch):
    path = write(tmp_path / "data", "FAKE_RICH", rich_rows(), 4)
    first = parquet.load_table(path, big_rows=0)
    saved = (index_dir / "101_FAKE_RICH.json").read_text(encoding="utf-8")
    assert str(tmp_path) not in saved and "stamp" not in saved  # nothing local: the folder may be shared
    scans = []
    real = parquet_store.scan
    monkeypatch.setattr(parquet_store, "scan", lambda *a: scans.append(1) or real(*a))
    os.utime(path, ns=(1, 1))  # another date, the same bytes (a copy on another PC): reused
    assert parquet.load_table(path, big_rows=0).meta.file_sha256 == first.meta.file_sha256 and not scans
    _replace_keeping_mtime(path, _relabel(rich_rows(), "남성"))  # other bytes, same date: scanned again
    changed = parquet.load_table(path, big_rows=0)
    assert scans == [1] and changed.meta.file_sha256 == parquet.sha256_of(path) != first.meta.file_sha256
    assert index_of(changed).axes["성별"].labels["1"] == "남성"
    monkeypatch.setattr(parquet_store, "SCAN_CODE", "another version of the code")
    parquet.load_table(path, big_rows=0)
    assert scans == [1, 1]


def test_the_descriptors_file_must_be_the_one_of_its_scan(tmp_path, index_dir):
    path = write(tmp_path / "data", "FAKE_RICH", rich_rows(), 4)
    parquet.load_table(path, big_rows=0)
    descriptors = index_dir / "101_FAKE_RICH.descriptors.parquet"
    old = descriptors.read_bytes()
    _replace_keeping_mtime(path, _relabel(rich_rows(), "남성"))  # the same number of descriptors
    parquet.load_table(path, big_rows=0)
    descriptors.write_bytes(old)  # the other file's descriptors beside this file's facts
    table = parquet.load_table(path, big_rows=0)
    assert index_of(table).axes["성별"].labels["1"] == "남성"
    assert index_view(index_of(table)) == index_view(index_of(parquet.load_table(path, big_rows=10**9)))


def test_a_table_missing_from_tables_csv_fails_before_any_scan(tmp_path, monkeypatch, index_dir):
    path = write(tmp_path / "data", "FAKE_RICH", rich_rows(), 4)
    (tmp_path / "data" / "tables.csv").unlink()
    monkeypatch.setattr(parquet_store, "scan", lambda *a: pytest.fail("scanned"))
    with pytest.raises(SchemaError, match="tables.csv"):
        parquet.load_table(path, big_rows=0)
    assert not index_dir.exists()


def test_an_unwritable_index_folder_is_logged_once(tmp_path, monkeypatch, caplog):
    path = write(tmp_path / "data", "FAKE_RICH", rich_rows(), 4)
    blocker = tmp_path / "a_file"
    blocker.write_text("x")
    monkeypatch.setattr(parquet_store, "INDEX_DIR", blocker / "index")
    with caplog.at_level(logging.WARNING, logger=parquet_store.__name__):
        tables = [parquet.load_table(path, big_rows=0) for _ in range(3)]
    warnings = [r for r in caplog.records if r.name == parquet_store.__name__]
    assert len(warnings) == 1 and str(blocker / "index") in warnings[0].getMessage()
    assert tables[0].store.facts.descriptors is not None  # not saved: held
    assert index_view(index_of(tables[0])) == index_view(index_of(parquet.load_table(path, big_rows=10**9)))


def test_the_store_does_not_hold_the_saved_descriptors(tmp_path, index_dir):
    path = write(tmp_path / "data", "FAKE_RICH", rich_rows(), 4)
    memory = index_view(index_of(parquet.load_table(path, big_rows=10**9)))
    for _ in range(2):  # the fresh scan, then the saved one
        table = parquet.load_table(path, big_rows=0)
        facts = table.store.facts
        assert facts.descriptors is None and facts.descriptors_file == index_dir / "101_FAKE_RICH.descriptors.parquet"
        assert index_view(index_of(table)) == memory
    assert set(facts.to_json()) == {f.name for f in dataclasses.fields(facts)} - {"descriptors", "descriptors_file",
                                                                                                "drop", "conflict"}
    cells = list(table.store.index_cells())
    facts.descriptors_file.unlink()  # gone after the load: the file is scanned again for them
    assert list(table.store.index_cells()) == cells
    _replace_keeping_mtime(path, _relabel(rich_rows(), "남성"))  # and the file changed too: an error, not a mix
    with pytest.raises(RuntimeError, match="changed after it was loaded"):
        list(table.store.index_cells())


def test_the_memos_are_lru_and_the_series_memo_counts_cells(tmp_path, monkeypatch):
    monkeypatch.setattr(parquet_store, "MEMO_CELLS", 2)
    monkeypatch.setattr(parquet_store, "MEMO_SERIES_CELLS", 6)
    memory, file = load_both(write(tmp_path / "data", "FAKE_RICH", rich_rows(), 4))
    store = file.table.store
    a, b, c = [cell_key(cell) for cell in memory.table.cells[:3]]
    store.cell(a), store.cell(b), store.cell(a)  # a hit makes a the newest
    store.cell(c)  # drops b
    before = store.reads
    store.cell(a)
    assert store.reads == before
    store.cell(b)
    assert store.reads == before + 1
    s1, s2 = ("T1", (("성별", "1"), ("행정구역별", "11"))), ("T1", (("성별", "2"), ("행정구역별", "21")))
    assert len(store.series(*s1)) == 5 and len(store.series(*s2)) == 5
    assert store._series.size <= 6 and s1 not in store._series and s2 in store._series


def test_no_handle_on_the_file_stays_open_between_requests(rich_path, monkeypatch):
    fds = Path("/proc/self/fd")
    if not fds.exists():
        pytest.skip("needs /proc (Linux)")

    def open_on_the_file() -> int:
        found = 0
        for fd in fds.iterdir():
            try:
                found += Path(os.readlink(fd)) == rich_path.resolve()
            except OSError:
                pass
        return found

    file = load_resources(parquet.load_table(rich_path, big_rows=0))
    store = file.table.store
    assert open_on_the_file() == 0
    for parallel in (10**9, 0):
        monkeypatch.setattr(parquet_store, "PARALLEL_ROWS", parallel)
        store.cells_at([("T1", (("성별", "0"), ("행정구역별", "00")), "2023")])
        store.series("T1", (("성별", "1"),))
        store.summary({"성별": {"0"}}, store.periods)
        assert open_on_the_file() == 0, parallel


def test_requests_convert_only_the_rows_they_return(tmp_path):
    """The column masks alone keep the cross product of the codes asked for, and a key that names fewer levels than
    the rows have keeps every row of the item: the whole-key mask leaves only the rows asked for."""
    memory, file = load_both(write(tmp_path / "data", "FAKE_RICH", rich_rows(scenario=False), 4))
    store = file.table.store
    years = ["2020", "2021", "2022", "2023"]
    diagonal = [("T1", (("성별", sc), ("행정구역별", rc)), year) for (rc, _), (sc, _), year in zip(REGIONS, SEXES, years)]
    refs = [c.source_ref for c in memory.table.cells[::7]]
    requests = [lambda: store.cells_at(diagonal), lambda: store.by_refs(refs),
                lambda: store.series_at([("T1", ()), ("T1", (("성별", "0"),)), ("T2", (("행정구역별", "11"),))]),
                lambda: store.cells_at([("T1", (), "2023"), ("T1", (("성별", "1"),), "2023")]),
                lambda: store.by_refs([refs[0].replace("|2020", "|2021|x"), "FAKE_RICH|T1|00|2020"])]
    for request in requests:
        before = store.rows_converted
        got = request()
        returned = sum(len(v) if isinstance(v, tuple) else 1 for v in got.values())
        assert store.rows_converted - before == returned, request
    assert len(store.cells_at(diagonal)) == 3
    before = store.rows_masked  # a level no key names: only rows without a code on it are read past the key columns
    got = store.series_at([("T1", (("행정구역별", "21"),)), ("T2", (("행정구역별", "21"),))])
    assert list(got) == [("T2", (("행정구역별", "21"),))] and store.rows_masked - before == len(got["T2", (("행정구역별", "21"),)]) == 5


def test_row_groups_are_pruned_and_those_of_losing_codes_skipped(tmp_path):
    rows = sorted(rich_rows(scenario=False), key=lambda r: (r["period"] or "", r["item_id"]))
    memory, file = load_both(write(tmp_path / "data", "FAKE_RICH", rows, 4))
    store = file.table.store
    groups = store._groups
    store.cell(("T1", (("성별", "0"), ("행정구역별", "00")), "2021"))
    assert 1 <= store.row_groups_read <= 3 < groups
    cycle = [r for r in cycle_rows() if r["prd_se"] == "Q"] + [r for r in cycle_rows() if r["prd_se"] != "Q"]
    memory, file = load_both(write(tmp_path / "cycle", "FAKE_CYCLE", cycle, 4))
    store = file.table.store
    assert store.facts.losing == ["Q"] and len(store._lost) == 6  # 24 quarter rows in 6 groups of their own
    store.summary({}, store.periods)
    assert store.row_groups_read == store._groups - 6
    compare_stores(memory, file, summaries=20)


def test_the_parallel_reader_answers_alike(tmp_path, monkeypatch):
    monkeypatch.setattr(parquet_store, "PARALLEL_ROWS", 0)
    assert_same(write(tmp_path / "data", "FAKE_RICH", rich_rows(1), 3), summaries=40)
    assert_same(write(tmp_path / "cycle", "FAKE_CYCLE", cycle_rows(), 5), summaries=20)


def _order_cases() -> list[list[dict]]:
    def sex(code: str, label: str) -> list[tuple]:
        return [("성별", code, label)]

    return [
        [_row("", "T1", sex("0", "계"), ""), _row("2020", "T1", sex("1", "남자"), "5"),  # first seen in a rule-1 row
         _row("2020", "T1", sex("0", "계"), "7"), _row("2021", "T1", sex("1", "남자"), "6")],
        [_row("202001", "T1", sex("0", "계"), "1", prd_se="Q"), _row("202001", "T1", sex("1", "남자"), "2", prd_se="M"),
         _row("202001", "T1", sex("0", "계"), "3", prd_se="M"), _row("202002", "T1", sex("1", "남자"), "4", prd_se="M")],
        [_row("202001", "T1", sex("0", "합계"), "1", prd_se="Q"), _row("202001", "T1", sex("1", "남자"), "2", prd_se="M"),
         _row("202001", "T1", sex("0", "계"), "3", prd_se="M")],  # the first label comes from a kept row only
        # tidy rule 3 drops the first row of the descriptor naming 다른 분류 (the key reads the level's code, not its
        # name): the descriptor takes its place from its first kept row, after the one naming 연령별
        [_row("2020", "T1", sex("1", "남자"), "1"), _row("2020", "T1", [("다른 분류", "1", "남자")], "1"),
         _row("2021", "T1", [("연령별", "1", "남자")], "2"), _row("2022", "T1", [("다른 분류", "1", "남자")], "3")],
    ]


@pytest.mark.parametrize("case", range(len(_order_cases())))
def test_first_seen_order_counts_kept_rows_only(tmp_path, case):
    assert_same(write(tmp_path / "data", "FAKE_ORDER", _order_cases()[case], 1), summaries=10)


def _fuzz_rows(rng: random.Random) -> list[dict]:
    """The reviewer's generator (made-up tables): levels with one or two names, odd codes, labels and periods."""
    names = ["성별", "행정구역별", "연령별", "출산율 가정별", "항목", " ", ""]
    level_names = [rng.sample(names, rng.choice([1, 1, 1, 2])) for _ in range(rng.choice([0, 1, 1, 2, 2, 3, 8]))]
    codes = ["0", "1", "11", "", None, "None", "가", "A"]
    labels = ["계", "남자", "전국 (명)", None, "", "합계출산율", " 계 "]
    periods = rng.sample(["2020", "2021", "202001", "202002", "2022", "", " 2020", "2020 ", None], rng.randint(1, 4))
    if rng.random() < 0.7:
        periods = [p for p in periods if p and p.strip() == p] or ["2020"]
    items = rng.sample(["T1", "T2", "", None, "None"], rng.randint(1, 2))
    if rng.random() < 0.7:
        items = [i for i in items if i] or ["T1"]
    prds = rng.choice([["Y"], ["Y"], ["M", "Q"], ["Y", "M", None], ["M", "Q", "Y", ""], ["A", "Y"]])
    rows = []
    for _ in range(rng.randint(1, 6)):
        levels = [(rng.choice(cand) if rng.random() < 0.9 else None, rng.choice(codes), rng.choice(labels))
                  for cand in level_names]
        item, unit = rng.choice(items), rng.choice(["명", None, "", " % "])
        name = rng.choice(["가상 인구", "가상 비율 (%)", " "]) if rng.random() < 0.3 else "가상 인구"
        for period in periods:
            if rng.random() < 0.75:
                rows.append(_row(period, item, levels, rng.choice(["1", "2,000", "-", "x", "560p", "", None, "3.5"]),
                                 prd_se=rng.choice(prds), unit=unit, item_name=name))
    if rows and rng.random() < 0.6:  # tidy rule 3: copies, equal or not, some copied again
        for _ in range(rng.randint(1, 2 * len(rows))):
            rows.append(duplicate(rng.choice(rows), rng.choice(DUPLICATES)))
    rng.shuffle(rows)
    return rows or [_row("2020", "T1", [], "1")]


def test_fixed_seed_fuzz_answers_alike_or_falls_back(tmp_path_factory, monkeypatch):
    outcomes = {"same": 0, "memory": 0, "refused": 0}
    for seed in range(120):
        rng = random.Random(seed)
        if seed % 4 == 0:
            monkeypatch.setattr(parquet_store, "PARALLEL_ROWS", 0)
        else:
            monkeypatch.setattr(parquet_store, "PARALLEL_ROWS", 10**9)
        path = write(tmp_path_factory.mktemp("fuzz"), "FAKE_FUZZ", _fuzz_rows(rng), rng.choice([1, 2, 3, 5, 100]))
        try:
            memory = load_resources(parquet.load_table(path, big_rows=10**9))
        except SchemaError as exc:
            with pytest.raises(SchemaError) as again:
                parquet.load_table(path, big_rows=0)
            assert str(again.value) == str(exc), seed
            outcomes["refused"] += 1
            continue
        file = load_resources(parquet.load_table(path, big_rows=0))
        if not file.table.file_backed:
            assert same(file.table.cells, memory.table.cells) and file.table.meta == memory.table.meta, seed
            outcomes["memory"] += 1
            continue
        assert file.table.meta == memory.table.meta, seed
        assert index_view(file.index) == index_view(memory.index), seed
        compare_stores(memory, file, summaries=15, seed=seed)
        outcomes["same"] += 1
    assert outcomes["same"] >= 30, outcomes


# ---------------------------------------------------------------- many values, a small stack


def test_requests_with_thousands_of_values_run_on_a_one_megabyte_stack(tmp_path):
    """No request builds a filter expression (pyarrow recurses over a long OR and overflows the C stack at a few
    hundred terms on 1 MB, the Windows default): 1,500 periods and 3,000 keys, refs, series and codes."""
    periods = [f"{1900 + n // 12}{n % 12 + 1:02d}" for n in range(1500)]
    rows = [_row(p, "T1", [("성별", sc, sl)], str(n + int(sc)), prd_se="M") for n, p in enumerate(periods)
            for sc, sl in SEXES[:2]]
    memory, file = load_both(write(tmp_path / "data", "FAKE_LONG", rows, 400))
    mem, fil = store_of(memory.table), store_of(file.table)
    keys = [cell_key(c) for c in memory.table.cells]
    refs = [c.source_ref for c in memory.table.cells]
    series = [("T1", (("성별", sc),)) for sc, _ in SEXES]
    codes = {ITEM_AXIS: {"T1", *(f"X{n}" for n in range(1500))}, "성별": {"0", "1", "", *(str(n) for n in range(2, 1500))}}
    outcome: dict = {}

    def run() -> None:
        try:
            outcome["got"] = (fil.cells_at(keys), fil.by_refs(refs), fil.series_at(series), fil.summary(codes, periods),
                              index_of(file.table).periods)
        except BaseException as exc:  # handed to the test thread
            outcome["error"] = exc

    old = threading.stack_size(1 << 20)
    try:
        worker = threading.Thread(target=run)
        worker.start()
        worker.join()
    finally:
        threading.stack_size(old)
    assert "error" not in outcome, outcome.get("error")
    got = outcome["got"]
    assert got[0] == mem.cells_at(keys) and got[1] == mem.by_refs(refs)
    assert {k: tuple(v) for k, v in got[2].items()} == {k: tuple(v) for k, v in mem.series_at(series).items()}
    assert got[3] == mem.summary(codes, periods) and got[3].count == 3000 and got[4] == sorted(periods)


# ---------------------------------------------------------------- the provided tables


VALUES = workspace_root() / "data" / "values"
PARITY_ROWS = 300_000  # the provided tables up to this size are compared both ways


def _provided() -> list[Path]:
    if not VALUES.exists():
        return []
    return sorted(p for p in VALUES.glob("*.parquet") if pq.read_metadata(p).num_rows <= PARITY_ROWS)


@pytest.mark.skipif(not VALUES.exists(), reason="provided data is not in the repo (gitignored); place it at data/")
def test_provided_tables_answer_alike_both_ways(tmp_path, monkeypatch, record_property):
    monkeypatch.setattr(parquet_store, "INDEX_DIR", tmp_path / "index")  # the workspace may be read-only
    compared, ineligible, refused = 0, {}, {}
    for path in _provided():
        try:
            memory = load_resources(parquet.load_table(path, big_rows=10**9))
        except SchemaError as exc:  # the memory route refuses it: so must the file route
            refused[path.stem] = str(exc)[:80]
            with pytest.raises(SchemaError):
                parquet.load_table(path, big_rows=0)
            continue
        file = load_resources(parquet.load_table(path, big_rows=0))
        if not file.table.file_backed:
            ineligible[path.stem] = parquet_store.scan(path, parquet.sha256_of).failure
            assert file.table.meta == memory.table.meta
            continue
        assert file.table.meta == memory.table.meta, path.stem
        assert index_view(file.index) == index_view(memory.index), path.stem
        compare_stores(memory, file, summaries=50, sample=50)
        compared += 1
    record_property("parity", {"compared": compared, "ineligible": ineligible, "refused": refused})
    print(f"\nparity: compared {compared}, ineligible {ineligible}, refused by both routes {len(refused)}")
    assert compared
