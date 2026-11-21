"""A table read from its Parquet file on demand (T1-31 S3): the index scan, the saved scan, and the file cell store.

A table over adapters.parquet.BIG_TABLE_ROWS is never loaded whole (every row as a Python Cell costs ~3.4 KB; the
provided files go up to 111 million rows). Instead:

1. scan(): one pass over the file, row group by row group, column by column (dictionary ids and numpy, no Python value
   per row), gathers the facts the memory route derives from all rows: the prd_se codes, what prepare.convert.tidy_rows
   rules 1 and 2 drop (and their counts for the note), the distinct descriptors (everything prepare.index reads from a
   cell but period and value) in first-seen order of the kept rows, the sorted periods of the kept rows, the file
   sha256, and whether the table is eligible (below).
   Tidy rule 3 (rows that share tidy_rows' key after rules 1 and 2): the first pass finds the keys with two kept rows
   or more. Only when there is one does a second pass (_merge_pass) read the file again, with the value columns of
   those keys' rows only, and settle the rule as tidy_rows does: the later rows of a key are dropped, a key whose rows
   do not all carry the first row's (value_raw, value, value_kind) is a conflict (its first row is read with value_raw
   "", value None, value_kind "blank"), and the same_rows / conflict_cells counts go into the note. "Equal" is Python's
   equality of the tuples tidy_rows puts in a set: None equals None, 0.0 equals -0.0, and a NaN value equals nothing
   (to_pylist gives each NaN its own float object), so a key with a NaN value on any of its rows is a conflict. The
   other facts (periods, descriptors and their order, the conditions below) are then those of the rows rule 3 keeps.
2. The facts are saved under <data dir>/index/ (INDEX_DIR overrides it): data facts only; the built TableIndex is
   saved apart, keyed by its code (the saved table index below). A saved scan is used only when its file's sha256
   is the current file's (computed at every load) and it was made by the same code (SCAN_CODE, a hash of this module
   and prepare/convert.py); the descriptors file carries the sha256 too. Nothing in it names a local path: the folder
   may be shared between PCs. A table with rule-3 rows also has <key>.merge.parquet: the sorted row numbers (0-based,
   over the whole file) of the rows rule 3 drops (conflict false) and of the first rows of conflict keys (conflict
   true), with the sha256 in its schema metadata like the descriptors file; it is loaded with the facts and held by
   the store (8 bytes a row). A table without such rows has no such file and holds nothing for it. On load the merge
   file must hold what the facts say (_read_merge), else the file is scanned again. When a source file cannot be
   read for the code keys (prepare.store.source_hash gives None), nothing is saved or reused: every load scans.
3. ParquetStore answers each CellStore request by reading only the matching rows: row groups are skipped by their
   footer statistics, key columns are masked with pyarrow.compute (is_in on arrays: no filter expression, so no deep
   OR, which crashes pyarrow at a few hundred terms on a 1 MB stack), the rows left are checked against the whole keys
   (or refs) asked for on the Arrow side, converted by the real prepare.convert.cells_of + prepare.clean.clean_cell,
   and an exact check on each cell's key or ref keeps only the cells asked for. Bounded LRU memos keep what was read.
   Each read opens the file (with the footer read at load) and closes it: no handle is held between requests. Every
   read masks out the rows rule 3 drops and blanks the values of the first rows of conflict keys (the counts of
   summary included) by their row numbers.

Eligibility: the file route answers exactly as the memory route only when a key maps to one row and a ref parses back
into one key. scan() checks, over the rows kept after tidy rules 1, 2 and 3 (ELIGIBILITY lists the messages):
- rule 3 needs at most MAX_MERGE_ROWS rows in keys of two rows or more and MAX_MERGE_BYTES of their first rows' raw
  values (the memory its second pass holds: see there);
- no two kept rows share a CellKey or a source_ref (so cell, by_ref and series each find at most one row per key and
  period); both are checked over the rows rule 3 keeps, which never share tidy_rows' key, so what they catch is two
  rows of different tidy keys (a null code and '', or a level named on one row only);
- no '|' in an item id, a named level's code or a period (the table's tbl_id is checked at load);
- every period is non-empty, not null, and has no surrounding whitespace (period_raw is the raw period);
- item ids are all null or none null (null and "" both read as item code "");
- each category name sits at one level on every row (a name maps to one level column);
- every cell has a non-empty item name (prepare.schemas.validate; its unit check cannot fail: infer_unit always
  gives a unit);
- at most MAX_CODES prd_se values, at most MAX_KEY_CELLS descriptor x period cells, the expected column types.
A table that fails is not file-backed: adapters.parquet loads it whole when it is within ROW_LIMIT, else refuses it
naming the condition.
"""

from __future__ import annotations

import bisect
import io
import json
import logging
import os
import tempfile
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from collections.abc import Callable, Collection, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field, fields
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from kosis_chat.prepare import convert
from kosis_chat.prepare.clean import clean_cell
from kosis_chat.prepare.convert import DIM_LEVELS, PARQUET_COLUMNS, _blank, cells_of
from kosis_chat.prepare.schemas import Cell, TableMeta
from kosis_chat.prepare.store import (ITEM_AXIS, CellKey, DimKey, RangeSummary, SeriesKey, cell_key, dim_key,
                                      source_hash)

log = logging.getLogger(__name__)

INDEX_DIR: Path | None = None  # where scans are saved; None: <data dir>/index (tests and tools set it)
MAX_CODES = 8  # prd_se values (null included) one scan tells apart: one bit each
# Period ids x descriptors of the scan's two one-byte grids. At the limit the scan holds about 3 bytes a combination
# (the grids, plus a third while one grows or `present` in _settle): measured 2.6 GB peak for 749 periods x 1,000,000
# descriptors with the limit at 750,000,000 (the rest is the descriptors' own cost, about 0.5 KB each).
MAX_KEY_CELLS = 750_000_000
# Tidy rule 3 (_merge_pass), what one scan settles: at most MAX_MERGE_ROWS rows in tidy keys of two kept rows or
# more, and at most MAX_MERGE_BYTES of the raw values (value_raw, Arrow large_string: its bytes and 8 more) of those
# keys' first rows, which the second pass holds to compare the later rows with. Past either: Ineligible("merge_rows"),
# before the second pass when the first already counts more than MAX_MERGE_ROWS / 2 such keys.
# The pass holds about 50 bytes for each such key (first row, descriptor, value, null flag, kind, raw position,
# conflict flag, later-row count), the raw values, and 8 bytes for each row it drops (the store holds those too); it
# reads the value columns MERGE_BATCH_ROWS rows at a time. Scan peaks measured (Linux, VmHWM, MiB; every key two
# rows, copy 2 of every key after copy 1, row groups of 500,000): 20 million rows in 10 million keys with short raw
# values 1,664 (899 before the second pass); the same with 72-character raw values (763 MiB held, just within
# MAX_MERGE_BYTES) 2,379, and with 99-character ones refused at 2,223; 2 million rows in 1 million keys with
# 760-character raw values 1,836; 5 million rows in 2.5 million keys with 200-character ones 1,492. 30 million rows
# in 15 million keys are refused after the first pass (17 s, 796).
MAX_MERGE_ROWS = 20_000_000
MAX_MERGE_BYTES = 768 << 20
MEMO_CELLS = 50_000  # cells (by key and by ref) a store keeps
MEMO_SERIES_CELLS = 50_000  # cells of the series a store keeps (an empty series counts as one)
MEMO_SUMMARIES = 256
MERGE_BATCH_ROWS = 65_536  # rows of a row group whose value columns the second pass reads at once
VALUE_COLUMNS = ["value_raw", "value", "value_kind"]
READERS = 4  # row groups a request reads at once (pyarrow decodes and filters without the GIL)
PARALLEL_ROWS = 1_000_000  # below this many rows to read, a request reads in its own thread

LEVEL_PARTS = [(level, part) for level in DIM_LEVELS for part in ("name", "code", "label")]
DESC_COLUMNS = ["item_id", "item_name", "unit"] + [f"dim{level}_{part}" for level, part in LEVEL_PARTS]
SCAN_COLUMNS = ["tbl_id", "prd_se", "period"] + DESC_COLUMNS
STRING_COLUMNS = [name for name in PARQUET_COLUMNS if name != "value"]  # read as dictionaries, cast before to_pylist
_UNSET = np.iinfo(np.int64).max


def _code_hash() -> str | None:
    """The sha256 of the source text of the modules the saved facts depend on (this module and prepare/convert.py;
    line endings made alike: a checkout on Windows and one on Linux give one hash), so that a code update invalidates
    old scans by itself. None when either cannot be read or is not a .py file (prepare.store.source_hash): then no
    scan is saved or reused (load_facts) and no table index either (ParquetStore._index_head)."""
    files = [globals().get("__file__"), getattr(convert, "__file__", None)]
    return source_hash(((f"{Path(f).parent.name}/{Path(f).name}", Path(f)) for f in files if f and f.endswith(".py")),
                       required=2)


SCAN_CODE = _code_hash()

ELIGIBILITY = {
    "types": "a required column is missing or has an unexpected type",
    "codes": f"more than {MAX_CODES} period types (prd_se values)",
    "grid": f"more than {MAX_KEY_CELLS:,} combinations of a period and a value description",
    "descriptors": "too many different items and categories",
    "merge_rows": f"too many rows share their item, categories and period with another row (more than "
                  f"{MAX_MERGE_ROWS:,} such rows, or more than {MAX_MERGE_BYTES >> 20:,} MB of their values to "
                  f"compare): too many to settle within the scan's memory budget",
    "period": "a period that is empty or has spaces around it",
    "pipe": "a '|' inside an item id, a category code or a period",
    "item_null": "an item id that is missing on some rows only",
    "level": "a category that sits in more than one category column",
    "cell_key": "two values with the same item, category codes and period",
    "ref": "two values with the same source reference",
    "item_name": "a value with no item name",
    "tbl_id": "a '|' inside the table id",
}


class Ineligible(Exception):
    def __init__(self, condition: str) -> None:
        super().__init__(ELIGIBILITY[condition])
        self.condition = condition


@dataclass
class Facts:
    """What scan() learns about one file. The memory route's derivations from all rows, in data terms."""

    sha256: str
    rows: int
    failure: str | None = None  # ELIGIBILITY key of the first failed condition; then only sha256 and rows are set
    codes_all: list[str] = field(default_factory=list)  # period_codes of every row (TableMeta period_type, note)
    tbl_ids: list[str] = field(default_factory=list)  # str() of every row's tbl_id (CHECK_CATALOG_TBL_ID)
    blank_rows: int = 0  # tidy rule 1
    cycle_rows: int = 0  # tidy rule 2
    same_rows: int = 0  # tidy rule 3: later rows equal to their key's first row
    conflict_cells: int = 0  # tidy rule 3: keys whose rows differ (the first row is read blank)
    merge_rows: int = 0  # rows of the merge file: len(drop) + len(conflict)
    losing: list[str | None] = field(default_factory=list)  # prd_se values rule 2 drops
    periods: list[str] = field(default_factory=list)  # sorted periods of the kept rows
    levels: dict[str, int] = field(default_factory=dict)  # category name -> its level
    layouts: list[list[int]] = field(default_factory=list)  # the distinct sets of named levels
    item_null: bool = False  # every item id is null
    descriptor_count: int = 0  # the kept rows' distinct descriptors
    # DESC_COLUMNS of those descriptors in first-seen order: held only while they are not saved (descriptors_file)
    descriptors: pa.Table | None = None
    descriptors_file: Path | None = None  # the saved descriptors (not part of the saved facts)
    # tidy rule 3, sorted row numbers over the whole file (int64; None when the table has no such row): the rows it
    # drops, and the first rows of conflict keys. Saved in the merge file, not in the facts; always held.
    drop: np.ndarray | None = None
    conflict: np.ndarray | None = None

    def to_json(self) -> dict:
        """The saved facts: every field but the descriptor table, its file and the rule-3 rows (no deep copy)."""
        return {f.name: getattr(self, f.name) for f in fields(self) if f.name not in NOT_SAVED}


NOT_SAVED = ("descriptors", "descriptors_file", "drop", "conflict")


# ---------------------------------------------------------------- the scan


class _Vocab:
    """Global ids of one column's strings across row groups: 0 is null, then first-seen order."""

    def __init__(self) -> None:
        self.ids: dict[str, int] = {}
        self.values: list[str | None] = [None]

    def ids_of(self, column: pa.ChunkedArray) -> np.ndarray:
        parts = []
        for chunk in column.chunks:
            if chunk.null_count == len(chunk):
                parts.append(np.zeros(len(chunk), dtype=np.int64))
                continue
            if not pa.types.is_dictionary(chunk.type):
                chunk = pc.dictionary_encode(chunk)
            words = chunk.dictionary
            known = pc.index_in(words, value_set=self._known()) if len(self.values) > 1 else pa.nulls(len(words))
            if known.null_count:  # words this column has not shown before
                for word in words.filter(known.is_null()).to_pylist():
                    if word not in self.ids:
                        self.ids[word] = len(self.values)
                        self.values.append(word)
                self._array = None
                known = pc.index_in(words, value_set=self._known())
            lut = np.append(pc.add(known, 1).to_numpy(zero_copy_only=False).astype(np.int64), 0)
            indices = chunk.indices if not chunk.null_count else pc.fill_null(chunk.indices, len(words))
            parts.append(lut[indices.to_numpy(zero_copy_only=False)])
        return np.concatenate(parts) if parts else np.zeros(0, dtype=np.int64)

    def _known(self) -> pa.Array:
        if getattr(self, "_array", None) is None:
            self._array = pa.array(self.values[1:], type=pa.string())
        return self._array


def _encode(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(distinct values in first-seen order, each value's position among them): a hash pass, no sort."""
    encoded = pc.dictionary_encode(pa.array(values))
    return encoded.dictionary.to_numpy(), encoded.indices.to_numpy().astype(np.int64)


def _firsts(inverse: np.ndarray, count: int) -> np.ndarray:
    """The first position of each of `count` groups, given each position's group."""
    first = np.full(count, len(inverse), dtype=np.int64)
    np.minimum.at(first, inverse, np.arange(len(inverse), dtype=np.int64))
    return first


def _check_types(schema: pa.Schema) -> None:
    for name in PARQUET_COLUMNS:
        if name not in schema.names:
            raise Ineligible("types")
        kind = schema.field(name).type
        if pa.types.is_dictionary(kind):
            kind = kind.value_type
        ok = pa.types.is_floating(kind) if name == "value" else (pa.types.is_string(kind) or pa.types.is_large_string(kind))
        if not ok:
            raise Ineligible("types")


class _Grid:
    """Per (period, descriptor): the bits of the prd_se codes of its kept rows (mask), the bits seen twice (dup);
    per (descriptor, code bit): the first row number (first). Grown as ids appear."""

    def __init__(self) -> None:
        self.mask = np.zeros((0, 0), dtype=np.uint8)
        self.dup = np.zeros((0, 0), dtype=np.uint8)
        self.first = np.zeros((0, 0), dtype=np.int64)

    def fit(self, periods: int, descs: int, codes: int) -> None:
        """Room for `periods` x `descs` (Ineligible("grid") past MAX_KEY_CELLS of them). The room grows by half or
        double, clamped so that it never passes the limit; the bits held lie within the counts seen so far."""
        if periods * descs > MAX_KEY_CELLS:
            raise Ineligible("grid")
        pcap, dcap = self.mask.shape
        if periods > pcap or descs > dcap:
            pcap = max(pcap, periods) if periods <= pcap else max(2 * pcap, periods, 16)
            dcap = max(dcap, descs) if descs <= dcap else max(dcap + dcap // 2, descs, 1024)
            if pcap * dcap > MAX_KEY_CELLS:  # less spare room, never less than the counts
                dcap = max(descs, MAX_KEY_CELLS // pcap)
            if pcap * dcap > MAX_KEY_CELLS:
                pcap = max(periods, MAX_KEY_CELLS // dcap)
            for name in ("mask", "dup"):
                old = getattr(self, name)
                new = np.zeros((pcap, dcap), dtype=np.uint8)
                keep_p, keep_d = min(pcap, old.shape[0]), min(dcap, old.shape[1])
                new[:keep_p, :keep_d] = old[:keep_p, :keep_d]
                setattr(self, name, new)
                del old
        if dcap > self.first.shape[0] or codes > self.first.shape[1]:
            first = np.full((dcap, max(codes, self.first.shape[1])), _UNSET, dtype=np.int64)
            first[: self.first.shape[0], : self.first.shape[1]] = self.first
            self.first = first


class _Triples:
    """Global ids of the (a, b, c) column-id triples of a level (name, code, label) or of the item (id, name, unit):
    0 is the all-null triple."""

    def __init__(self) -> None:
        self.ids: dict[tuple[int, int, int], int] = {(0, 0, 0): 0}
        self.rows: list[tuple[int, int, int]] = [(0, 0, 0)]

    def ids_of(self, a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
        if not (a.any() or b.any() or c.any()):
            return np.zeros(len(a), dtype=np.int64)
        distinct, inverse, first = _distinct([a, b, c])
        lut = np.empty(len(first), dtype=np.int64)
        for j, triple in enumerate(zip(a[first].tolist(), b[first].tolist(), c[first].tolist())):
            found = self.ids.get(triple)
            if found is None:
                found = self.ids[triple] = len(self.rows)
                self.rows.append(triple)
            lut[j] = found
        return lut[inverse]

    def table(self) -> np.ndarray:
        return np.array(self.rows, dtype=np.int64).reshape(-1, 3)


def _distinct(columns: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(distinct keys, each row's position among them, each distinct key's first row) of rows of integer columns."""
    key = np.zeros(len(columns[0]), dtype=np.int64)
    bound = 1
    for column in columns:
        low = int(column.min()) if len(column) else 0
        size = int(column.max()) - low + 1 if len(column) else 1
        if size == 1:
            continue
        if bound * size >= 1 << 62:  # keep the combined key exact: renumber it densely first
            distinct, key = _encode(key)
            bound = len(distinct)
        key *= size
        key += column
        if low:
            key -= low
        bound *= size
    distinct, inverse = _encode(key)
    return distinct, inverse, _firsts(inverse, len(distinct))


class _Descriptors:
    """Global descriptor ids: a descriptor is the item triple and the eight level triples (_Triples ids), packed into
    one int64 key and looked up in a few sorted runs (merged as they grow), so a row group costs a binary search per
    distinct descriptor, not a pass over every descriptor seen."""

    WIDTH = 1 + len(DIM_LEVELS)

    def __init__(self) -> None:
        self.item = _Triples()
        self.levels = {level: _Triples() for level in DIM_LEVELS}
        self.rows: list[np.ndarray] = []  # the triple ids of the descriptors, in id order
        self.count = 0
        self.widths: list[int] = [0] * self.WIDTH  # bits of each triple id in a key
        self.runs: list[tuple[np.ndarray, np.ndarray]] = []  # (sorted keys, their descriptor ids)

    def _pack(self, rows: np.ndarray) -> np.ndarray:
        key = np.zeros(len(rows), dtype=np.int64)
        for j, width in enumerate(self.widths):
            key = (key << width) | rows[:, j].astype(np.int64)
        return key

    def _fit(self, rows: np.ndarray) -> None:
        """Widen the key when a triple id outgrows its bits (with room to spare), and repack the known keys."""
        need = [int(rows[:, j].max()).bit_length() if len(rows) else 0 for j in range(self.WIDTH)]
        if all(n <= w for n, w in zip(need, self.widths)):
            return
        widths = [w if n <= w else n + 2 for n, w in zip(need, self.widths)]
        if sum(widths) > 63:
            widths = [max(n, w) for n, w in zip(need, self.widths)]
        if sum(widths) > 63:
            raise Ineligible("descriptors")
        self.widths = widths
        known = self.all_rows()
        if len(known):
            keys = self._pack(known)
            order = np.argsort(keys)
            self.runs = [(keys[order], order.astype(np.int64))]

    def all_rows(self) -> np.ndarray:
        if len(self.rows) != 1:
            self.rows = [np.concatenate(self.rows) if self.rows else np.zeros((0, self.WIDTH), dtype=np.int32)]
        return self.rows[0]

    def ids_of(self, ids: dict[str, np.ndarray]) -> np.ndarray:
        """The descriptor id of every row of one row group (new descriptors get the next ids, in row order)."""
        parts = [self.item.ids_of(ids["item_id"], ids["item_name"], ids["unit"])]
        parts += [self.levels[level].ids_of(*(ids[f"dim{level}_{part}"] for part in ("name", "code", "label")))
                  for level in DIM_LEVELS]
        _, inverse, first = _distinct(parts)
        rows = np.stack([part[first] for part in parts], axis=1).astype(np.int32)
        self._fit(rows)
        keys = self._pack(rows)
        found = np.full(len(keys), -1, dtype=np.int64)
        for run_keys, run_ids in self.runs:
            at = np.minimum(np.searchsorted(run_keys, keys), len(run_keys) - 1)
            hit = run_keys[at] == keys
            found[hit] = run_ids[at[hit]]
        new = found < 0
        if new.any():
            fresh = self.count + np.arange(int(new.sum()), dtype=np.int64)
            found[new] = fresh
            self.count += len(fresh)
            self.rows.append(rows[new])
            order = np.argsort(keys[new])
            self.runs.append((keys[new][order], fresh[order]))
            while len(self.runs) > 1 and len(self.runs[-1][0]) * 2 >= len(self.runs[-2][0]):
                (k2, i2), (k1, i1) = self.runs.pop(), self.runs.pop()
                k, i = np.concatenate([k1, k2]), np.concatenate([i1, i2])
                order = np.argsort(k, kind="stable")
                self.runs.append((k[order], i[order]))
        return found[inverse]

    def triples(self, which: np.ndarray, column: int) -> np.ndarray:
        """The triple ids of one key column (0: the item, 1-8: a level) of the descriptors `which`."""
        return self.all_rows()[which, column]

    def column(self, name: str, which: np.ndarray) -> np.ndarray:
        """DESC_COLUMNS ids of one column for the descriptors `which`."""
        if name in ("item_id", "item_name", "unit"):
            return self.item.table()[self.triples(which, 0), ("item_id", "item_name", "unit").index(name)]
        level, part = int(name[3]), name[5:]
        return self.levels[level].table()[self.triples(which, level), ("name", "code", "label").index(part)]


def scan(path: Path, sha256: Callable[[Path], str]) -> Facts:
    """The facts of one Parquet file (see the module docstring); never raises for an ineligible table."""
    with pq.ParquetFile(path, read_dictionary=SCAN_COLUMNS) as pf:
        rows = pf.metadata.num_rows
        try:
            return _scan(pf, rows, sha256(path))
        except Ineligible as exc:
            return Facts(sha256=sha256(path), rows=rows, failure=exc.condition)


def _scan(pf: pq.ParquetFile, rows: int, digest: str) -> Facts:
    _check_types(pf.schema_arrow)
    vocab = {name: _Vocab() for name in SCAN_COLUMNS}
    bit_of: dict[int, int] = {}  # prd_se id -> bit
    blank_period = np.ones(1, dtype=bool)  # by period id: _blank (id 0, null, is blank)
    grid = _Grid()
    descs = _Descriptors()
    kept_count = np.zeros(MAX_CODES, dtype=np.int64)  # rows after rule 1, per code bit
    codes_seen: set[int] = set()
    tbl_seen: set[int] = set()
    blank_rows = 0
    offset = 0
    for group in range(pf.num_row_groups):
        t = pf.read_row_group(group, columns=SCAN_COLUMNS)
        n = t.num_rows
        if not n:
            continue
        ids = {name: vocab[name].ids_of(t.column(name)) for name in SCAN_COLUMNS}
        tbl_seen.update(_encode(ids["tbl_id"])[0].tolist())
        codes = _encode(ids["prd_se"])[0].tolist()
        codes_seen.update(codes)
        for code in codes:
            if code not in bit_of:
                if len(bit_of) == MAX_CODES:
                    raise Ineligible("codes")
                bit_of[code] = len(bit_of)
        periods = vocab["period"].values
        if len(blank_period) < len(periods):
            blank_period = np.concatenate([blank_period, [_blank(p) for p in periods[len(blank_period):]]])
        keep = _rule1(pf, group, blank_period[ids["period"]])
        blank_rows += n - int(keep.sum())
        desc = descs.ids_of(ids)
        bit_lut = np.zeros(max(bit_of) + 1, dtype=np.int64)
        for code, bit in bit_of.items():
            bit_lut[code] = bit
        d, p, b = desc[keep], ids["period"][keep], bit_lut[ids["prd_se"][keep]]
        kept_count += np.bincount(b, minlength=MAX_CODES)
        grid.fit(len(periods), descs.count, len(bit_of))
        rownum = offset + np.nonzero(keep)[0]
        new = grid.first[d, b] == _UNSET
        if new.any():
            pair, inverse = _encode(d[new] * MAX_CODES + b[new])
            grid.first[pair // MAX_CODES, pair % MAX_CODES] = rownum[new][_firsts(inverse, len(pair))]
        dcap = grid.mask.shape[1]
        cells, inverse = _encode((p * dcap + d) * MAX_CODES + b)
        counts = np.bincount(inverse, minlength=len(cells))
        flat, bits = cells // MAX_CODES, (1 << (cells % MAX_CODES)).astype(np.uint8)
        mask, dup = grid.mask.reshape(-1), grid.dup.reshape(-1)
        twice = (counts > 1) | ((mask[flat] & bits) != 0)
        np.bitwise_or.at(dup, flat[twice], bits[twice])
        np.bitwise_or.at(mask, flat, bits)
        offset += n
        del mask, dup  # views of the grids: _settle frees the grids
    return _settle(vocab, bit_of, codes_seen, tbl_seen, kept_count, blank_rows, grid, descs, rows, digest,
                   lambda keep_bits, groups, keys, present: _merge_pass(pf, vocab, descs, bit_of, keep_bits, groups,
                                                                        keys, present))


def _rule1(pf: pq.ParquetFile, group: int, blank: np.ndarray) -> np.ndarray:
    """The rows of a row group tidy rule 1 keeps (blank: whether each row's period is blank): a row with a blank
    period, no value and a blank raw value is dropped."""
    keep = np.ones(len(blank), dtype=bool)
    if blank.any():
        where = np.nonzero(blank)[0]
        values = pf.read_row_group(group, columns=["value", "value_raw"]).take(pa.array(where))
        nulls = values.column("value").is_null().to_pylist()
        raws = values.column("value_raw").to_pylist()
        keep[where[[null and _blank(raw) for null, raw in zip(nulls, raws)]]] = False
    return keep


def _group_ids(parts: list[np.ndarray]) -> np.ndarray:
    """The group of each descriptor: descriptors equal in every part (integer arrays, -1 allowed) share one."""
    if not len(parts[0]):
        return np.zeros(0, dtype=np.int64)
    return _distinct(parts)[1]


def _sorted_groups(groups: np.ndarray, only_shared: bool) -> tuple[np.ndarray, np.ndarray]:
    """(column order that puts each group's members together, where each group starts in it)."""
    counts = np.bincount(groups)
    columns = np.nonzero(counts[groups] > 1)[0] if only_shared else np.arange(len(groups))
    columns = columns[np.argsort(groups[columns], kind="stable")]
    starts = np.nonzero(np.diff(groups[columns], prepend=-1))[0]
    return columns, starts


def _groups_overlap(present: np.ndarray, groups: np.ndarray) -> bool:
    """Whether two descriptors of one group are both present in one period (present: periods x descriptors)."""
    columns, starts = _sorted_groups(groups, only_shared=True)
    for at in range(0, len(starts), 4096):  # a slice of the groups at a time: bounded memory
        end = starts[at + 4096] if at + 4096 < len(starts) else len(columns)
        part = columns[starts[at]:end]
        if (np.add.reduceat(present[:, part].astype(np.int32), starts[at:at + 4096] - starts[at], axis=1) > 1).any():
            return True
    return False


def _merged(mask: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """mask with the columns of one group OR-ed together (one column per group)."""
    if not len(groups) or int(groups.max()) + 1 == len(groups):
        return mask
    columns, starts = _sorted_groups(groups, only_shared=False)
    return np.bitwise_or.reduceat(mask[:, columns], starts, axis=1)


def _lut(values: list, test: Callable) -> np.ndarray:
    """test() of every value of a column vocabulary, by id."""
    return np.array([bool(test(v)) for v in values], dtype=bool)


def _settle(vocab, bit_of, codes_seen, tbl_seen, kept_count, blank_rows, grid, descs, rows, digest, merge) -> Facts:
    """tidy_rows rules 1-3 and the eligibility conditions over the gathered grid, column-wise over the descriptors:
    each key part is a lookup table over the item and level triples, read through each descriptor's triple ids.
    Rule 3 runs merge(keep_bits, tidy groups, tidy keys of two kept rows or more, present) only when there is such a
    key: it updates `present` and gives the rows to drop and to blank (_Merged)."""
    pa.default_memory_pool().release_unused()
    value_of = {name: vocab[name].values for name in SCAN_COLUMNS}
    prd = value_of["prd_se"]
    descs_n = descs.count
    every = np.arange(descs_n)
    mask = grid.mask[:len(value_of["period"]), :descs_n]
    dup = grid.dup[:len(value_of["period"]), :descs_n]
    items = descs.item.table()  # item triple -> (item_id, item_name, unit) vocabulary ids
    level_rows = {level: descs.levels[level].table() for level in DIM_LEVELS}  # level triple -> (name, code, label)
    named_t = {level: _lut(value_of[f"dim{level}_name"], bool)[t[:, 0]] for level, t in level_rows.items()}
    small = {level: t.astype(np.int32) for level, t in level_rows.items()}  # lookups give int32 arrays
    item_t = descs.triples(every, 0)
    level_t = {level: descs.triples(every, level) for level in DIM_LEVELS}
    named = {level: named_t[level][level_t[level]] for level in DIM_LEVELS}
    # tidy_rows' key: the raw item id and, for every named level, its raw code (null too)
    tidy_parts = [items[:, 0].astype(np.int32)[item_t]] + [np.where(named_t[level], t[:, 1], -1)[level_t[level]]
                                                           for level, t in small.items()]
    tidy_groups = _group_ids(tidy_parts)
    del tidy_parts
    # a few periods at a time (bounded temporaries: the scan's peak is the two grids and `present`, 3 bytes a cell)
    step = max(1, (1 << 24) // max(1, descs_n))
    slices = [slice(start, start + step) for start in range(0, mask.shape[0], step)]
    # tidy rule 2, as tidy_rows: the real codes left after rule 1; every code but the first loses where two share a key
    real = sorted(str(prd[code]) for code, bit in bit_of.items() if kept_count[bit] and prd[code])
    losing_bits = 0
    if len(real) > 1:
        multi = 0
        for part in slices:
            merged = _merged(mask[part], tidy_groups)
            shared = merged[(merged & (merged - 1)) != 0]
            multi |= int(np.bitwise_or.reduce(shared)) if shared.size else 0
        first = next(bit for code, bit in bit_of.items() if prd[code] == real[0])
        losing_bits = multi & ~(1 << first)
    keep_bits = np.uint8(0xFF & ~losing_bits)
    present = np.empty(mask.shape, dtype=bool)
    # tidy rule 3: the tidy keys (period id * group_count + tidy group) of two kept rows or more
    group_count = int(tidy_groups.max()) + 1 if descs_n else 0
    grouped = group_count < descs_n  # some descriptors share a tidy group (else a descriptor's group is its id)
    if grouped:  # the groups are 0 .. group_count - 1, each with a member: the j-th start is group j's
        columns, starts = _sorted_groups(tidy_groups, only_shared=False)
    keys = []
    shared_keys = 0
    for part in slices:
        kept = mask[part] & keep_bits
        present[part] = kept != 0
        # each cell's kept rows, counted up to two: one a code bit, one more when a code bit was seen twice
        count = _POPCOUNT[kept]
        count += (dup[part] & keep_bits) != 0
        del kept
        if grouped:
            count = np.add.reduceat(count[:, columns], starts, axis=1, dtype=np.int32)
        at, group = np.nonzero(count > 1)
        del count
        shared_keys += len(at)
        if 2 * shared_keys > MAX_MERGE_ROWS:  # each key has two rows or more: refused before the second pass
            raise Ineligible("merge_rows")
        if len(at):
            keys.append((at + part.start) * group_count + group)
    del mask, dup
    grid.mask = grid.dup = None  # the bits are in `present` now: free them before the key checks
    settled = merge(keep_bits, tidy_groups, np.sort(np.concatenate(keys)), present) if keys else None
    del tidy_groups, keys
    alive = np.nonzero(present.any(axis=0))[0]
    periods = [value_of["period"][i] for i in np.nonzero(present.any(axis=1))[0].tolist()]
    if any(p is None or not p.strip() or p != p.strip() for p in periods):
        raise Ineligible("period")
    periods.sort()
    item_t, level_t = item_t[alive], {level: t[alive] for level, t in level_t.items()}
    named = {level: n[alive] for level, n in named.items()}
    if len(alive) < present.shape[1]:
        present = present[:, alive]
    item_ids = items[:, 0][item_t]

    def has_pipe(name: str) -> np.ndarray:
        return _lut(value_of[name], lambda v: v is not None and "|" in v)

    if any("|" in p for p in periods) or has_pipe("item_id")[item_ids].any() or any(
            (named_t[level] & has_pipe(f"dim{level}_code")[t[:, 1]])[level_t[level]].any()
            for level, t in level_rows.items()):
        raise Ineligible("pipe")
    nulls = int((item_ids == 0).sum())
    if 0 < nulls < len(alive):
        raise Ineligible("item_null")
    levels: dict[str, int] = {}
    for level, t in level_rows.items():
        used = np.unique(level_t[level][named[level]])
        for name_id in np.unique(t[used, 0]).tolist():
            if levels.setdefault(str(value_of[f"dim{level}_name"][name_id]), level) != level:
                raise Ineligible("level")
    # CellKey: item code (null and '' alike), and (name, code) of every named level with a code; a name has one level
    item_code = np.array([i if v else 0 for i, v in enumerate(value_of["item_id"])], dtype=np.int32)
    store_parts = [item_code[item_ids]]
    for level, t in level_rows.items():
        pair = t[:, 0] * (int(t[:, 1].max()) + 1) + t[:, 1]
        pair = np.where(named_t[level] & (t[:, 1] != 0), pair, -1)
        store_parts.append((pair.astype(np.int32) if pair.max() < 1 << 31 else pair)[level_t[level]])
    if _groups_overlap(present, _group_ids(store_parts)):
        raise Ineligible("cell_key")
    del store_parts
    # source_ref: str(item id), then the codes of the named levels in level order ('' for null), joined by '|'.
    # Codes are compared as strings across levels; no level is written like one empty code ('t|i||p').
    shared: dict[str | None, int] = {None: 0, "": 0}
    ref_item = np.array([shared.setdefault(str(v), len(shared)) for v in value_of["item_id"]], dtype=np.int32)
    ref_parts = [ref_item[item_ids]]
    position = np.zeros(len(alive), dtype=np.int32)
    sequence = np.full((len(alive), len(DIM_LEVELS)), -1, dtype=np.int32)
    for level, t in level_rows.items():
        code = np.array([shared.setdefault(v, len(shared)) for v in value_of[f"dim{level}_code"]], dtype=np.int32)
        on = named[level]
        sequence[np.nonzero(on)[0], position[on]] = code[t[:, 1]][level_t[level][on]]
        position += on
    sequence[position == 0, 0] = 0
    ref_parts += [sequence[:, n] for n in range(len(DIM_LEVELS))]
    if _groups_overlap(present, _group_ids(ref_parts)):
        raise Ineligible("ref")
    del ref_parts, sequence, present
    # validate's cell checks: the unit needs none (infer_unit always gives a non-empty unit)
    if (~_lut(value_of["item_name"], lambda v: str(v or "").strip()))[items[:, 1][item_t]].any():
        raise Ineligible("item_name")
    layout_of = _distinct([named[level].astype(np.int64) for level in DIM_LEVELS])[2].tolist()
    layouts = sorted({tuple(level for level in DIM_LEVELS if named[level][d]) for d in layout_of})
    # the kept descriptors in the order of their first kept row
    if settled is None:
        keep_mask = np.array([bool(keep_bits >> bit & 1) for bit in range(grid.first.shape[1])])
        firsts = np.where(keep_mask, grid.first[alive], _UNSET).min(axis=1)
    else:
        firsts = settled.firsts[alive]
    order = alive[np.argsort(firsts, kind="stable")]
    losing = [prd[code] for code, bit in bit_of.items() if losing_bits >> bit & 1]
    facts = Facts(
        sha256=digest, rows=rows,
        codes_all=sorted({str(prd[c]) for c in codes_seen if prd[c]}),
        tbl_ids=sorted({str(value_of["tbl_id"][t]) for t in tbl_seen}),
        blank_rows=blank_rows,
        cycle_rows=int(sum(kept_count[bit_of[c]] for c in bit_of if prd[c] in losing)),
        losing=losing, periods=periods, levels=levels, layouts=layouts,
        item_null=bool(len(alive)) and nulls == len(alive),
        descriptor_count=len(order), descriptors=_descriptor_table(descs, order, vocab),
    )
    if settled is not None:
        facts.same_rows, facts.conflict_cells = settled.same_rows, settled.conflict_cells
        facts.drop, facts.conflict = settled.drop, settled.conflict
        facts.merge_rows = len(settled.drop) + len(settled.conflict)
    return facts


@dataclass
class _Merged:
    """Tidy rule 3 settled (_merge_pass)."""

    drop: np.ndarray  # sorted row numbers of the rows it drops (a key's later rows)
    conflict: np.ndarray  # sorted row numbers of the first rows of conflict keys
    same_rows: int
    conflict_cells: int
    firsts: np.ndarray  # per descriptor id: its first row kept by rules 1 to 3 (_UNSET: none)


_POPCOUNT = np.array([n.bit_count() for n in range(256)], dtype=np.uint8)


class _Raws:
    """The raw values of the first rows of rule-3 keys, in met order (positions 0, 1, ...): large_string chunks that
    are never concatenated whole, and compared chunk by chunk (a copy of them all for each row group's lookup doubled
    the second pass's peak with long raw values). A new chunk is merged into the one before while it is at least half
    its size and both stay within CHUNK bytes (as _Descriptors merges its runs): a lookup meets a few dozen chunks."""

    CHUNK = 64 << 20

    def __init__(self) -> None:
        self.chunks: list[pa.Array] = []
        self.starts: list[int] = []  # each chunk's first position
        self.count = 0  # values held
        self.nbytes = 0  # their Arrow bytes (data, offsets, validity)

    def add(self, values: pa.Array) -> None:
        self.chunks.append(values)
        self.starts.append(self.count)
        self.count += len(values)
        self.nbytes += values.nbytes
        while (len(self.chunks) > 1 and 2 * self.chunks[-1].nbytes >= self.chunks[-2].nbytes
               and self.chunks[-1].nbytes + self.chunks[-2].nbytes <= self.CHUNK):
            last = self.chunks.pop()
            self.starts.pop()
            self.chunks[-1] = pa.concat_arrays([self.chunks[-1], last])

    def equal(self, values: pa.Array, at: np.ndarray, positions: np.ndarray) -> np.ndarray:
        """Whether each values[at[i]] equals the value held at positions[i], as Python compares them (None equals
        None only). One chunk at a time: only the values compared with that chunk are copied."""
        same = np.zeros(len(at), dtype=bool)
        chunk = np.searchsorted(np.array(self.starts, dtype=np.int64), positions, side="right") - 1
        order = np.argsort(chunk, kind="stable")
        bounds = np.searchsorted(chunk[order], np.arange(len(self.chunks) + 1))
        for c, (low, high) in enumerate(zip(bounds[:-1].tolist(), bounds[1:].tolist())):
            if high == low:
                continue
            which = order[low:high]
            mine = pc.cast(values.take(pa.array(at[which])), pa.large_string())
            theirs = self.chunks[c].take(pa.array(positions[which] - self.starts[c]))
            same[which] = pc.or_(pc.fill_null(pc.equal(mine, theirs), False),
                                 pc.and_(mine.is_null(), theirs.is_null())).to_numpy(zero_copy_only=False)
        return same


def _merge_pass(pf: pq.ParquetFile, vocab: dict[str, _Vocab], descs: _Descriptors, bit_of: dict[int, int],
                keep_bits: np.uint8, groups: np.ndarray, keys: np.ndarray, present: np.ndarray) -> _Merged:
    """Tidy rule 3 as tidy_rows settles it, over the rows of `keys` (sorted tidy keys of two kept rows or more:
    period id * group count + tidy group of the descriptor): a second pass over the file, the key columns of every
    row group again (ids from the first pass's vocabularies) and the value columns of those keys' rows only.

    In file order (among the rows rules 1 and 2 keep), a key's first row is kept and its later rows are dropped; the
    key is a conflict when a later row's (value_raw, value, value_kind) is unequal to the first row's, in Python's
    terms: None equals None, 0.0 equals -0.0, NaN equals nothing. A later row whose descriptor is not its key's first
    row's leaves `present` (its (period, descriptor) cell holds no kept row: the cell's rows are all of that key), and
    each descriptor's first kept row is found again (its old first row may be gone).

    Memory: per key of `keys`, the first row's number, descriptor, value, null flag, kind id, raw value position, a
    conflict flag and its count of later rows (about 50 bytes); the first rows' raw values (_Raws); per later row, its
    row number (8 bytes: `drop`). Ineligible("merge_rows") as soon as more than MAX_MERGE_ROWS rows of `keys` are met,
    before their values are read, or as soon as the raw values held pass MAX_MERGE_BYTES."""
    count = len(keys)
    group_count = int(groups.max()) + 1
    blank_period = _lut(vocab["period"].values, _blank)
    bit_lut = np.zeros(max(bit_of) + 1, dtype=np.int64)
    for code, bit in bit_of.items():
        bit_lut[code] = bit
    kept_bit = np.array([bool(keep_bits >> bit & 1) for bit in range(MAX_CODES)])
    firsts = np.full(descs.count, _UNSET, dtype=np.int64)
    first_row = np.full(count, -1, dtype=np.int64)
    first_desc = np.zeros(count, dtype=np.int64)
    first_value = np.zeros(count, dtype=np.float64)
    first_null = np.zeros(count, dtype=bool)
    first_kind = np.zeros(count, dtype=np.int32)
    first_raw = np.zeros(count, dtype=np.int64)  # place of the first row's raw value in `raws`
    conflict = np.zeros(count, dtype=bool)
    later_rows = np.zeros(count, dtype=np.int64)
    raws = _Raws()
    kinds = _Vocab()
    drops: list[np.ndarray] = []
    met = offset = 0
    for group in range(pf.num_row_groups):
        t = pf.read_row_group(group, columns=SCAN_COLUMNS)
        n = t.num_rows
        if not n:
            continue
        ids = {name: vocab[name].ids_of(t.column(name)) for name in SCAN_COLUMNS}
        del t
        keep = _rule1(pf, group, blank_period[ids["period"]]) & kept_bit[bit_lut[ids["prd_se"]]]
        desc = descs.ids_of(ids)
        period = ids["period"]
        del ids
        key = period * group_count + groups[desc]
        at = np.minimum(np.searchsorted(keys, key), count - 1)
        shared = keep & (keys[at] == key)
        where = np.nonzero(shared)[0]
        kept = keep & ~shared  # rule 3 keeps these, and the first rows of keys below
        if len(where):
            met += len(where)
            if met > MAX_MERGE_ROWS:
                raise Ineligible("merge_rows")
        # the value columns of those rows, a batch of the row group at a time: with long raw values a whole row
        # group's column (and its copies) cost more than the raw values held
        start = 0
        batches = pf.iter_batches(batch_size=MERGE_BATCH_ROWS, row_groups=[group], columns=VALUE_COLUMNS,
                                  use_threads=False) if len(where) else ()
        for batch in batches:
            low, high = np.searchsorted(where, [start, start + batch.num_rows])
            part = where[low:high]  # in row group order
            if not len(part):
                start += batch.num_rows
                continue
            pos = at[part]
            values = pa.Table.from_batches([batch]).take(pa.array(part - start))
            start += batch.num_rows
            del batch
            column = values.column("value")
            null = column.is_null().to_numpy()
            value = pc.fill_null(column, 0).to_numpy().astype(np.float64)
            kind = kinds.ids_of(values.column("value_kind")).astype(np.int32)
            raw = values.column("value_raw").combine_chunks()
            del values, column
            seen, head = np.unique(pos, return_index=True)  # each key's first row in this batch
            new = first_row[seen] < 0  # ... and the key's first row of all
            seen, head = seen[new], head[new]
            first_row[seen] = offset + part[head]
            first_desc[seen] = desc[part[head]]
            first_value[seen], first_null[seen], first_kind[seen] = value[head], null[head], kind[head]
            first_raw[seen] = raws.count + np.arange(len(head))
            raws.add(pc.cast(raw.take(pa.array(head)), pa.large_string()))
            if raws.nbytes > MAX_MERGE_BYTES:
                raise Ineligible("merge_rows")
            kept[part[head]] = True
            later = np.ones(len(part), dtype=bool)
            later[head] = False
            later = np.nonzero(later)[0]
            if len(later):
                to = pos[later]
                same = np.where(null[later] | first_null[to], null[later] & first_null[to],
                                value[later] == first_value[to])
                same &= kind[later] == first_kind[to]
                same &= raws.equal(raw, later, first_raw[to])
                conflict[to[~same]] = True
                np.add.at(later_rows, to, 1)
                rows = part[later]
                drops.append(offset + rows)
                gone = desc[rows] != first_desc[to]
                present[period[rows[gone]], desc[rows[gone]]] = False
        alone = np.nonzero(kept)[0]
        if len(alone):
            seen, head = np.unique(desc[alone], return_index=True)
            firsts[seen] = np.minimum(firsts[seen], offset + alone[head])
        offset += n
    agree = ~conflict & (later_rows > 0)
    return _Merged(drop=np.concatenate(drops) if drops else np.zeros(0, dtype=np.int64),
                   conflict=np.sort(first_row[conflict]), same_rows=int(later_rows[agree].sum()),
                   conflict_cells=int(conflict.sum()), firsts=firsts)


def _descriptor_table(descs: _Descriptors, order: np.ndarray, vocab: dict[str, _Vocab]) -> pa.Table:
    arrays = []
    for name in DESC_COLUMNS:
        ids = descs.column(name, order)
        dictionary = pa.array(vocab[name].values[1:], type=pa.string())
        indices = pa.array(ids - 1, type=pa.int32(), mask=ids == 0)
        arrays.append(pa.DictionaryArray.from_arrays(indices, dictionary))
    return pa.table(arrays, names=DESC_COLUMNS)


# ---------------------------------------------------------------- the saved scan


def _index_files(folder: Path, key: str) -> tuple[Path, Path, Path]:
    """(facts, descriptors, rule-3 rows) of one table's saved scan; the last only for a table with rule-3 rows."""
    return folder / f"{key}.json", folder / f"{key}.descriptors.parquet", folder / f"{key}.merge.parquet"


_SHA_KEY = b"kosis_source_sha256"  # the descriptors file's schema metadata: the sha256 of the file it was scanned from
_warned: set[str] = set()  # index folders that could not be written, warned about once per process


def load_facts(path: Path, folder: Path | None, sha256: Callable[[Path], str]) -> Facts:
    """The saved facts of this file when they were made from the same bytes (its sha256, computed now) by the same
    code (SCAN_CODE), else a new scan, saved when the folder is writable. The facts' sha256 is always the current
    file's. Once saved, the descriptor table is not held: ParquetStore reads it back from descriptors_file. The
    rule-3 rows (drop, conflict) are always held: read back from the merge file with the facts.

    A saved scan that cannot be read is scanned again; a folder that cannot be written is logged once and skipped.
    With no code key (SCAN_CODE None: a source could not be read) nothing is read from the folder or saved there:
    every load scans, and the descriptor table stays held."""
    digest = sha256(path)
    head = {"code": SCAN_CODE}
    if SCAN_CODE is None:
        folder = None
    if folder is not None:
        found = _read_saved(folder, path.stem, head, digest)
        if found is not None:
            return found
    facts = scan(path, lambda _: digest)
    if folder is not None and _write_saved(folder, path.stem, head, facts):
        facts.descriptors, facts.descriptors_file = None, _index_files(folder, path.stem)[1]
    return facts


def _read_saved(folder: Path, key: str, head: dict, digest: str) -> Facts | None:
    meta_path, desc_path, merge_path = _index_files(folder, key)
    try:
        saved = json.loads(meta_path.read_text(encoding="utf-8"))
        if saved.get("head") != head or saved["facts"]["sha256"] != digest:
            return None
        facts = Facts(**saved["facts"])
        if facts.failure is None:
            if not _descriptors_match(desc_path, digest, facts.descriptor_count):
                return None
            facts.descriptors_file = desc_path
            if facts.merge_rows:
                facts.drop, facts.conflict = _read_merge(merge_path, digest, facts)
        facts.layouts = [list(layout) for layout in facts.layouts]
        return facts
    except Exception:  # missing, half-written by another tool, or from another format: scan again
        return None


def _descriptors_match(desc_path: Path, digest: str, count: int) -> bool:
    """Whether the saved descriptors file was written from the file of this sha256 (its footer only is read)."""
    footer = pq.read_metadata(desc_path)
    schema = footer.schema.to_arrow_schema()
    return ((schema.metadata or {}).get(_SHA_KEY) == digest.encode("ascii") and schema.names == DESC_COLUMNS
            and footer.num_rows == count)


_MERGE_SCHEMA = pa.schema([("row", pa.int64()), ("conflict", pa.bool_())])


def _merge_table(facts: Facts) -> pa.Table:
    """The rule-3 rows as saved: every row number of drop and conflict, sorted, with whether it is a conflict."""
    rows = np.concatenate([facts.drop, facts.conflict])
    flags = np.concatenate([np.zeros(len(facts.drop), dtype=bool), np.ones(len(facts.conflict), dtype=bool)])
    order = np.argsort(rows, kind="stable")
    return pa.table([pa.array(rows[order]), pa.array(flags[order])], schema=_MERGE_SCHEMA).replace_schema_metadata(
        {_SHA_KEY: facts.sha256.encode("ascii")})


def _read_merge(merge_path: Path, digest: str, facts: Facts) -> tuple[np.ndarray, np.ndarray]:
    """(drop, conflict) from the saved merge file of the file of this sha256 and these facts; ValueError when it is
    another one or does not hold what the facts say: as many rows as merge_rows, no null, row numbers strictly
    increasing within [0, rows), as many conflict rows as conflict_cells, and at least same_rows + conflict_cells rows
    dropped (a conflict key drops one later row or more). The store reads these rows as they are (searchsorted on
    them, one flag per row): a file that passed only the sha256 could give wrong answers or an IndexError."""
    table = pq.read_table(merge_path)
    if ((table.schema.metadata or {}).get(_SHA_KEY) != digest.encode("ascii") or table.num_rows != facts.merge_rows
            or not table.schema.equals(_MERGE_SCHEMA) or any(column.null_count for column in table.columns)):
        raise ValueError(f"{merge_path.name} is not the saved rule-3 rows of this file")
    rows = table.column("row").to_numpy()
    flags = table.column("conflict").to_numpy()
    conflicts = int(flags.sum())
    if (rows[0] < 0 or rows[-1] >= facts.rows or (np.diff(rows) <= 0).any() or conflicts != facts.conflict_cells
            or len(rows) - conflicts < facts.same_rows + facts.conflict_cells):
        raise ValueError(f"{merge_path.name} does not hold the rule-3 rows its facts name")
    return rows[~flags], rows[flags]


def _write_saved(folder: Path, key: str, head: dict, facts: Facts) -> bool:
    """Save the scan; False (and one warning per folder per process) when the folder cannot be written. The facts
    go last: they name the other files' counts and sha256, so a stop half-way leaves a scan that is not used."""
    meta_path, desc_path, merge_path = _index_files(folder, key)
    try:
        folder.mkdir(parents=True, exist_ok=True)
        if facts.descriptors is not None:
            table = facts.descriptors.replace_schema_metadata({_SHA_KEY: facts.sha256.encode("ascii")})
            _atomic(desc_path, lambda f: pq.write_table(table, f, compression="zstd"))
        if facts.merge_rows:
            merge = _merge_table(facts)
            _atomic(merge_path, lambda f: pq.write_table(merge, f, compression="zstd"))
        else:
            merge_path.unlink(missing_ok=True)  # one of an earlier version of the file
        text = json.dumps({"head": head, "facts": facts.to_json()}, ensure_ascii=False)
        _atomic(meta_path, lambda f: f.write(text.encode("utf-8")))
        return True
    except OSError as exc:
        if str(folder) not in _warned:
            _warned.add(str(folder))
            log.warning("cannot save the scan of a big table in %s (%s): each load scans the file again", folder, exc)
        return False


# Waits (seconds) before each new try of the rename in _atomic: on Windows a virus scanner or OneDrive may hold the
# target (or the new file) for a moment, and the rename then fails with PermissionError. About a second in all.
REPLACE_RETRIES = (0.05, 0.1, 0.2, 0.25, 0.4)


def _atomic(target: Path, write: Callable[[object], object], *, keep: Callable[[Path], bool] | None = None) -> bool:
    """Write through a temporary file in the same folder, then rename it over the target (tried again a few times on
    PermissionError, REPLACE_RETRIES). keep(temporary path), when given, decides after the write whether to put the
    file in place at all; False when it did not (the temporary file is removed)."""
    handle, temp = tempfile.mkstemp(dir=target.parent, prefix=f".{target.name}.", suffix=".tmp")
    try:
        with os.fdopen(handle, "wb") as f:
            write(f)
        if keep is not None and not keep(Path(temp)):
            Path(temp).unlink(missing_ok=True)
            return False
        os.chmod(temp, 0o644)  # mkstemp makes it private; the saved scan is an ordinary data file
        for wait in (*REPLACE_RETRIES, None):
            try:
                os.replace(temp, target)
                return True
            except PermissionError:
                if wait is None:
                    raise
                time.sleep(wait)
    except BaseException:
        Path(temp).unlink(missing_ok=True)
        raise


# ---------------------------------------------------------------- the saved table index


def _adapter_sources() -> Iterator[tuple[str, Path]]:
    """(name, path) of every module in this folder (the scan, the descriptor cells, the loader that builds the meta),
    sorted. A generator: a folder that cannot be listed fails inside source_hash, which never raises."""
    yield from ((source.name, source) for source in sorted(Path(__file__).resolve().parent.glob("*.py")))


def _adapters_code() -> str | None:
    """The sha256 of the source text of the modules of this folder, line endings made alike: the store's part of the
    key of a saved table index; None when one cannot be read or there is none (prepare.store.source_hash)."""
    return source_hash(_adapter_sources())


ADAPTERS_CODE = _adapters_code()


# The largest saved table index file written or read (prepare.index.MAX_SAVED_ENTRIES bounds what is offered; this
# bounds the file itself, whatever its labels' length): json.load peaks at about 16 times the file (a 33 MB file
# raised the peak by 0.5 GB), and a bigger file costs more to load than the index to build.
MAX_INDEX_BYTES = 64 << 20


def _table_index_file(folder: Path, key: str) -> Path:
    return folder / f"{key}.table_index.json"


def _read_table_index(path: Path, head: dict, digest: str) -> object | None:
    """The saved payload when it was saved for this head (code keys) and file sha256, else None (missing, corrupt,
    half-written, another version or another file, or over MAX_INDEX_BYTES)."""
    try:
        if path.stat().st_size > MAX_INDEX_BYTES:
            return None
        with path.open(encoding="utf-8") as f:
            saved = json.load(f)
        if saved.get("head") != head or saved.get("sha256") != digest:
            return None
        return saved["index"]
    except Exception:
        return None


def _write_table_index(path: Path, head: dict, digest: str, payload: object) -> bool:
    """Save a payload atomically, streamed to the file (no whole text in memory); False when not: a file over
    MAX_INDEX_BYTES is not kept, and a folder that cannot be written is logged once per process (shared with the
    scan's warning)."""

    def dump(f) -> None:
        text = io.TextIOWrapper(f, encoding="utf-8", newline="")
        json.dump({"head": head, "sha256": digest, "index": payload}, text, ensure_ascii=False, separators=(",", ":"))
        text.flush()
        text.detach()  # the file is closed by _atomic

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        return _atomic(path, dump, keep=lambda temp: temp.stat().st_size <= MAX_INDEX_BYTES)
    except Exception as exc:  # OSError mostly; a payload that is not JSON too: never raise, the index was built
        folder = str(path.parent)
        if folder not in _warned:
            _warned.add(folder)
            log.warning("cannot save the index of a big table in %s (%s): each process builds it again", folder, exc)
        return False


# ---------------------------------------------------------------- the store


_NO_VALUE = {"period": "", "value_raw": None, "value": None, "value_kind": None}
_MISSING = object()
_SEP, _JOIN = "\x1e", "\x1f"  # between the parts of a key string; between a level's name and its code


def _plain(table: pa.Table) -> pa.Table:
    """The table with its dictionary columns cast to plain strings (to_pylist is about ten times faster on those)."""
    if not any(pa.types.is_dictionary(f.type) for f in table.schema):
        return table
    return table.cast(pa.schema([pa.field(f.name, pa.string()) if pa.types.is_dictionary(f.type) else f
                                 for f in table.schema]))


class _Memo:
    """A least-recently-used memo bounded by the total weight of what it holds (a hit makes an entry the newest)."""

    def __init__(self, limit: Callable[[], int], weight: Callable[[object], int] = lambda value: 1) -> None:
        self._items: OrderedDict = OrderedDict()
        self._limit, self._weight = limit, weight
        self.size = 0

    def __len__(self) -> int:
        return len(self._items)

    def __contains__(self, key) -> bool:
        return key in self._items

    def get(self, key, default=_MISSING):
        found = self._items.get(key, _MISSING)
        if found is _MISSING:
            return default
        self._items.move_to_end(key)
        return found

    def put(self, key, value) -> None:
        old = self._items.pop(key, _MISSING)
        if old is not _MISSING:
            self.size -= self._weight(old)
        self._items[key] = value
        self.size += self._weight(value)
        limit = self._limit()
        while self.size > limit and self._items:
            _, dropped = self._items.popitem(last=False)
            self.size -= self._weight(dropped)

    def clear(self) -> None:
        self._items.clear()
        self.size = 0


class ParquetStore:
    """A CellStore (prepare.store.FileTable) over one eligible Parquet file; see the module docstring.

    reads / row_groups_read / rows_read / rows_masked / rows_converted count the requests that touched the file, the
    row groups whose key columns were read, the rows of those, the rows the column masks kept, and the rows turned
    into Python values (tests and measurements read them)."""

    def __init__(self, path: Path, meta: TableMeta, facts: Facts) -> None:
        self.path, self.meta, self.facts = path, meta, facts
        self.periods = list(facts.periods)
        self._periods = set(facts.periods)
        self._levels = dict(facts.levels)
        self._named = sorted({level for layout in facts.layouts for level in layout})
        self._names_at = {level: {name for name, lvl in self._levels.items() if lvl == level} for level in self._named}
        self._layouts = [tuple(layout) for layout in facts.layouts]
        self._code_counts = {max(1, len(layout)) for layout in self._layouts}  # parts between item and period in a ref
        self._losing = list(facts.losing)
        self._metadata = pq.read_metadata(path)  # kept: each request opens the file with it and closes it at the end
        self._groups = self._metadata.num_row_groups
        self._sizes = [self._metadata.row_group(g).num_rows for g in range(self._groups)]
        # each row group's first row number, int64 as the row numbers of facts.drop and facts.conflict (cumsum's
        # default integer is 32 bits on Windows with NumPy 1.x)
        self._offsets = np.cumsum([0] + self._sizes, dtype=np.int64)
        self._stats = [self._stats_of(self._metadata.row_group(g)) for g in range(self._groups)]
        self._lost = {g for g in range(self._groups) if self._all_losing(self._stats[g])}  # tidy rule 2 drops every row
        self._cell_columns = ["period", "item_id", "item_name", "unit", "value_raw", "value", "value_kind"] + [
            f"dim{level}_{part}" for level in self._named for part in ("name", "code", "label")]
        self._unnamed = {f"dim{level}_{part}": None for level in DIM_LEVELS if level not in self._named
                         for part in ("name", "code", "label")}
        self._cells = _Memo(lambda: MEMO_CELLS)
        self._refs = _Memo(lambda: MEMO_CELLS)
        self._series = _Memo(lambda: MEMO_SERIES_CELLS, lambda cells: max(1, len(cells)))
        self._summaries = _Memo(lambda: MEMO_SUMMARIES)
        self.reads = self.row_groups_read = self.rows_read = self.rows_masked = self.rows_converted = 0

    @staticmethod
    def _stats_of(group) -> dict[str, tuple]:
        """column -> (min, max, null_count) where the footer has statistics. A min or max the writer truncated is
        still a bound of the values (a prefix sorts no later than the value, a raised one no earlier): pruning holds."""
        out = {}
        for n in range(group.num_columns):
            chunk = group.column(n)
            stats = chunk.statistics
            if stats is None or not stats.has_min_max:
                continue
            nulls = stats.null_count if stats.has_null_count else None
            out[chunk.path_in_schema] = (stats.min, stats.max, nulls)
        return out

    def _all_losing(self, stats: dict[str, tuple]) -> bool:
        found = stats.get("prd_se")
        if not self._losing or found is None:
            return False
        low, high, nulls = found
        return nulls == 0 and low == high and low in self._losing

    def _rule3(self, rows: np.ndarray | None, group: int) -> np.ndarray | None:
        """A row group's flags of the rows of `rows` (facts.drop or facts.conflict), None when it has none."""
        if rows is None:
            return None
        start, size = int(self._offsets[group]), self._sizes[group]
        low, high = np.searchsorted(rows, [start, start + size])
        if low == high:
            return None
        flags = np.zeros(size, dtype=bool)
        flags[rows[low:high] - start] = True
        return flags

    @staticmethod
    def _blanked(table: pa.Table, flags: np.ndarray | None) -> pa.Table:
        """The table with the value columns it has made blank on the flagged rows (the first rows of conflict keys,
        as tidy_rows writes them: value_raw "", value None, value_kind "blank")."""
        if flags is None or not flags.any():
            return table
        on = pa.array(flags)
        for name, fill in (("value_raw", ""), ("value", None), ("value_kind", "blank")):
            at = table.schema.get_field_index(name)
            if at < 0:
                continue
            column = table.column(at)
            if pa.types.is_dictionary(column.type):
                column = pc.cast(column, column.type.value_type)
            table = table.set_column(at, name, pc.if_else(on, pa.scalar(fill, column.type), column))
        return table

    def _open(self) -> pq.ParquetFile:
        """A handle on the file for one read (closed by its caller): the footer is not read again."""
        return pq.ParquetFile(self.path, metadata=self._metadata, read_dictionary=STRING_COLUMNS)

    # ------------------------------------------------------------ index facts

    def index_cells(self) -> Iterator[Cell]:
        for batch in self._descriptor_batches():
            rows = _plain(pa.Table.from_batches([batch])).to_pylist()
            yield from cells_of([row | _NO_VALUE for row in rows], self.meta)

    def _descriptor_batches(self) -> Iterator[pa.RecordBatch]:
        """The descriptor table: held when it was not saved, else read back from the saved file (a file that went
        away or was replaced since the load: the data file is scanned again, and must be the one loaded)."""
        facts = self.facts
        if facts.descriptors is not None:
            yield from facts.descriptors.to_batches(max_chunksize=20_000)
            return
        try:
            saved = facts.descriptors_file is not None and _descriptors_match(facts.descriptors_file, facts.sha256,
                                                                                facts.descriptor_count)
        except (pa.ArrowException, OSError):
            saved = False
        if saved:
            with pq.ParquetFile(facts.descriptors_file, read_dictionary=DESC_COLUMNS) as pf:
                yield from pf.iter_batches(batch_size=20_000)
            return
        from kosis_chat.adapters.parquet import sha256_of

        again = scan(self.path, sha256_of)
        if again.failure is not None or again.sha256 != facts.sha256:
            raise RuntimeError(f"{self.path.name} changed after it was loaded: load the table again")
        yield from again.descriptors.to_batches(max_chunksize=20_000)

    # The saved table index (prepare.store.FileTable): one more file beside the saved scan, kept only where the scan
    # was saved (no index folder, or one that could not be written: nothing). index_state tells a measurement what
    # happened: "loaded" (a payload was handed out and not built again after it), "saved", "not saved", or None.
    index_state: str | None = None

    def _index_head(self, code: str | None) -> tuple[Path, dict] | None:
        """(the saved index file, its head); None when nothing may be saved: no saved scan (SCAN_CODE None
        included: load_facts then saves none), or a code key that is None (prepare.store.source_hash could not read
        a source)."""
        saved = self.facts.descriptors_file
        if saved is None or code is None or ADAPTERS_CODE is None:
            return None
        return _table_index_file(saved.parent, self.path.stem), {"scan": SCAN_CODE, "adapters": ADAPTERS_CODE,
                                                                 "index": code}

    def load_index(self, code: str) -> object | None:
        found = self._index_head(code)
        payload = None if found is None else _read_table_index(found[0], found[1], self.facts.sha256)
        if payload is not None:
            self.index_state = "loaded"
        return payload

    def save_index(self, code: str, payload: object) -> None:
        found = self._index_head(code)
        if found is None:
            return
        written = _write_table_index(found[0], found[1], self.facts.sha256, payload)
        self.index_state = "saved" if written else "not saved"

    # ------------------------------------------------------------ reading

    def _may_hold(self, group: int, want: Mapping[str, list]) -> bool:
        stats = self._stats[group]
        for name, values in want.items():
            found = stats.get(name)
            if found is None:
                continue
            low, high, nulls = found
            if values and values[0] is None:
                if nulls is None or nulls > 0:
                    continue
                values = values[1:]
            at = bisect.bisect_left(values, low)
            if at == len(values) or values[at] > high:
                return False
        return True

    def _read(self, want: Mapping[str, Collection], *, empty: Collection[int] = (), mask_of: Callable | None = None,
              exact: Callable | None = None, columns: Sequence[str] = (),
              rows: bool = True) -> Iterator[tuple[int, pa.Table, pa.Array, pa.Table | None]]:
        """(row group, key columns, mask, rows) of every row group with kept rows whose columns hold one of the
        wanted values (None: null) and whose levels `empty` are unnamed or have no code; mask_of adds an exact mask
        over the key columns. rows (when asked): the cell columns of the masked rows as plain strings, those that
        exact (a mask over them, built on the Arrow side) keeps. One request; each row group's read opens the file
        and closes it."""
        want = {name: sorted(values, key=lambda v: (v is not None, v or "")) for name, values in want.items()}
        need = set(want) | set(columns) | {f"dim{level}_{part}" for level in empty for part in ("name", "code")}
        if self._losing:
            need.add("prd_se")
        if self.facts.blank_rows and "period" not in want:
            need.add("period")
        groups = [group for group in range(self._groups) if group not in self._lost and self._may_hold(group, want)]
        if not groups:
            return
        self.reads += 1
        losing = pa.array(self._losing, type=pa.string())
        sets = {name: pa.array(values, type=pa.string()) for name, values in want.items()}
        kept_periods = pa.array(self.periods, type=pa.string())
        size = sum(self._sizes[group] for group in groups)
        parallel = size >= PARALLEL_ROWS
        self.row_groups_read += len(groups)
        self.rows_read += size

        def masked(group: int) -> tuple[int, pa.Table, pa.Array, pa.Table | None] | None:
            with self._open() as pf:
                keys = pf.read_row_group(group, columns=sorted(need), use_threads=not parallel)
                mask = pa.array(np.ones(keys.num_rows, dtype=bool))
                for name, values in sets.items():
                    mask = pc.and_(mask, pc.is_in(keys.column(name), value_set=values, skip_nulls=False))
                for level in empty:  # no key of the request names this level: a matching row has no code on it
                    name, code = keys.column(f"dim{level}_name"), keys.column(f"dim{level}_code")
                    mask = pc.and_(mask, pc.or_kleene(pc.or_kleene(pc.is_null(name), pc.equal(name, "")),
                                                      pc.is_null(code)))
                if self._losing:  # tidy rule 2
                    mask = pc.and_(mask, pc.invert(pc.is_in(keys.column("prd_se"), value_set=losing, skip_nulls=False)))
                if self.facts.blank_rows and "period" not in want:  # tidy rule 1: only those rows have a blank period
                    mask = pc.and_(mask, pc.is_in(keys.column("period"), value_set=kept_periods))
                drop = self._rule3(self.facts.drop, group)
                if drop is not None:  # tidy rule 3: a key's later rows
                    mask = pc.and_(mask, pa.array(~drop))
                conflict = self._rule3(self.facts.conflict, group)
                keys = self._blanked(keys, conflict)
                if mask_of is not None:
                    mask = pc.and_(mask, mask_of(keys))
                mask = pc.fill_null(mask, False)
                if not pc.any(mask).as_py():
                    return None
                if not rows:
                    return group, keys, mask, None
                found = _plain(pf.read_row_group(group, columns=self._cell_columns, use_threads=not parallel).filter(mask))
            if conflict is not None:
                found = self._blanked(found, conflict[mask.to_numpy(zero_copy_only=False)])
            if exact is not None:
                found = found.filter(pc.fill_null(exact(found), False))
            return group, keys, mask, found

        def counted(results: Iterable) -> Iterator[tuple[int, pa.Table, pa.Array, pa.Table | None]]:
            for result in filter(None, results):
                self.rows_masked += pc.sum(result[2]).as_py() or 0
                if result[3] is None or result[3].num_rows:
                    yield result

        if not parallel:
            yield from counted(map(masked, groups))
            return
        # READERS row groups at a time, in windows, so that results come in file order and few are held
        with ThreadPoolExecutor(max_workers=READERS) as pool:
            for start in range(0, len(groups), 2 * READERS):
                yield from counted(pool.map(masked, groups[start:start + 2 * READERS]))

    def _dicts(self, found: pa.Table) -> list[dict]:
        """Rows as cells_of reads them (unnamed levels as None)."""
        rows = found.to_pylist()
        self.rows_converted += len(rows)
        return [row | self._unnamed for row in rows] if self._unnamed else rows

    def _rows_at(self, group: int, mask: pa.Array) -> list[dict]:
        """The rows of a row group that mask keeps, unblanked: only _summary's sample reads it, and the sample is
        never the first row of a conflict key. _summary picks a row whose value_kind is not "blank" in `keys`, and
        _read has blanked `keys` (value_kind "blank" on every row of facts.conflict) before handing it out; so the
        rows facts.conflict names are never picked, and no other row is blanked."""
        with self._open() as pf:
            return self._dicts(_plain(pf.read_row_group(group, columns=self._cell_columns).filter(mask)))

    def _cell(self, row: dict) -> Cell:
        cell = clean_cell(cells_of([row], self.meta)[0])
        self._cells.put(cell_key(cell), cell)
        self._refs.put(cell.source_ref, cell)
        return cell

    def _row_key(self, row: dict) -> CellKey:
        dims = tuple(sorted((str(row[f"dim{level}_name"]), str(row[f"dim{level}_code"])) for level in self._named
                            if row[f"dim{level}_name"] and row[f"dim{level}_code"] is not None))
        return str(row["item_id"] or ""), dims, str(row["period"]).strip()

    def _row_ref(self, row: dict) -> str:
        codes = "|".join("" if row[f"dim{level}_code"] is None else str(row[f"dim{level}_code"])
                         for level in DIM_LEVELS if row[f"dim{level}_name"])
        return f"{self.meta.tbl_id}|{row['item_id']}|{codes}|{row['period']}"

    # The whole key (or ref) of each row, built on the Arrow side over the rows the column masks left, so that only
    # the rows asked for are turned into Python values (the column masks alone keep the cross product of the codes
    # asked for). The exact check on each cell's key or ref stays.

    @staticmethod
    def _token(table: pa.Table, level: int) -> pa.ChunkedArray:
        """name + _JOIN + code on the rows that name the level and have a code on it, else ''."""
        name, code = table.column(f"dim{level}_name"), table.column(f"dim{level}_code")
        named = pc.fill_null(pc.not_equal(name, ""), False)
        return pc.fill_null(pc.if_else(named, pc.binary_join_element_wise(name, code, _JOIN), ""), "")

    def _key_strings(self, table: pa.Table, period: bool) -> pa.ChunkedArray:
        parts = [pc.fill_null(table.column("item_id"), "")] + [self._token(table, level) for level in self._named]
        if period:
            parts.append(table.column("period"))
        return pc.binary_join_element_wise(*parts, _SEP)

    def _key_string(self, item: str, dims: DimKey, period: str | None = None) -> str | None:
        """_key_strings of a key (or of a series key without its period); None when no row can have it."""
        tokens = dict.fromkeys(self._named, "")
        for name, code in dims:
            level = self._levels[name]
            if tokens[level]:  # two categories on one level
                return None
            tokens[level] = f"{name}{_JOIN}{code}"
        return _SEP.join([str(item), *tokens.values()] + ([str(period)] if period is not None else []))

    def _ref_strings(self, table: pa.Table) -> pa.ChunkedArray:
        """_row_ref of each row without its 'tbl_id|'."""
        tokens = []
        for level in self._named:
            name, code = table.column(f"dim{level}_name"), table.column(f"dim{level}_code")
            named = pc.fill_null(pc.not_equal(name, ""), False)
            tokens.append(pc.if_else(named, pc.binary_join_element_wise(pc.fill_null(code, ""), "|", ""), ""))
        codes = pc.utf8_slice_codeunits(pc.binary_join_element_wise(*tokens, ""), 0, -1) if tokens else ""
        return pc.binary_join_element_wise(pc.fill_null(table.column("item_id"), "None"), codes,
                                           table.column("period"), "|")

    def _item_values(self, item_codes: Iterable[str]) -> list | None:
        """The item_id values of these item codes (None: null), or None when no row can have them."""
        if self.facts.item_null:
            return [None] if "" in item_codes else None
        return list(item_codes)

    def _dims_ok(self, dims: DimKey) -> bool:
        return all(name in self._levels for name, _ in dims)

    def _level_filters(self, dim_keys: Iterable[DimKey]) -> tuple[dict[str, set], list[int]]:
        """(level code columns every key pins, with the codes the keys name; the named levels no key pins)."""
        dim_keys = list(dim_keys)
        pinned: dict[str, set] = {}
        empty: list[int] = []
        if not dim_keys:
            return pinned, empty
        for level in self._named:
            names = self._names_at[level]
            codes = [next((code for name, code in dims if name in names), None) for dims in dim_keys]
            if None not in codes:
                pinned[f"dim{level}_code"] = set(codes)
            elif all(code is None for code in codes):
                empty.append(level)
        return pinned, empty

    # ------------------------------------------------------------ CellStore

    def cell(self, key: CellKey) -> Cell | None:
        return self.cells_at([key]).get(key)

    def cells_at(self, keys: Iterable[CellKey]) -> dict[CellKey, Cell]:
        keys = list(keys)
        got: dict = {}  # this request's answers (the memo is bounded)
        for k in keys:
            found = self._cells.get(k)
            if found is not _MISSING:
                got[k] = found
        todo: dict[CellKey, str] = {}
        for k in keys:
            if k in got or k in todo:
                continue
            text = None
            if k[2] in self._periods and self._dims_ok(k[1]) and self._item_values([k[0]]) is not None:
                text = self._key_string(k[0], k[1], k[2])
            if text is None:
                got[k] = None
                self._cells.put(k, None)
            else:
                todo[k] = text
        if todo:
            want = {"period": {k[2] for k in todo}, "item_id": set(self._item_values({k[0] for k in todo}))}
            pinned, empty = self._level_filters(k[1] for k in todo)
            strings = pa.array(sorted(set(todo.values())), type=pa.string())
            found = set()
            for _, _, _, rows in self._read(want | pinned, empty=empty,
                                            exact=lambda t: pc.is_in(self._key_strings(t, True), value_set=strings)):
                for row in self._dicts(rows):
                    key = self._row_key(row)
                    if key in todo:
                        got[key] = self._cell(row)
                        found.add(key)
            for k in set(todo) - found:
                got[k] = None
                self._cells.put(k, None)
        return {k: got[k] for k in keys if got[k] is not None}

    def _parse(self, ref: str) -> tuple | None:
        """(item_id value, codes, period) of a ref that can name a cell of this table, else None."""
        parts = ref.split("|")
        if len(parts) < 4 or parts[0] != self.meta.tbl_id:
            return None
        item, codes, period = parts[1], tuple(parts[2:-1]), parts[-1]
        if period not in self._periods or len(codes) not in self._code_counts:
            return None
        if self.facts.item_null:
            if item != "None":
                return None
            item = None
        return item, codes, period

    def by_ref(self, ref: str) -> Cell | None:
        return self.by_refs([ref]).get(ref)

    def by_refs(self, refs: Iterable[str]) -> dict[str, Cell]:
        refs = list(refs)
        got: dict = {}  # this request's answers
        for ref in refs:
            found = self._refs.get(ref)
            if found is not _MISSING:
                got[ref] = found
        todo = {}
        for ref in refs:
            if ref in got or ref in todo:
                continue
            parsed = self._parse(ref) if isinstance(ref, str) else None
            if parsed is None:
                got[ref] = None
                self._refs.put(ref, None)
            else:
                todo[ref] = parsed
        if todo:
            want = {"period": {p for _, _, p in todo.values()}, "item_id": {i for i, _, _ in todo.values()}}
            if len(self._layouts) == 1:  # each code sits at a known level ('' may be a null code)
                layout = self._layouts[0]
                for at, level in enumerate(layout):
                    codes = {c[at] if at < len(c) else "" for _, c, _ in todo.values()}
                    if all(len(c) == len(layout) for _, c, _ in todo.values()) and "" not in codes:
                        want[f"dim{level}_code"] = codes
            prefix = len(self.meta.tbl_id) + 1
            strings = pa.array(sorted({ref[prefix:] for ref in todo}), type=pa.string())
            found = set()
            for _, _, _, rows in self._read(want, exact=lambda t: pc.is_in(self._ref_strings(t), value_set=strings)):
                for row in self._dicts(rows):
                    ref = self._row_ref(row)
                    if ref in todo:
                        got[ref] = self._cell(row)
                        found.add(ref)
            for ref in set(todo) - found:
                got[ref] = None
                self._refs.put(ref, None)
        return {ref: got[ref] for ref in refs if got[ref] is not None}

    def series(self, item_code: str, dims: DimKey) -> Sequence[Cell]:
        return self.series_at([(item_code, dims)]).get((item_code, dims), ())

    def series_at(self, keys: Iterable[SeriesKey]) -> dict[SeriesKey, Sequence[Cell]]:
        keys = list(keys)
        got: dict = {}  # this request's answers
        for k in keys:
            found = self._series.get(k)
            if found is not _MISSING:
                got[k] = found
        todo: dict[SeriesKey, str] = {}
        for k in keys:
            if k in got or k in todo:
                continue
            text = None
            if self._dims_ok(k[1]) and self._item_values([k[0]]) is not None:
                text = self._key_string(k[0], k[1])
            if text is None:
                got[k] = ()
                self._series.put(k, ())
            else:
                todo[k] = text
        if todo:
            want = {"item_id": set(self._item_values({k[0] for k in todo}))}
            pinned, empty = self._level_filters(k[1] for k in todo)
            strings = pa.array(sorted(set(todo.values())), type=pa.string())
            groups: dict[SeriesKey, list[Cell]] = {k: [] for k in todo}
            for _, _, _, rows in self._read(want | pinned, empty=empty,
                                            exact=lambda t: pc.is_in(self._key_strings(t, False), value_set=strings)):
                for row in self._dicts(rows):
                    key = self._row_key(row)
                    if (key[0], key[1]) in groups:
                        groups[key[0], key[1]].append(self._cell(row))
            for k, cells in groups.items():
                got[k] = tuple(sorted(cells, key=lambda c: c.period_raw))
                self._series.put(k, got[k])
        return {k: got[k] for k in keys if got[k]}

    def summary(self, codes: Mapping[str, Collection[str]], periods: Collection[str]) -> RangeSummary:
        periods = set(periods) & self._periods
        memo_key = (tuple(sorted((name, tuple(sorted(named))) for name, named in codes.items())), tuple(sorted(periods)))
        found = self._summaries.get(memo_key)
        if found is _MISSING:
            found = self._summary(codes, periods)
            self._summaries.put(memo_key, found)
        return found

    def _summary(self, codes: Mapping[str, Collection[str]], periods: set[str]) -> RangeSummary:
        empty = RangeSummary(count=0, kinds=frozenset(), numeric_periods=(), sample=None)
        if not periods:
            return empty
        want: dict[str, set] = {} if periods == self._periods else {"period": periods}
        exact: list[tuple[str, int | None, set]] = []  # (category name or ITEM_AXIS, level, codes) checked by mask
        columns = {"period", "value_kind"}
        for name, named in codes.items():
            named = set(named)
            if name == ITEM_AXIS:
                values = self._item_values(named)
                if values is None:
                    return empty
                want["item_id"] = set(values)
                continue
            level = self._levels.get(name)
            if level is None:  # no cell has this category: each cell's code on it is ''
                if "" not in named:
                    return empty
                continue
            if "" not in named:
                want[f"dim{level}_code"] = named
            columns |= {f"dim{level}_name", f"dim{level}_code"}
            exact.append((name, level, named))

        def mask_of(keys: pa.Table) -> pa.Array:
            mask = pa.array(np.ones(keys.num_rows, dtype=bool))
            for name, level, named in exact:
                code = pc.cast(keys.column(f"dim{level}_code"), pa.string())
                has = pc.and_(pc.equal(keys.column(f"dim{level}_name"), name), pc.is_valid(code))
                mask = pc.and_(mask, pc.is_in(pc.if_else(pc.fill_null(has, False), code, ""),
                                              value_set=pa.array(sorted(named), type=pa.string())))
            return mask

        count, kinds, numeric, sample = 0, set(), set(), None
        for group, keys, mask, _ in self._read(want, mask_of=mask_of, columns=sorted(columns), rows=False):
            chosen = keys.filter(mask)
            count += chosen.num_rows
            kind = chosen.column("value_kind")
            kinds.update(pc.unique(kind).to_pylist())
            numeric.update(pc.unique(chosen.column("period").filter(pc.fill_null(pc.equal(kind, "ok"), False))).to_pylist())
            if sample is None:
                not_blank = pc.fill_null(pc.not_equal(keys.column("value_kind"), "blank"), True)
                at = pc.index(pc.and_(mask, not_blank), True).as_py()
                if at >= 0:
                    one = pa.array(np.arange(keys.num_rows) == at)
                    sample = self._cell(self._rows_at(group, one)[0])
        return RangeSummary(count=count, kinds=frozenset(kinds), numeric_periods=tuple(sorted(numeric)), sample=sample)
