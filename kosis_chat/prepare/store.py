"""The cell store: every read of a table's cells after loading goes through one object (T1-31 S1, S2).

Today a table holds all its cells in memory (TableInput.cells), and each stage built its own dicts over them. A table
of tens of millions of rows cannot be held like that, so the stages ask a CellStore for the cells they need instead:
by key, by source_ref, series, or a summary of a range. MemoryStore answers from TableInput.cells, with the same
dict semantics the stages had (on a duplicate key or ref the last cell wins; a series keeps table order within one
period). A table too big to hold (T1-31 S3) is file-backed: its TableInput.store answers each request by reading only
the matching rows of the Parquet file (kosis_chat/adapters/parquet_store.py), and store_of returns it.

Facts about the whole table (the periods, the item codes, whether the item is named, the units) are not asked of the
store: the table index (prepare.index) gathers them once when the table is loaded.

prepare is the first stage: this module imports only prepare.schemas, so every stage (and prepare.index) may use it.
"""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Collection, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Protocol

from kosis_chat.prepare.schemas import Cell, TableInput

log = logging.getLogger(__name__)

ITEM_AXIS = "항목"  # the item column, next to the category names (prepare.index re-exports it)

DimKey = tuple[tuple[str, str], ...]  # sorted (category name, code) pairs of one cell
CellKey = tuple[str, DimKey, str]  # (item_code, DimKey, period_raw)
SeriesKey = tuple[str, DimKey]  # (item_code, DimKey): one cell per period

_STORE_CACHE = "prepare.store"  # LookupCaches name of the store


def dim_key(cell: Cell) -> DimKey:
    return tuple(sorted(cell.dim_codes.items()))


def cell_key(cell: Cell) -> CellKey:
    return cell.item_code, dim_key(cell), cell.period_raw


def axis_code(cell: Cell, name: str) -> str:
    """The cell's code on the item column or a category ('' when the cell lacks that category)."""
    return cell.item_code if name == ITEM_AXIS else cell.dim_codes.get(name, "")


@dataclass(frozen=True)
class RangeSummary:
    """What the judge rules need to know about the cells a question points at, without the cells themselves."""

    count: int
    kinds: frozenset[str | None]  # value_kind of every cell
    numeric_periods: tuple[str, ...]  # sorted distinct period_raw of the cells whose value_kind is "ok"
    sample: Cell | None  # the first cell in table order whose value_kind is not "blank"

    @classmethod
    def of(cls, cells: Iterable[Cell]) -> RangeSummary:
        cells = list(cells)
        return cls(
            count=len(cells),
            kinds=frozenset(c.value_kind for c in cells),
            numeric_periods=tuple(sorted({c.period_raw for c in cells if c.value_kind == "ok"})),
            sample=next((c for c in cells if c.value_kind != "blank"), None),
        )


class CellStore(Protocol):
    """The cells of one table, asked for by what a stage needs.

    A store serves a table that passed prepare.schemas.validate(), so no two cells share a source_ref. Unique refs do
    not make unique keys, even through the real loader: period "2020" next to " 2020", item_id null next to "", or
    the same two categories on swapped levels give two refs whose cells share one CellKey. MemoryStore then answers
    as the stages' dicts did: cell and cells_at give the last of those cells in table order, series and series_at
    give all of them, summary counts all of them, and by_ref and by_refs give each ref its own cell. Such a table is
    outside what a store must reproduce: a file store checks that keys are unique when it builds its index and is not
    used for a table that fails (kosis_chat/adapters/parquet_store.py lists the conditions).

    It returns the cleaned cells (value and value_flag set, as prepare.clean.clean_cell gives); MemoryStore returns
    the cells it was given, so callers still clean a cell whose value_flag is None (tests hand in raw tables).

    Every call is one request: a file store answers it with one filtered read, and a read costs about the same for one
    cell as for a few hundred. So a caller asks for every cell a step needs at once (cells_at, by_refs, series_at),
    never one call per cell in a loop. A batch method given no keys or refs returns {} without reading; callers
    still skip such a call where that is cheap."""

    def cell(self, key: CellKey) -> Cell | None: ...

    def cells_at(self, keys: Iterable[CellKey]) -> dict[CellKey, Cell]:
        """The cells of many keys in one request; a key with no cell is absent."""
        ...

    def by_ref(self, ref: str) -> Cell | None: ...

    def by_refs(self, refs: Iterable[str]) -> dict[str, Cell]:
        """The cells of many source_refs in one request; a ref with no cell is absent."""
        ...

    def series(self, item_code: str, dims: DimKey) -> Sequence[Cell]:
        """Every cell of one (item, dims), sorted by period_raw; table order within one period."""
        ...

    def series_at(self, keys: Iterable[SeriesKey]) -> dict[SeriesKey, Sequence[Cell]]:
        """series() of many (item, dims) in one request; a key with no cell is absent."""
        ...

    def summary(self, codes: Mapping[str, Collection[str]], periods: Collection[str]) -> RangeSummary:
        """The cells in `periods` whose code on each named axis (ITEM_AXIS or a category) is one of its codes."""
        ...


class RefReader(Protocol):
    """The one CellStore method compose.caveats needs: graph/multi.py hands in several tables' stores as one."""

    def by_refs(self, refs: Iterable[str]) -> dict[str, Cell]: ...


class MemoryStore:
    """A CellStore over cells held in memory. Each lookup is built on first use and kept; callers must not change
    a returned value. key_map and ref_map are the whole dicts, for the helpers kept for in-memory tables."""

    def __init__(self, cells: Iterable[Cell]) -> None:
        self.cells = tuple(cells)

    @classmethod
    def of(cls, table: TableInput) -> MemoryStore:
        """The store of this TableInput object, built once (TableInput.caches)."""
        return table.caches.get(_STORE_CACHE, lambda: cls(table.cells))

    @cached_property
    def key_map(self) -> dict[CellKey, Cell]:
        return {cell_key(c): c for c in self.cells}

    @cached_property
    def ref_map(self) -> dict[str, Cell]:
        return {c.source_ref: c for c in self.cells}

    @cached_property
    def _series(self) -> dict[tuple[str, DimKey], tuple[Cell, ...]]:
        groups: dict[tuple[str, DimKey], list[Cell]] = {}
        for c in self.cells:
            groups.setdefault((c.item_code, dim_key(c)), []).append(c)
        return {key: tuple(sorted(cells, key=lambda c: c.period_raw)) for key, cells in groups.items()}

    def cell(self, key: CellKey) -> Cell | None:
        return self.key_map.get(key)

    def cells_at(self, keys: Iterable[CellKey]) -> dict[CellKey, Cell]:
        found = self.key_map
        return {k: found[k] for k in keys if k in found}

    def by_ref(self, ref: str) -> Cell | None:
        return self.ref_map.get(ref)

    def by_refs(self, refs: Iterable[str]) -> dict[str, Cell]:
        found = self.ref_map
        return {r: found[r] for r in refs if r in found}

    def series(self, item_code: str, dims: DimKey) -> Sequence[Cell]:
        return self._series.get((item_code, dims), ())

    def series_at(self, keys: Iterable[SeriesKey]) -> dict[SeriesKey, Sequence[Cell]]:
        found = self._series
        return {k: found[k] for k in keys if k in found}

    def summary(self, codes: Mapping[str, Collection[str]], periods: Collection[str]) -> RangeSummary:
        periods = set(periods)
        return RangeSummary.of(
            c for c in self.cells
            if c.period_raw in periods and all(axis_code(c, name) in named for name, named in codes.items())
        )


class FileTable(CellStore, Protocol):
    """The store of a file-backed table (TableInput.store), with what prepare.index reads instead of the cells: the
    sorted periods of the table and one cell per distinct "descriptor" (everything a cell has but its period and
    value), in the order the table first shows each, made by the same conversion as the cells."""

    periods: Sequence[str]

    def index_cells(self) -> Iterator[Cell]: ...

    def load_index(self, code: str | None) -> object | None:
        """The payload save_index saved for this file's bytes and this code (prepare.index.INDEX_CODE), else None.
        The store adds its own code to the key (the scan and the descriptor cells are its part); it never raises.
        None when either code is None (source_hash: no trustworthy code key)."""
        ...

    def save_index(self, code: str | None, payload: object) -> None:
        """Keep a JSON-able payload (prepare.index decides it: the built index) for load_index; nothing when the
        store has nowhere to save it, or when either code is None. It never raises (a folder that cannot be written
        is logged once)."""
        ...


def store_of(table: TableInput) -> CellStore:
    """The cell store of this TableInput object: the file store of a file-backed table, else its MemoryStore (one per
    object, TableInput.caches)."""
    return table.store if table.file_backed else MemoryStore.of(table)


_untrusted_warned = False  # source_hash has logged its warning in this process


def source_hash(sources: Iterable[tuple[str, Path]], *, required: int = 1) -> str | None:
    """The code key of saved data: the sha256 over the name and text (CRLF read as LF: one hash on every OS) of each
    (name, path) of `sources`, in the order given. Paths that are not files are skipped (a folder named like a module).

    None, the key that cannot be trusted, when a source cannot be read (a file locked by a scanner, a cloud-only
    placeholder, a folder that cannot be listed: any OSError) or fewer than `required` files were hashed (an install
    without .py files, or a zipped one: the hash would be the same for every version). Saved data is then neither
    used nor saved. Never raises: computed at import, a failure here must not stop the package from loading. One
    warning per process."""
    digest = hashlib.sha256()
    hashed = 0
    try:
        for name, path in sources:
            if not path.is_file():
                continue
            text = hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
            digest.update(f"{name}\0{text}\n".encode("utf-8"))
            hashed += 1
    except OSError as exc:
        return _untrusted(f"{type(exc).__name__}: {exc}")
    if hashed < required:
        return _untrusted(f"{hashed} source files found, {required} needed")
    return digest.hexdigest()


def _untrusted(why: str) -> None:
    global _untrusted_warned
    if not _untrusted_warned:
        _untrusted_warned = True
        log.warning("cannot hash the package's source files (%s): saved scans and table indexes of big tables are "
                    "neither used nor saved, so each load scans them again", why)
    return None
