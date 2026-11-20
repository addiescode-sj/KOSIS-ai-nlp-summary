"""The cell store: every read of a table's cells after loading goes through one object (T1-31 S1).

Today a table holds all its cells in memory (TableInput.cells), and each stage built its own dicts over them. A table
of tens of millions of rows cannot be held like that, so the stages ask a CellStore for the cells they need instead:
by key, by source_ref, one series, or a summary of a range. MemoryStore answers from TableInput.cells, with the same
dict semantics the stages had (on a duplicate key or ref the last cell wins; a series keeps table order within one
period). A later stage adds a store that reads only the needed rows from the Parquet file.

prepare is the first stage: this module imports only prepare.schemas, so every stage (and prepare.index) may use it.
"""

from __future__ import annotations

from collections.abc import Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass
from functools import cached_property
from typing import Protocol

from kosis_chat.prepare.schemas import Cell, TableInput

ITEM_AXIS = "항목"  # the item column, next to the category names (prepare.index re-exports it)

DimKey = tuple[tuple[str, str], ...]  # sorted (category name, code) pairs of one cell
CellKey = tuple[str, DimKey, str]  # (item_code, DimKey, period_raw)

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
    """The cells of one table (or of several tables seen as one), asked for by what a stage needs."""

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

    def summary(self, codes: Mapping[str, Collection[str]], periods: Collection[str]) -> RangeSummary:
        """The cells in `periods` whose code on each named axis (ITEM_AXIS or a category) is one of its codes."""
        ...

    def item_codes(self) -> frozenset[str]: ...

    def first(self) -> Cell | None:
        """The first cell in table order."""
        ...

    def units(self) -> list[str]:
        """Sorted distinct non-empty units."""
        ...


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

    @cached_property
    def _item_codes(self) -> frozenset[str]:
        return frozenset(c.item_code for c in self.cells)

    @cached_property
    def _units(self) -> list[str]:
        return sorted({c.unit for c in self.cells if c.unit})

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

    def summary(self, codes: Mapping[str, Collection[str]], periods: Collection[str]) -> RangeSummary:
        periods = set(periods)
        return RangeSummary.of(
            c for c in self.cells
            if c.period_raw in periods and all(axis_code(c, name) in named for name, named in codes.items())
        )

    def item_codes(self) -> frozenset[str]:
        return self._item_codes

    def first(self) -> Cell | None:
        return self.cells[0] if self.cells else None

    def units(self) -> list[str]:
        return list(self._units)


def store_of(table: TableInput) -> CellStore:
    """The cell store of this TableInput object, one per object (TableInput.caches)."""
    return MemoryStore.of(table)
