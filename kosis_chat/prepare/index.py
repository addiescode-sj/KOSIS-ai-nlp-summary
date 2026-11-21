"""The table index: the item column and each category as code -> label trees, the periods, and each code's measures.

Moved from kosis_chat/judge/question.py in T1-28 S2 with what it needs (is_total, cell_measure, the axis patterns);
judge/question.py re-exports every name. index_of(table) keeps one index per TableInput (TableInput.caches), so the
pipeline, the judge and the compute checks share it instead of building it again (compute/semantics.py had its own
module-global cache keyed by id(table)).
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from kosis_chat.prepare.schemas import Cell, TableInput
from kosis_chat.prepare.store import ITEM_AXIS, axis_code, source_hash

try:
    from kosis_ai import units
except ImportError:  # same fallback as judge/question.py: src/ is not on the path for `python -m ...` (S6 removes it)
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
    from kosis_ai import units

from kosis_chat.shared.patterns import UNIT_PAREN  # noqa: E402  (after the block above: it imports kosis_ai)

_UNIT_PAREN = UNIT_PAREN  # old name (T1-28 S2b: one trailing '(...)' pattern, kosis_ai.units.UNIT_PAREN)
_REGION_AXIS = re.compile(r"행정구역|시도|시군구|지역")
# Categories that are alternative scenarios (장래추계 고위/중위/저위), not parts of a whole.
SCENARIO_AXIS = re.compile(r"가정|시나리오|추계")
TOTAL_LABELS = ("합계", "계", "총계", "전체")

_INDEX_CACHE = "prepare.index"  # LookupCaches name of the index


def is_total(label: str) -> bool:
    core = _UNIT_PAREN.sub("", label).strip()
    return core in TOTAL_LABELS or core == "전국" or core.endswith(("_계", " 계"))


@dataclass
class Axis:
    """The item column or one category: code -> label, code -> parent code (KOSIS codes nest by prefix)."""

    name: str
    labels: dict[str, str] = field(default_factory=dict)
    parents: dict[str, str | None] = field(default_factory=dict)

    @property
    def regional(self) -> bool:
        return bool(_REGION_AXIS.search(self.name))

    @property
    def has_total(self) -> bool:
        return any(is_total(label) for label in self.labels.values())

    def depth(self, code: str) -> int:
        n, parent = 0, self.parents.get(code)
        while parent is not None:
            n, parent = n + 1, self.parents.get(parent)
        return n

    def ancestors(self, code: str) -> list[str]:
        out, parent = [], self.parents.get(code)
        while parent is not None:
            out.append(parent)
            parent = self.parents.get(parent)
        return out


def _link_parents(axis: Axis) -> None:
    codes = set(axis.labels)
    for code in axis.labels:
        axis.parents[code] = next((code[:k] for k in range(len(code) - 1, 0, -1) if code[:k] in codes), None)


# ponytail: S2 keeps this cache: a pure function of its value arguments (not of a table object), bounded, and
# shared by tables that repeat the same item / labels; moving it into LookupCaches would only lose that sharing.
@lru_cache(maxsize=4096)
def _measure(tbl_id: str, unit: str | None, item: str, dim_labels: tuple[str, ...]) -> tuple[str | None, str | None]:
    """(unit, measure_kind) by the T1-05 rules (src/kosis_ai/units.py, re-exported by kosis_ai.normalize)."""
    resolved, _ = units.resolve_unit(unit, item, list(dim_labels))
    return resolved, units.measure_kind(tbl_id, resolved, item, list(dim_labels))


def cell_measure(table: TableInput, cell: Cell) -> tuple[str | None, str | None]:
    given = cell.unit if cell.unit_source == "given" else None
    # Scenario labels name every assumption ('출산율-중위 / 기대수명-중위'), which the keyword rules would read
    # as a rate in 년 for every cell; the measure is in the other labels.
    labels = tuple(label for name, label in cell.dims.items() if not _scenario(name))
    return _measure(table.meta.tbl_id, given, cell.item, labels)


# ponytail: a pure function of a category name, bounded (a table has a few names); not per table.
@lru_cache(maxsize=1024)
def _scenario(name: str) -> bool:
    """SCENARIO_AXIS.search(name), once per category name (cell_measure runs per cell; a table has a few names)."""
    return bool(SCENARIO_AXIS.search(name))


@dataclass
class TableIndex:
    table: TableInput
    axes: dict[str, Axis]
    periods: list[str]
    # (axis, code) -> every (unit, measure_kind) of its cells; a measure label has exactly one.
    measures: dict[tuple[str, str], set[tuple[str | None, str | None]]]
    # False when one item only wraps a '기본항목별' category that holds the real items (DT_1B8000F): facts then
    # leave the item out (compute/engine.py). Here, not in the cell store, because it needs the whole table.
    item_named: bool
    units: list[str]  # sorted distinct non-empty units of the cells (the multi-table prompt shows them)

    def axis_of(self, cell: Cell, name: str) -> str:
        return axis_code(cell, name)

    def year_ok(self, year: str) -> bool:
        return any(p.startswith(year) for p in self.periods)

    def period_ok(self, period: str) -> bool:
        return period in self.periods if len(period) > 4 else self.year_ok(period)

    def measure_of(self, axis: str, code: str) -> tuple[str | None, str | None] | None:
        found = self.measures.get((axis, code), set())
        return next(iter(found)) if len(found) == 1 else None


def build_index(table: TableInput) -> TableIndex:
    """The index of a table: from its cells, or, for a file-backed table, from its store's descriptor cells (one per
    distinct cell but for period and value, in first-seen order) and its periods. Both give the same index for one
    file (T1-31 S3): the loop below reads nothing a descriptor cell does not carry, and a later cell with the same
    descriptor changes nothing (setdefault, sets). A file-backed table's index is saved by its store once built and
    loaded from there by the next process (the saved index, below)."""
    if table.file_backed:
        saved = _saved_index(table)
        if saved is not None:
            return saved
        index = index_from(table, table.store.index_cells(), table.store.periods)
        _save_index(table, index)
        return index
    return index_from(table, table.cells)


def index_from(table: TableInput, cells: Iterable[Cell], periods: Iterable[str] | None = None) -> TableIndex:
    """The index over `cells` in table order (read once); periods: the table's periods, else those of the cells."""
    axes: dict[str, Axis] = {ITEM_AXIS: Axis(ITEM_AXIS)}
    measures: dict[tuple[str, str], set] = {}
    found: set[str] = set() if periods is None else set(periods)
    units: set[str] = set()
    item_codes: set[str] = set()  # not axes[ITEM_AXIS]: a category named '항목' is filed under that axis too
    first = None
    for c in cells:
        if first is None:
            first = c
        if periods is None:
            found.add(c.period_raw)
        item_codes.add(c.item_code)
        if c.unit:
            units.add(c.unit)
        axes[ITEM_AXIS].labels.setdefault(c.item_code, c.item)
        measure = cell_measure(table, c)
        measures.setdefault((ITEM_AXIS, c.item_code), set()).add(measure)
        for name, code in c.dim_codes.items():
            axis = axes.get(name)  # not setdefault(name, Axis(name)): no Axis made and dropped per cell and category
            if axis is None:
                axis = axes[name] = Axis(name)
            axis.labels.setdefault(code, c.dims.get(name, ""))
            measures.setdefault((name, code), set()).add(measure)
    for axis in axes.values():
        _link_parents(axis)
    item_named = len(item_codes) > 1 or first is None or not any("항목" in name for name in first.dims)
    return TableIndex(table=table, axes=axes, periods=sorted(found), measures=measures, item_named=item_named,
                      units=sorted(units))


# ---------------------------------------------------------------- the saved index of a file-backed table (T1-31 S3)
#
# Building the index of a file-backed table loops over every descriptor (up to millions, 25-45 us each), in every
# process. So the built index is saved by the table's store (FileTable.save_index: beside the saved scan, keyed by the
# file's sha256 and the store's own code) as plain JSON data, and the next index_of loads it instead. The payload is
# decided here: every field of the TableIndex but the table, in dict order, with the tbl_id of the meta (the measure
# rules read it, and it comes from tables.csv, not from the file). INDEX_CODE is the code key: the source text of
# every module the index content depends on (below), so an edit of any of them rebuilds the index by itself.
# A payload that does not read back exactly (another format, another table, a wrong type) is built again; nothing
# here raises, and a memory table never saves anything. Nothing is saved or loaded when INDEX_CODE is None (a source
# could not be read: prepare.store.source_hash), nor for an index of more than MAX_SAVED_ENTRIES entries (its JSON
# would cost about as much to load as the index costs to build; the store also refuses to write or read a file over
# its own byte limit, parquet_store.MAX_INDEX_BYTES).

SAVED_FORMAT = 1
# Entries of a saved index (_entries: codes of every axis, measures, periods, units). The largest provided tables
# have about 8,600 (a 700 KB file, 60 to 80 bytes an entry). Measured on made-up tables: 720,000 entries, a 33 MB
# file, loaded in 2.3 s (built in 5.7 s), the load raising the peak by about 0.5 GB, as the build does; 3,000,000
# codes on one category (about 6 million entries), a 288 MB file whose load peaked above its build.
MAX_SAVED_ENTRIES = 1_000_000


def _code_folders() -> list[Path]:
    """The folders whose .py sources the index content depends on: kosis_chat/prepare (this module, the cells'
    conversion in convert.py and clean.py, the store protocol, the schemas), kosis_chat/shared (the label patterns,
    imported through kosis_ai too) and the kosis_ai package (units: the measure and unit rules; schemas, which
    shared imports). The file side (kosis_chat/adapters: the scan and the descriptor cells) is the store's part of
    the key."""
    chat = Path(__file__).resolve().parents[1]
    return [chat / "prepare", chat / "shared", Path(units.__file__).resolve().parent]


def _sources(folders: Iterable[Path]) -> Iterator[tuple[str, Path]]:
    """(folder name / relative path, path) of each folder's .py files, sorted within each folder."""
    for folder in folders:
        for source in sorted(p for p in folder.rglob("*.py") if "__pycache__" not in p.parts):
            yield f"{folder.name}/{source.relative_to(folder).as_posix()}", source


def _source_hash(folders: Iterable[Path]) -> str | None:
    """The sha256 over each folder's .py files (relative path and text, CRLF read as LF: one hash on every OS); None
    when a file cannot be read or there is none (prepare.store.source_hash: the saved index is then not used)."""
    return source_hash(_sources(folders))


INDEX_CODE = _source_hash(_code_folders())


def _payload(index: TableIndex) -> dict:
    return {
        "format": SAVED_FORMAT,
        "tbl_id": index.table.meta.tbl_id,
        "axes": [[key, axis.name, list(axis.labels.items()), list(axis.parents.items())]
                 for key, axis in index.axes.items()],
        "periods": list(index.periods),
        "measures": [[axis, code, [list(m) for m in found]] for (axis, code), found in index.measures.items()],
        "item_named": index.item_named,
        "units": list(index.units),
    }


def _text(value) -> str:
    if type(value) is not str:
        raise TypeError("not a string")
    return value


def _maybe(value) -> str | None:
    return None if value is None else _text(value)


def _from_payload(table: TableInput, payload: dict) -> TableIndex:
    """The TableIndex of a saved payload; an error when it is not exactly one of this table's (format, tbl_id,
    types)."""
    if payload["format"] != SAVED_FORMAT or payload["tbl_id"] != table.meta.tbl_id:
        raise ValueError("another format or table")
    axes: dict[str, Axis] = {}
    for key, name, labels, parents in payload["axes"]:
        axes[_text(key)] = Axis(_text(name), {_text(c): _text(label) for c, label in labels},
                                {_text(c): _maybe(parent) for c, parent in parents})
    measures: dict[tuple[str, str], set] = {}
    for axis, code, found in payload["measures"]:
        measures[_text(axis), _text(code)] = {(_maybe(unit), _maybe(kind)) for unit, kind in found}
    item_named = payload["item_named"]
    if type(item_named) is not bool:
        raise TypeError("item_named")
    return TableIndex(table=table, axes=axes, periods=[_text(p) for p in payload["periods"]], measures=measures,
                      item_named=item_named, units=[_text(u) for u in payload["units"]])


def _entries(index: TableIndex) -> int:
    return sum(len(axis.labels) for axis in index.axes.values()) + len(index.measures) + len(index.periods) + len(
        index.units)


def _saved_index(table: TableInput) -> TableIndex | None:
    if INDEX_CODE is None:
        return None
    try:
        payload = table.store.load_index(INDEX_CODE)
        return None if payload is None else _from_payload(table, payload)
    except Exception:  # a store without saving, or a payload that does not read back: build it
        return None


def _save_index(table: TableInput, index: TableIndex) -> None:
    if INDEX_CODE is None or _entries(index) > MAX_SAVED_ENTRIES:
        return
    try:
        table.store.save_index(INDEX_CODE, _payload(index))
    except Exception:  # the store logs what it cannot write; a store without saving saves nothing
        return


def index_of(table: TableInput) -> TableIndex:
    """The index of this TableInput object, built once (TableInput.caches). Callers must not change it."""
    return table.caches.get(_INDEX_CACHE, lambda: build_index(table))
