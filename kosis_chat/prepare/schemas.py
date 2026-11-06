"""Shared table types (TableMeta, Cell, TableInput) and their validation, as defined in the baseline PLAN.

Moved from kosis_chat/data/schema.py in T1-28 S2 (data/schema.py re-exports every name). prepare is the first stage,
so every stage may import these. TableInput also carries the per-table lookup caches (LookupCaches) that replaced the
module-global dicts keyed by id(table) in compute/ (T1-28 S2).
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, TypeVar

REQUIRED_META_FIELDS = (
    "tbl_id",
    "tbl_nm",
    "survey_nm",
    "org_nm",
    "period_type",
    "source_url",
    "fetched_at",
)
PERIOD_TYPES = ("Y", "M", "Q", "H")
INPUT_FORMATS = ("parquet",)  # data/values/<org>_<tbl>.parquet
VALUE_FLAGS = ("ok", "missing", "suppressed", "flagged", "provisional", "unparsed")

T = TypeVar("T")


@dataclass(frozen=True)
class Problem:
    """One validation problem: where it is and why it is a problem."""

    where: str
    message: str

    def __str__(self) -> str:
        return f"{self.where}: {self.message}"


class SchemaError(ValueError):
    """Raised when input violates the input contract; carries every problem found."""

    def __init__(self, problems: list[Problem]) -> None:
        self.problems = problems
        super().__init__("; ".join(str(p) for p in problems))


class LookupCaches:
    """Lookups derived from one TableInput (cells by key, series, the index), built on first use and kept with it.

    Replaces the module-global dicts keyed by id(table) (T1-28 S2). A new TableInput (dataclasses.replace, clean,
    scope.apply) starts with empty caches, as the identity check of the old dicts did. Callers must not change a
    returned value. No lock: two threads asking for the same name at once may both build it; both results are equal
    and the last one is kept, as with the old dicts (one pipeline is used by one thread at a time today, T2-10).
    """

    __slots__ = ("_store",)

    def __init__(self) -> None:
        self._store: dict[str, Any] = {}

    def get(self, name: str, build: Callable[[], T]) -> T:
        try:
            return self._store[name]
        except KeyError:
            value = self._store[name] = build()
            return value

    def __contains__(self, name: str) -> bool:
        return name in self._store

    def __len__(self) -> int:
        return len(self._store)

    def clear(self) -> None:
        self._store.clear()


@dataclass(frozen=True)
class TableMeta:
    tbl_id: str
    tbl_nm: str
    survey_nm: str
    org_nm: str
    period_type: str
    source_url: str
    fetched_at: str
    input_format: str
    file_sha256: str
    note: str = ""


@dataclass(frozen=True)
class Cell:
    item: str
    dims: dict[str, str]
    period_raw: str
    value_raw: str
    unit: str
    source_ref: str
    dim_codes: dict[str, str] = field(default_factory=dict)
    item_code: str = ""  # KOSIS item id (T1 ...); the item part of source_ref
    # What the Parquet build read from value_raw: value_kind (ok, blank, dash, x_secret, asterisk, ...) and the number.
    value_kind: str | None = None
    source_value: float | None = None
    # Derived by preprocessing; value_raw is never changed.
    value: float | None = None
    value_flag: str | None = None
    # Where the unit came from: given | item_name | dim_label | override | unknown (see prepare.convert.infer_unit).
    unit_source: str = "given"


@dataclass(frozen=True)
class TableInput:
    meta: TableMeta
    cells: tuple[Cell, ...]
    # Not a field callers pass or compare: lookups derived from this object (LookupCaches). init=False, so
    # dataclasses.replace gives the new table its own empty caches.
    caches: LookupCaches = field(default_factory=LookupCaches, init=False, repr=False, compare=False)

    @property
    def period_range(self) -> tuple[str, str] | None:
        """Smallest and largest raw period string, or None for an empty table."""
        periods = sorted(c.period_raw for c in self.cells)
        return (periods[0], periods[-1]) if periods else None

    @property
    def dim_names(self) -> list[str]:
        """Category names in first-seen order."""
        names: dict[str, None] = {}
        for c in self.cells:
            names.update(dict.fromkeys(c.dims))
        return list(names)


def validate_meta_dict(meta: dict, where: str = "meta", *, allow_empty: tuple[str, ...] = ()) -> list[Problem]:
    """Check required meta fields before a TableMeta is built. allow_empty: required fields that may be empty
    (fetched_at of a table given in memory with no date, T1-28 S2); files always have every field."""
    problems = [
        Problem(f"{where}.{name}", "required field is missing or empty")
        for name in REQUIRED_META_FIELDS
        if name not in allow_empty and not str(meta.get(name) or "").strip()
    ]
    if meta.get("period_type") and meta["period_type"] not in PERIOD_TYPES:
        problems.append(Problem(f"{where}.period_type", f"must be one of {PERIOD_TYPES}"))
    if meta.get("fetched_at"):
        try:
            dt.date.fromisoformat(str(meta["fetched_at"]))
        except ValueError:
            problems.append(Problem(f"{where}.fetched_at", "must be a date in YYYY-MM-DD form"))
    return problems


def validate(table: TableInput, *, allow_empty: tuple[str, ...] = ()) -> list[Problem]:
    """Return every contract problem in a built table; an empty list means valid."""
    meta_dict = {name: getattr(table.meta, name) for name in REQUIRED_META_FIELDS}
    problems = validate_meta_dict(meta_dict, allow_empty=allow_empty)
    if table.meta.input_format not in INPUT_FORMATS:
        problems.append(Problem("meta.input_format", f"must be one of {INPUT_FORMATS}"))
    if not table.meta.file_sha256:
        problems.append(Problem("meta.file_sha256", "required field is missing"))
    if not table.cells:
        problems.append(Problem("cells", "table has no cells"))
    seen: dict[str, int] = {}
    for i, cell in enumerate(table.cells):
        # Two rows with one source_ref would silently overwrite each other in the calculation indexes (T0-03 v2).
        first = seen.setdefault(cell.source_ref, i)
        if first != i:
            problems.append(Problem(f"cells[{i}].source_ref", f"duplicate of cells[{first}] ({cell.source_ref})"))
        for name in ("item", "period_raw", "unit", "source_ref"):
            if not str(getattr(cell, name) or "").strip():
                problems.append(Problem(f"cells[{i}].{name}", "required field is missing or empty"))
        if cell.value_raw is None:
            problems.append(Problem(f"cells[{i}].value_raw", "required field is missing"))
        if cell.value_flag is not None and cell.value_flag not in VALUE_FLAGS:
            problems.append(Problem(f"cells[{i}].value_flag", f"must be one of {VALUE_FLAGS}"))
    return problems
