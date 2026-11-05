"""The table index: the item column and each category as code -> label trees, the periods, and each code's measures.

Moved from kosis_chat/judge/question.py in T1-28 S2 with what it needs (is_total, cell_measure, the axis patterns);
judge/question.py re-exports every name. index_of(table) keeps one index per TableInput (TableInput.caches), so the
pipeline, the judge and the compute checks share it instead of building it again (compute/semantics.py had its own
module-global cache keyed by id(table)).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from kosis_chat.prepare.schemas import Cell, TableInput

try:
    from kosis_ai import normalize
except ImportError:  # same fallback as judge/question.py: src/ is not on the path for `python -m ...` (S6 removes it)
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
    from kosis_ai import normalize

ITEM_AXIS = "항목"  # the item column, next to the category names

_UNIT_PAREN = re.compile(r"\s*\(([^()]*)\)\s*$")
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
    """(unit, measure_kind) by the T1-05 rules (src/kosis_ai/normalize.py)."""
    resolved, _ = normalize.resolve_unit(unit, item, list(dim_labels))
    return resolved, normalize.measure_kind(tbl_id, resolved, item, list(dim_labels))


def cell_measure(table: TableInput, cell: Cell) -> tuple[str | None, str | None]:
    given = cell.unit if cell.unit_source == "given" else None
    # Scenario labels name every assumption ('출산율-중위 / 기대수명-중위'), which the keyword rules would read
    # as a rate in 년 for every cell; the measure is in the other labels.
    labels = tuple(label for name, label in cell.dims.items() if not SCENARIO_AXIS.search(name))
    return _measure(table.meta.tbl_id, given, cell.item, labels)


@dataclass
class TableIndex:
    table: TableInput
    axes: dict[str, Axis]
    periods: list[str]
    # (axis, code) -> every (unit, measure_kind) of its cells; a measure label has exactly one.
    measures: dict[tuple[str, str], set[tuple[str | None, str | None]]]

    def axis_of(self, cell: Cell, name: str) -> str:
        return cell.item_code if name == ITEM_AXIS else cell.dim_codes.get(name, "")

    def year_ok(self, year: str) -> bool:
        return any(p.startswith(year) for p in self.periods)

    def period_ok(self, period: str) -> bool:
        return period in self.periods if len(period) > 4 else self.year_ok(period)

    def measure_of(self, axis: str, code: str) -> tuple[str | None, str | None] | None:
        found = self.measures.get((axis, code), set())
        return next(iter(found)) if len(found) == 1 else None


def build_index(table: TableInput) -> TableIndex:
    axes: dict[str, Axis] = {ITEM_AXIS: Axis(ITEM_AXIS)}
    measures: dict[tuple[str, str], set] = {}
    periods: set[str] = set()
    for c in table.cells:
        periods.add(c.period_raw)
        axes[ITEM_AXIS].labels.setdefault(c.item_code, c.item)
        found = cell_measure(table, c)
        measures.setdefault((ITEM_AXIS, c.item_code), set()).add(found)
        for name, code in c.dim_codes.items():
            axes.setdefault(name, Axis(name)).labels.setdefault(code, c.dims.get(name, ""))
            measures.setdefault((name, code), set()).add(found)
    for axis in axes.values():
        _link_parents(axis)
    return TableIndex(table=table, axes=axes, periods=sorted(periods), measures=measures)


def index_of(table: TableInput) -> TableIndex:
    """The index of this TableInput object, built once (TableInput.caches). Callers must not change it."""
    return table.caches.get(_INDEX_CACHE, lambda: build_index(table))
