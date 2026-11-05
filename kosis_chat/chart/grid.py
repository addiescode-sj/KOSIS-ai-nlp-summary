"""The answer's cell facts as a grid of targets by periods (S6): read by the grouped / multi-line charts and by
StatAnswer.table (render/table.py).

A grid needs two or more targets (the categories that differ between the cells) and two or more periods, in one
unit; the table and the charts hold fact ids, the numbers stay in the facts.
T1-28 S3a: moved here from answer/table.py (a re-export shim now).
"""

from __future__ import annotations

from dataclasses import dataclass

from kosis_chat.compute import Computed
from kosis_chat.shared.labels import item_label

from kosis_ai.schemas import Fact  # isort: skip


@dataclass(frozen=True)
class Grid:
    periods: list[str]
    rows: list[tuple[str, list[Fact | None]]]  # (target name, its fact at each period)
    unit: str | None


def grid(computed: Computed) -> Grid | None:
    by_id = {f.id: f for f in computed.result.facts}
    cells = [by_id[i] for i in dict.fromkeys(computed.main) if i in by_id and by_id[i].source_ref and not by_id[i].derivation]
    if len(cells) < 4 or len({f.unit for f in cells}) != 1:
        return None
    periods = sorted({f.period or "" for f in cells})
    keys = [k for k in dict.fromkeys(k for f in cells for k in (f.dims or {}))
            if len({(f.dims or {}).get(k) for f in cells}) > 1]
    if len(periods) < 2 or not keys:
        return None
    rows: dict[str, dict[str, Fact]] = {}
    for f in cells:
        name = " ".join(item_label((f.dims or {}).get(k, "")) for k in keys).strip()
        if f.period in rows.setdefault(name, {}):
            return None  # two facts in one place: not a grid
        rows[name][f.period or ""] = f
    if len(rows) < 2:
        return None
    return Grid(periods, [(name, [row.get(p) for p in periods]) for name, row in rows.items()], cells[0].unit)

