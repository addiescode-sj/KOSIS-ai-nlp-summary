"""StatAnswer.table (S6): the grid of targets by periods (chart/grid.py) as period columns and rows of fact ids.

T1-28 S3a: moved here from answer/table.py (a re-export shim now).
"""

from __future__ import annotations

from kosis_chat.chart.grid import grid
from kosis_chat.compute import Computed
from kosis_chat.shared.periods import period_text

from kosis_ai.schemas import TablePart, TableRow  # isort: skip


def table_part(computed: Computed) -> TablePart | None:
    g = grid(computed)
    if g is None:
        return None
    return TablePart(columns=[period_text(p) for p in g.periods],
                     rows=[TableRow(label=name, fact_ids=[f.id if f else None for f in facts]) for name, facts in g.rows])
