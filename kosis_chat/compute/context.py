"""Cells shown beside an answer (S6), never computed with: Built.context, listed in StatAnswer.context_fact_ids.

The one rule so far is global to projection tables: a scenario category (장래추계 가정별: 고위 / 중위 / 저위) the
question did not choose is answered at the base scenario (judge/answerability.defaults), and the same cell under the
other scenarios is shown with it, so the answer does not pass one assumption off as the only number. It applies to
every table with a scenario category (21 of the 2,364 tables of data/table_metadata on 10/28).
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from kosis_chat.compute.build import Built
from kosis_chat.data.schema import TableInput
from kosis_chat.judge.question import SCENARIO_AXIS, TableIndex

MAX_ALTERNATIVES = 8


@dataclass(frozen=True)
class Alternatives:
    axis: str
    chosen: str  # label of the scenario answered
    refs: list[str]


def scenario_alternatives(table: TableInput, index: TableIndex, built: Built, defaulted: dict[str, str]) -> list[Alternatives]:
    """For a one-cell answer: the cell under each other scenario of every scenario category the rules defaulted."""
    if built.pattern != "value":
        return []
    ref = built.plan.steps[0].cells[0]
    by_ref = {c.source_ref: c for c in table.cells}
    cell = by_ref.get(ref)
    if cell is None:
        return []
    by_key = {(c.item_code, tuple(sorted(c.dim_codes.items())), c.period_raw): c for c in table.cells}
    out = []
    for name in defaulted:
        if not SCENARIO_AXIS.search(name) or name not in cell.dim_codes or name not in index.axes:
            continue
        chosen = cell.dim_codes[name]
        refs = []
        for code in index.axes[name].labels:
            other = by_key.get((cell.item_code, tuple(sorted({**cell.dim_codes, name: code}.items())), cell.period_raw))
            if code != chosen and other is not None:
                refs.append(other.source_ref)
        if refs:
            out.append(Alternatives(name, cell.dims.get(name, chosen), refs[:MAX_ALTERNATIVES]))
    return out


def with_alternatives(built: Built, found: list[Alternatives]) -> Built:
    refs = [r for a in found for r in a.refs if r not in built.context]
    return replace(built, context=[*built.context, *refs]) if refs else built
