"""What the compute stage passes on (T1-28 S1b): the computed facts and what templates and charts read beside them.

Moved from compute/engine.py (TermResult, Computed), compute/context.py (Alternatives) and compute/resolve.py
(Resolved: the cells of one select step; not the judge's Resolved, which lives in judge/schemas.py); the old modules
re-export them. Same fields, order, defaults and frozenness as the dataclasses; see shared/boundary.py for how
validation is kept from changing a value. The plan engine's own records (plan_engine.py: Cells Scalar Ranking Found
Mapped PlanRun) and engine._Read stay dataclasses: they never leave the stage.
"""

from __future__ import annotations

from typing import Any

from pydantic import ConfigDict, Field, SkipValidation

from kosis_chat.analyze.schemas import Pattern
from kosis_chat.shared.boundary import BoundaryModel

from kosis_ai.schemas import StatResult  # isort: skip

# ---------------------------------------------------------------- compute/engine.py


class TermResult(BoundaryModel):
    """What one number of a period pattern became: its cell fact ids and the id of the number it stands for."""

    model_config = ConfigDict(frozen=True)

    cell_ids: SkipValidation[list[str]]
    value_id: str  # the aggregate / share fact, or the single cell
    periods: SkipValidation[tuple[str, str]]  # first and last period of the term's cells
    digits: int  # decimals the number keeps
    agg: str | None = None  # "avg" / "sum" / None (a cell or a share)


class Computed(BoundaryModel):
    model_config = ConfigDict(frozen=True)

    result: SkipValidation[StatResult]
    main: SkipValidation[list[str]]  # fact ids the answer should cite, in order (the rest are context)
    note: str | None = None
    # extreme: {"winner": fact id, "runner_up": fact id}
    picked: SkipValidation[dict[str, str]] = Field(default_factory=dict)
    # extreme: every ranked candidate for the chart, best first: {"dims", "period", "value", "unit"} (facts keep two)
    ranking: SkipValidation[list[dict]] = Field(default_factory=list)
    # value: the same cell in every period, for the chart: {"period", "value"} in period order (facts keep one)
    series: SkipValidation[list[dict]] = Field(default_factory=list)
    pattern: Pattern | None = None  # the shape the rules built (build.py): templates and charts choose by it
    # period patterns: what each compared number became, earlier first (templates and charts read them)
    terms: SkipValidation[list[TermResult]] = Field(default_factory=list)
    plan: Any = None  # the PlanV3 that made these facts (the dataclass typed it `object | None`)
    # allowed numbers that need a word of care (S6 shows them)
    cautions: SkipValidation[list[str]] = Field(default_factory=list)


# ---------------------------------------------------------------- compute/context.py


class Alternatives(BoundaryModel):
    model_config = ConfigDict(frozen=True)

    axis: str
    chosen: str  # label of the scenario answered
    refs: SkipValidation[list[str]]


# ---------------------------------------------------------------- compute/resolve.py


class Resolved(BoundaryModel):
    cells: SkipValidation[list[str]]
    cautions: SkipValidation[list[str]] = Field(default_factory=list)
