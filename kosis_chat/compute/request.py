"""What to compute for one question, by source_ref (T1-07 input).

The analyzer (T1-06) fills it from the question; the answer records the same choice in labels (StatQuery).
Cells are named by source_ref ('DT_1B8000H|T10|31|2025'), the key the golden set and the grader use.

v2 (T0-03 v2, T1-25): an operation over periods (op) works on terms, each a group of cells of one series that
aggregates to one number (an average, a sum, a cell, or a share of a total at one period). The op says how many
terms it takes and what each must look like; _shape checks that, so the engine never guesses.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from kosis_ai.schemas import Op, QuestionType

Agg = Literal["avg", "sum"]


class Term(BaseModel):
    """One number for an op: cells aggregated by agg, or a share of total (one period) when total is set."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    cells: list[str] = Field(min_length=1)  # source_refs of one series, in period order
    agg: Agg | None = None  # None: a single cell (or the parts of a share)
    total: str | None = None  # share: the denominator cell at the same period

    @model_validator(mode="after")
    def _shape(self) -> Term:
        if self.agg and self.total:
            raise ValueError("a term is either an aggregate over periods or a share at one period, not both")
        if self.agg and len(self.cells) < 2:
            raise ValueError(f"{self.agg} needs two or more cells, got {len(self.cells)}")
        if not self.agg and not self.total and len(self.cells) != 1:
            raise ValueError(f"a single-cell term holds one cell, got {len(self.cells)}")
        return self


# What each op takes: (number of terms, agg of each term, whether the terms are shares).
OP_TERMS: dict[str, tuple[int, tuple[Agg | None, ...], bool]] = {
    "avg": (1, ("avg",), False),
    "sum": (1, ("sum",), False),
    "pavg": (2, ("avg", "avg"), False),
    "psum": (2, ("sum", "sum"), False),
    "vsa": (2, ("avg", None), False),  # the average first, then the period compared with it
    "shr": (2, (None, None), True),  # a share at the earlier period, then at the later one
}


class CalcRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    type: QuestionType
    # value: one cell; trend: the series in period order; compare: two cells; share: the parts (summed).
    cells: list[str] = Field(default_factory=list)
    total: str | None = None  # share: the denominator, a total cell of the table
    reference: int | None = None  # compare: index in cells of the cell named as the reference ("전국보다")
    # extreme: cells to rank, and for a share ranking the total of each candidate in the same order.
    candidates: list[str] = Field(default_factory=list)
    candidate_totals: list[str] = Field(default_factory=list)
    pick: Literal["max", "min"] | None = None
    context: list[str] = Field(default_factory=list)  # extra cells worth showing: the 전국 value, a newer '-' period
    note: str | None = None  # what the answer should say about the choice (missing years, default period)
    # v2: an operation over periods; its terms replace cells / total (which stay empty).
    op: Op | None = None
    terms: list[Term] = Field(default_factory=list)

    @model_validator(mode="after")
    def _shape(self) -> CalcRequest:
        if self.op:
            return self._op_shape()
        if self.terms:
            raise ValueError("terms need an op")
        need = {"value": (1, 1), "trend": (2, None), "compare": (2, 2), "share": (1, None)}
        if self.type in need:
            low, high = need[self.type]
            if len(self.cells) < low or (high is not None and len(self.cells) > high):
                raise ValueError(f"{self.type} needs {low}{'' if high == low else '+'} cells, got {len(self.cells)}")
        if self.type == "share" and not self.total:
            raise ValueError("share needs a total cell")
        if self.type == "extreme":
            if len(self.candidates) < 2 or self.pick is None:
                raise ValueError("extreme needs two or more candidates and pick")
            if self.candidate_totals and len(self.candidate_totals) != len(self.candidates):
                raise ValueError("candidate_totals must match candidates")
        if self.reference is not None and (self.type != "compare" or self.reference not in (0, 1)):
            raise ValueError("reference is an index (0 or 1) into the two compare cells")
        return self

    def _op_shape(self) -> CalcRequest:
        count, aggs, shares = OP_TERMS[self.op]
        if self.cells or self.total or self.candidates or self.reference is not None or self.pick:
            raise ValueError(f"op {self.op} takes terms only; cells, total, candidates, reference and pick stay empty")
        if self.type not in ("value", "trend", "share") or (self.type == "share") != shares:
            raise ValueError(f"op {self.op} goes with type {'share' if shares else 'value or trend'}, not {self.type}")
        if len(self.terms) != count:
            raise ValueError(f"op {self.op} needs {count} terms, got {len(self.terms)}")
        for n, (term, agg) in enumerate(zip(self.terms, aggs)):
            if term.agg != agg or bool(term.total) != shares:
                want = "a share of a total" if shares else (f"an {agg}" if agg else "a single cell")
                raise ValueError(f"op {self.op}: term {n + 1} must be {want}")
        if self.op == "shr" and len({len(t.cells) for t in self.terms}) != 1:
            raise ValueError("op shr: both shares need the same parts")
        return self
