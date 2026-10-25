"""What to compute for one question, by source_ref (T1-07 input).

The analyzer (T1-06) fills it from the question; the answer records the same choice in labels (StatQuery).
Cells are named by source_ref ('DT_1B8000H|T10|31|2025'), the key the golden set and the grader use.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from kosis_ai.schemas import QuestionType


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

    @model_validator(mode="after")
    def _shape(self) -> CalcRequest:
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
