"""What the analyze stage passes on (T1-28 S1b): the rules' plan, the built v3 plan and the model plan's outcome.

Moved from compute/analyze.py (Plan), compute/build.py (Pattern, Agg, TermSteps, Built) and llm/analyze_plan.py
(Placed); the old modules re-export them. Same fields, order, defaults and frozenness as the dataclasses; see
shared/boundary.py for how validation is kept from changing a value. AnalyzeFill and EventStart were Pydantic
already and stay in llm/analyze_fill.py and llm/event_lookup.py until S3 moves the LLM steps.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import ConfigDict, Field, SkipValidation

from kosis_chat.shared.boundary import BoundaryModel

from kosis_ai.schemas import PlanV3, QuestionType  # isort: skip

# What the plan does, for the sentence and the chart: the five plain types, one number over periods (avg, sum), two
# period groups compared (pavg, psum), a period against an average (vsa), two shares compared (shr), the gap between
# two targets at two periods (gap), and the change of two targets compared (grw); plan: a plan the model wrote
# (analyze_plan, S5b), whose shape has no name, so its sentence and chart list the numbers it shows.
Pattern = Literal["value", "trend", "compare", "share", "extreme", "avg", "sum", "pavg", "psum", "vsa", "shr", "gap", "grw",
                  "plan"]
Agg = Literal["avg", "sum"]


# ---------------------------------------------------------------- compute/analyze.py


class Plan(BoundaryModel):
    question: str
    type: QuestionType | None
    codes: SkipValidation[dict[str, list[str]]]  # axis -> named codes ("항목" for items), in the order the question names them
    periods: SkipValidation[list[str]]  # table periods asked (all of a range), [] when the question names none
    over: str | None = None  # extreme: the axis ranked over, or PERIOD_AXIS
    pick: str | None = None
    missing_years: SkipValidation[list[str]] = Field(default_factory=list)  # asked years the table does not have
    default_applied: SkipValidation[dict[str, str]] = Field(default_factory=dict)
    unresolved: SkipValidation[list[str]] = Field(default_factory=list)
    # A pattern over periods (avg, sum, pavg, psum, vsa, shr; see Pattern) and its period groups (table periods, in
    # question order). vsa: groups[0] is the averaged range, groups[1] the single period compared with it.
    pattern: Pattern | None = None
    groups: SkipValidation[list[list[str]]] = Field(default_factory=list)
    # (phrase, start) of calendar events periods came from
    events: SkipValidation[list[tuple[str, str]]] = Field(default_factory=list)


# ---------------------------------------------------------------- compute/build.py


class TermSteps(BoundaryModel):
    """One number a period pattern compares: its cell step (the parts for a share), its value step, its total."""

    model_config = ConfigDict(frozen=True)

    cells: str
    value: str
    total: str | None = None
    agg: str | None = None
    periods: SkipValidation[tuple[str, str]] = ("", "")


class Built(BoundaryModel):
    model_config = ConfigDict(frozen=True)

    plan: SkipValidation[PlanV3]
    pattern: Pattern
    terms: SkipValidation[list[TermSteps]] = Field(default_factory=list)
    # cells shown beside the answer, never computed (a 전국 value)
    context: SkipValidation[list[str]] = Field(default_factory=list)
    reference: int | None = None  # compare: which of the two cells the question names as the reference


# ---------------------------------------------------------------- llm/analyze_plan.py


class Placed(BoundaryModel):
    """What place() made of the model's plan: the computed answer, or the problems to say back (and, when the
    plan named something several codes could be, the ask-back)."""

    plan: SkipValidation[PlanV3 | None] = None
    query_plan: SkipValidation[Plan | None] = None  # the plan as the rules' Plan, for to_query (StatQuery slots)
    built: SkipValidation[Built | None] = None
    # compute.schemas.Computed: Any here, since compute/schemas.py imports this module (Pattern)
    computed: Any = None
    problems: SkipValidation[list[str]] = Field(default_factory=list)
    clarify: Any = None  # compute.resolve.NeedsClarify (an exception of the compute stage)
