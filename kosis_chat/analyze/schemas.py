"""What the analyze stage passes on (T1-28 S1b): the rules' plan, the built v3 plan and the model plan's outcome.

Moved from compute/analyze.py (Plan), compute/build.py (Pattern, Agg, TermSteps, Built) and llm/analyze_plan.py
(Placed); the old modules re-export them. Same fields, order, defaults and frozenness as the dataclasses; see
shared/boundary.py for how validation is kept from changing a value. AnalyzeFill and EventStart were Pydantic
already; S3b moved them with their modules to analyze/fill.py and analyze/event_lookup.py (the model's reply schemas).
S3a/S3b: the In/Out models of the analyze steps (analyze/node.py).
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import ConfigDict, Field, SkipValidation

from kosis_chat.schemas.state import LLMCall
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


# ---------------------------------------------------------------- analyze/node.py (T1-28 S3a)


class PlanRulesIn(BoundaryModel):
    """step_plan_rules's input: the question and the defaults the judge applied (Verdict.default_applied)."""

    question: str
    default_applied: SkipValidation[dict[str, str]] = Field(default_factory=dict)


class PlanRulesOut(BoundaryModel):
    """step_plan_rules's output: the rules' plan and, when the rules could place it, the built v3 plan.

    route "built": built is set. route "unresolved": built is None and unresolved names what the rules could not
    decide (Unresolved.what), for the model's fill (plan_fill_llm)."""

    rule: SkipValidation[Plan]
    built: SkipValidation[Built | None] = None
    unresolved: SkipValidation[list[str] | None] = None
    route: Literal["built", "unresolved"]


# ---------------------------------------------------------------- analyze/node.py (T1-28 S3b): the model steps


class PlanFillIn(BoundaryModel):
    """step_plan_fill_llm's input: the rules' plan, what they could not decide (PlanRulesOut.unresolved), and whether a
    model answer came back already for this question (then a call waits retry_interval first)."""

    rule: SkipValidation[Plan]
    unresolved: SkipValidation[list[str]] = Field(default_factory=list)
    called: bool = False


class PlanFillOut(BoundaryModel):
    """step_plan_fill_llm's output. route "built": the model's fill placed the question, plan and built are set.
    route "unplaced": built is None (plan: the last fill, or the rules' plan when no fill came back).
    calls: the model calls made (analyze.llm_fill, analyze.llm_fill_retry); meta: the decision_metadata entries
    written (route, fill, fill_status, fill_retry, fill_retry_status, unresolved_after_fill), in the order written."""

    plan: SkipValidation[Plan]
    built: SkipValidation[Built | None] = None
    calls: SkipValidation[list[LLMCall]] = Field(default_factory=list)
    meta: SkipValidation[dict[str, Any]] = Field(default_factory=dict)
    route: Literal["built", "unplaced"]


class PlanWriteIn(BoundaryModel):
    """step_plan_write_llm's input: the question, the rules' plan (what they found is shown to the model) and whether a
    model answer came back already for this question."""

    question: str
    rule: SkipValidation[Plan]
    called: bool = False


class PlanWriteOut(BoundaryModel):
    """step_plan_write_llm's output. route "placed": placed holds the computed plan (query_plan, built, computed).
    route "clarify": clarify is the ask-back (compute.resolve.NeedsClarify, two options or more). route "failed": no
    plan (the compact-v2 fallback answers). calls: analyze.llm_plan, event_lookup, analyze.llm_plan_retry in the order
    made; meta: event_lookup, route, plan, plan_status, plan_problems, plan_retry, ... in the order written."""

    placed: SkipValidation[Placed | None] = None
    clarify: Any = None  # compute.resolve.NeedsClarify
    calls: SkipValidation[list[LLMCall]] = Field(default_factory=list)
    meta: SkipValidation[dict[str, Any]] = Field(default_factory=dict)
    route: Literal["placed", "clarify", "failed"]


class EventLookupIn(BoundaryModel):
    """step_event_lookup's input: the event a model's plan named that the calendar lacks."""

    name: str
    called: bool = False


class EventLookupOut(BoundaryModel):
    """step_event_lookup's output: its start (YYYY-MM or YYYY), None when the model does not know or the call failed."""

    start: str | None = None
    call: SkipValidation[LLMCall]
