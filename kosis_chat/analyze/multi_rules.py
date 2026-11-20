"""The rules' plan over two or three tables: the place for it (loop engineering, 2026-11-20).

kosis_chat.calc.calculate calls step_plan_multi_rules for a question over several tables and computes the plan it
returns with compute(tables=): no model is called on this path. Until rules are written here every such question
comes back as "no plan".

What must stay (kosis_chat/calc.py reads them): the names PlanMultiRulesIn, PlanMultiRulesOut and
step_plan_multi_rules, and their fields. Everything else in this file can change.

To return a plan: build a kosis_ai.schemas.PlanV3 whose select steps name their table by tbl_id (res.by_id,
res.labels), make it a Built (analyze/blocks.py written(), as analyze/multi.py place() does for the model's plan)
and return PlanMultiRulesOut(built=..., route="built"). Each table's index and the one-table rules
(analyze.rules.rule_plan(question, r.index)) are there to read the question against every table.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field, SkipValidation

from kosis_chat.analyze.schemas import Built
from kosis_chat.schemas.resources import MultiResources
from kosis_chat.shared.boundary import BoundaryModel

NO_RULES = "multi_table_rules"  # nothing here plans a question over several tables yet


class PlanMultiRulesIn(BoundaryModel):
    """step_plan_multi_rules's input: the question."""

    question: str


class PlanMultiRulesOut(BoundaryModel):
    """step_plan_multi_rules's output. route "built": built is the plan to compute. route "unresolved": built is
    None and unresolved names what the rules could not decide."""

    built: SkipValidation[Built | None] = None
    unresolved: SkipValidation[list[str]] = Field(default_factory=list)
    route: Literal["built", "unresolved"]


def step_plan_multi_rules(inp: PlanMultiRulesIn, res: MultiResources) -> PlanMultiRulesOut:
    """The rules' plan for a question over res.tables (two or three), or what they could not decide.

    In: PlanMultiRulesIn(question)
    Out: PlanMultiRulesOut(built, unresolved, route "built" | "unresolved")
    Side effects: none (reads every table's index)
    """
    return PlanMultiRulesOut(unresolved=[NO_RULES], route="unresolved")
