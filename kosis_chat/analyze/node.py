"""analyze nodes (T1-28 S3a): the rules' plan. The model's fill and plan (plan_fill_llm, plan_write_llm) stay in
kosis_chat/pipeline.py until S3b.

A pure step: (PlanRulesIn, StatResources) -> PlanRulesOut. StatPipeline.ask calls it and records "analyze.rules".
"""

from __future__ import annotations

from kosis_chat.analyze.rules import Unresolved, build_plan, rule_plan
from kosis_chat.analyze.schemas import PlanRulesIn, PlanRulesOut
from kosis_chat.schemas.resources import StatResources


def step_plan_rules(inp: PlanRulesIn, res: StatResources, *, build=build_plan) -> PlanRulesOut:
    """The rules' plan for the question (rule_plan) and its v3 blocks (build_plan) when the rules can place it.

    In: PlanRulesIn(question, default_applied)
    Out: PlanRulesOut(rule, built, unresolved, route "built" | "unresolved")
    Side effects: none (reads resources.table and resources.index; fills the table's lookup caches)
    Node: plan_rules (span "analyze.rules")

    build: the block builder, build_plan unless a caller swaps it (StatPipeline passes its module's build_plan, which
    tests replace to force the model's route; the graph's node table will do the same, E4).
    """
    rule = rule_plan(inp.question, res.index, inp.default_applied)
    try:
        return PlanRulesOut(rule=rule, built=build(rule, res.index, res.table), route="built")
    except Unresolved as exc:
        return PlanRulesOut(rule=rule, unresolved=exc.what, route="unresolved")
