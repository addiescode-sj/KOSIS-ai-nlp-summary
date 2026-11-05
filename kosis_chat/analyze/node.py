"""analyze nodes (T1-28 S3a, S3b): the rules' plan, the model's fill of what the rules left open, the model's own plan
and the event lookup that plan may need.

Steps: (XxxIn, StatResources) -> XxxOut. step_plan_rules is pure. The model steps (step_plan_fill_llm,
step_plan_write_llm, step_event_lookup) call the model through a ModelAccess (keyword llm=, kosis_chat/llm/calls.py)
and return the calls they made as LLMCall records and the decision_metadata entries they wrote; StatPipeline.ask
adds both to the question's record in the order they come.
"""

from __future__ import annotations

from kosis_chat.analyze import event_lookup, fill, write
from kosis_chat.analyze.rules import Plan, Unresolved, build_plan, rule_plan
from kosis_chat.analyze.schemas import (
    EventLookupIn,
    EventLookupOut,
    PlanFillIn,
    PlanFillOut,
    PlanRulesIn,
    PlanRulesOut,
    PlanWriteIn,
    PlanWriteOut,
)
from kosis_chat.llm.calls import ModelAccess
from kosis_chat.llm.model import STATUS_OK
from kosis_chat.schemas.resources import StatResources
from kosis_chat.schemas.state import LLMCall, called_before

from kosis_ai.schemas import PlanV3  # isort: skip


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


# ---------------------------------------------------------------- plan_fill_llm (analyze-fill-v2)


def _fill(plan: Plan, user: str, key: str, res: StatResources, llm: ModelAccess, called: bool, calls: list[LLMCall],
          meta: dict):
    """(plan filled by the model, problems to say back, the type the model named), or (None, [], None)."""
    stage = "analyze.llm_fill" if key == "fill" else "analyze.llm_fill_retry"
    result, call = llm.ask(stage, fill.SYSTEM_PROMPT, user, fill.AnalyzeFill, called=called or called_before(calls))
    calls.append(call)
    if result.status != STATUS_OK:
        meta[f"{key}_status"] = result.status
        return None, [], None
    filled, ignored, problems = fill.apply(plan, result.value, res.index)
    meta[key] = result.value.model_dump(exclude_defaults=True) | ({"ignored": ignored} if ignored else {})
    return filled, problems, result.value.type


def _build(filled: Plan, kind, res: StatResources, meta: dict, build):
    """(plan, built) for the filled plan, or for it with the type the model named; None when neither builds."""
    tries = [filled]
    if kind and kind != filled.type and not filled.pattern:  # a period pattern fixes its own type (value/trend/share)
        tries.append(filled.model_copy(update={"type": kind}))
    for candidate in tries:
        try:
            return candidate, build(candidate, res.index, res.table)
        except Unresolved as exc:
            meta["unresolved_after_fill"] = exc.what
    return None


def step_plan_fill_llm(inp: PlanFillIn, res: StatResources, *, llm: ModelAccess, build=build_plan) -> PlanFillOut:
    """The model fills what the rules left open (analyze-fill-v2); a correctable fault is said back once.

    In: PlanFillIn(rule, unresolved, called)
    Out: PlanFillOut(plan, built, calls, meta, route "built" | "unplaced")
    Side effects: model calls through llm (at most 2: analyze.llm_fill, analyze.llm_fill_retry); none else
    Node: plan_fill_llm (spans "analyze.llm_fill", "analyze.llm_fill_retry")

    build: the block builder, as for step_plan_rules (StatPipeline passes its module's build_plan).
    """
    calls: list[LLMCall] = []
    meta: dict = {}
    plan = inp.rule.model_copy(update={"unresolved": inp.rule.unresolved or inp.unresolved})
    user = fill.user_prompt(plan.question, res.index, plan)
    meta["route"] = "llm_fill"

    def out(plan: Plan, built=None) -> PlanFillOut:
        return PlanFillOut(plan=plan, built=built, calls=calls, meta=meta, route="built" if built is not None else "unplaced")

    filled, problems, kind = _fill(plan, user, "fill", res, llm, inp.called, calls, meta)
    if filled is None:
        return out(plan)
    if (placed := _build(filled, kind, res, meta, build)) is not None:
        return out(*placed)
    if problems:  # the model's answer had a correctable fault: say it back once (analyze-fill-v2)
        again, _, kind = _fill(plan, fill.retry_prompt(user, problems), "fill_retry", res, llm, inp.called, calls, meta)
        if again is not None:
            if (placed := _build(again, kind, res, meta, build)) is not None:
                return out(*placed)
            filled = again
    return out(filled)


# ---------------------------------------------------------------- plan_write_llm (analyze-plan-v1, event-lookup-v1)


def step_event_lookup(inp: EventLookupIn, res: StatResources, *, llm: ModelAccess) -> EventLookupOut:
    """When an event the calendar lacks began, asked of the model (event-lookup-v1).

    In: EventLookupIn(name, called)
    Out: EventLookupOut(start (None when unknown or the call failed), call)
    Side effects: one model call through llm
    Node: event_lookup, called back from inside plan_write_llm (span "event_lookup")
    """
    result, call = llm.ask("event_lookup", event_lookup.SYSTEM_PROMPT, event_lookup.user_prompt(inp.name),
                           event_lookup.EventStart, called=inp.called, event=inp.name)
    return EventLookupOut(start=result.value.start if result.status == STATUS_OK else None, call=call)


def step_plan_write_llm(inp: PlanWriteIn, res: StatResources, *, llm: ModelAccess) -> PlanWriteOut:
    """The model's own plan as v3 blocks (analyze-plan-v1), computed by code and said back once when it fails.

    In: PlanWriteIn(question, rule, called)
    Out: PlanWriteOut(placed, clarify, calls, meta, route "placed" | "clarify" | "failed")
    Side effects: model calls through llm (at most 2 plans: analyze.llm_plan, analyze.llm_plan_retry, and one
      event_lookup per event a plan names that the calendar lacks, asked once per question, made while the plan is
      computed); computes the plan (compute stage, D12 a)
    Node: plan_write_llm (spans "analyze.llm_plan", "event_lookup", "analyze.llm_plan_retry")
    """
    calls: list[LLMCall] = []
    meta: dict = {}
    known: dict[str, str | None] = meta.setdefault("event_lookup", {})

    def lookup(name: str) -> str | None:
        if name not in known:
            looked = step_event_lookup(EventLookupIn(name=name, called=inp.called or called_before(calls)), res, llm=llm)
            calls.append(looked.call)
            known[name] = looked.start
        return known[name]

    def out(route, placed=None, clarify=None) -> PlanWriteOut:
        return PlanWriteOut(placed=placed, clarify=clarify, calls=calls, meta=meta, route=route)

    user = write.user_prompt(inp.question, res.index, inp.rule)
    meta["route"] = "llm_plan"
    clarify = None
    for key in ("plan", "plan_retry"):
        result, call = llm.ask(f"analyze.llm_{key}", write.SYSTEM_PROMPT, user, PlanV3,
                               called=inp.called or called_before(calls))
        calls.append(call)
        if result.status != STATUS_OK:
            meta[f"{key}_status"] = result.status
            return out("failed")
        meta[key] = result.value.model_dump(exclude_defaults=True)
        placed = write.place(inp.question, result.value, res.table, res.index, lookup)
        if not placed.problems:
            return out("placed", placed=placed)
        meta[f"{key}_problems"] = placed.problems
        clarify = placed.clarify
        user = write.retry_prompt(user, placed.problems)
    # what several codes could be is asked back; an event nobody knows, or a plan that is still wrong, is not
    if clarify is not None and len(clarify.options) >= 2:
        return out("clarify", clarify=clarify)
    return out("failed")
