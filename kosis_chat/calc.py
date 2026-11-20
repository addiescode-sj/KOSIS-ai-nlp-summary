"""The number path alone: a question and its tables -> the numbers code computes (loop engineering, 2026-11-20).

calculate(question, tables, llm=None) runs only the steps that make numbers.

The model is given from outside, as everywhere in Phase 1: llm=LLMConfig(api_key=..., model=..., provider=...) made
by the caller (Phase 3, an evaluation) from its own settings. This module reads no .env and no environment variable.
client= (a ready client: tests, a recording or caching client) wins over llm. With neither, the rules run alone.

One table:
  defaults (judge.rules.defaults: the period or scenario a question leaves open)
  -> the rules' plan (analyze.node.step_plan_rules)
  -> only when the rules leave something open and a model is given: the model fills it (analyze.node.step_plan_fill_llm)
  -> compute (compute.node.step_compute)
  -> the facts the answer uses (compute.answer_facts.step_answer_facts)

Two or three tables:
  the rules' plan over the tables (analyze.multi_rules.step_plan_multi_rules) -> compute(tables=)
  -> the facts the answer uses (the same step)

What it does not run: the judge's refusal and ask-back rules (judge.rules.RULES), the plan the model writes
(plan_write_llm, plan_multi_llm), the fallback chatbot, the sentence (compose, verify, template, render), the chart.
The chart's values are in CalcResult.computed (Computed.series, Computed.ranking) for a caller that draws one.

FIXED FILE. The improvement loop's evaluation calls calculate() and reads CalcResult; a later number tool will wrap
the same function. Improve the steps it calls (the question parser, the rules' plan, the fill prompt, compute, the
multi-table rules, the choice of the answer's facts), not this file: its name, arguments, the order of the steps and
CalcResult's fields stay.

CalcResult.answer is what the evaluation grades: the ids step_answer_facts chose among the computed facts.

origin says how the plan was made, and the evaluation weighs an item by it:
  "rules"        the rules alone built the plan
  "model_codes"  the model filled category codes; the question type, the ranking axis and max/min are the rules'
  "model_type"   the plan's type, ranking axis (over) or max/min (pick) is not what the rules had
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal

from pydantic import ConfigDict, Field, SkipValidation

import kosis_chat.shared.patterns  # noqa: F401  first: puts src/ on the path when kosis_ai is not installed
from kosis_chat.analyze import fill
from kosis_chat.analyze.node import step_plan_fill_llm, step_plan_rules
from kosis_chat.analyze.schemas import Plan, PlanFillIn, PlanRulesIn
from kosis_chat.compute import CalcRefused, compute
from kosis_chat.compute.node import step_compute
from kosis_chat.compute.plan_engine import PlanUnsupported
from kosis_chat.compute.resolve import NeedsClarify
from kosis_chat.compute.schemas import Computed, ComputeIn
from kosis_chat.judge.rules import defaults, parse, resolve, targets
from kosis_chat.llm.calls import ModelAccess
from kosis_chat.llm.model import STATUS_API_ERROR, STATUS_TIMEOUT, LLMClient
from kosis_chat.llm.providers import build_client
from kosis_chat.schemas.resources import LLMConfig, MultiResources, StatResources, load_resources
from kosis_chat.schemas.state import LLMCall
from kosis_chat.shared.boundary import BoundaryModel

Status = Literal["ok", "no_plan", "refused"]
Origin = Literal["rules", "model_codes", "model_type"]

# CalcResult.reason when there is no plan (a refused calculation carries the shared RefusalReason instead)
RULES_UNRESOLVED = "rules_unresolved"  # the rules left something open and no model was given (or: several tables)
FILL_UNPLACED = "fill_unplaced"  # the model's fill did not place the question either
MULTI_AMBIGUOUS = "multi_ambiguous"  # a multi-table plan named something several codes could be
MULTI_UNSUPPORTED = "multi_unsupported"  # a multi-table plan has a block compute cannot run


class CalcResult(BoundaryModel):
    """What calculate() returns.

    status "ok": computed holds the facts and answer the ids of the ones the answer uses (step_answer_facts; an
    evaluation grades these). computed.main is what the plan marked to cite (the facts of plan.show), which is
    where answer starts from. status "no_plan": no plan could be made
    (reason: one of the codes above; detail: what was left open). status "refused": compute refused the plan
    (reason: the RefusalReason, detail: its message)."""

    model_config = ConfigDict(frozen=True)

    status: Status
    origin: Origin | None = None  # how the plan was made; None when there is no plan
    computed: SkipValidation[Computed | None] = None
    answer: SkipValidation[list[str]] = Field(default_factory=list)  # ids of the facts the answer uses, in order
    reason: str | None = None
    detail: SkipValidation[list[str]] = Field(default_factory=list)
    default_applied: SkipValidation[dict[str, str]] = Field(default_factory=dict)
    rule: SkipValidation[Plan | None] = None  # one table: the rules' plan, before any fill
    plan: SkipValidation[Plan | None] = None  # one table: the plan that was computed
    tables: SkipValidation[list[str]] = Field(default_factory=list)  # tbl_ids, in input order
    calls: SkipValidation[list[LLMCall]] = Field(default_factory=list)  # the model calls made (at most 2)
    meta: SkipValidation[dict[str, Any]] = Field(default_factory=dict)  # the fill step's decision entries
    prompt_version: str | None = None  # the fill prompt's version when the model was called

    @property
    def main(self) -> list[str]:
        return list(self.computed.main) if self.computed is not None else []

    @property
    def facts(self) -> list:
        return list(self.computed.result.facts) if self.computed is not None else []

    def record(self) -> dict[str, Any]:
        """A JSON-ready summary: what an evaluation stores per question."""
        plan = self.plan
        return {
            "status": self.status, "origin": self.origin, "reason": self.reason, "detail": list(self.detail),
            "answer": list(self.answer), "main": self.main,
            "facts": [f.model_dump(exclude_none=True) for f in self.facts],
            "default_applied": dict(self.default_applied), "tables": list(self.tables),
            "type": plan.type if plan is not None else None,
            "pattern": self.computed.pattern if self.computed is not None else None,
            "llm_calls": len(self.calls), "llm_status": [c.status for c in self.calls],
            "prompt_version": self.prompt_version,
        }


def origin_of(rule: Plan, plan: Plan) -> Origin:
    """How a plan the model filled differs from the rules' plan: codes only, or what decides the formula too."""
    same = (plan.type, plan.over, plan.pick) == (rule.type, rule.over, rule.pick)
    return "model_codes" if same else "model_type"


def _answer(question: str, computed: Computed, res: StatResources | MultiResources) -> list[str]:
    """The ids step_answer_facts chose, each once and in its order. An id compute did not make is an error of the
    step (ValueError), not an answer."""
    from kosis_chat.compute.answer_facts import AnswerFactsIn, step_answer_facts  # the defender's file

    chosen = list(dict.fromkeys(step_answer_facts(AnswerFactsIn(question=question, computed=computed), res).fact_ids))
    made = {f.id for f in computed.result.facts}
    unknown = [i for i in chosen if i not in made]
    if unknown:
        raise ValueError(f"step_answer_facts returned ids compute did not make: {unknown[:5]}")
    return chosen


def _access(llm: LLMConfig | None, client: LLMClient | None, retry_interval: float) -> ModelAccess | None:
    """How the fill step reaches the model, or None when no model was given. The client is built from llm at the
    first call (LLMKeyMissing then), so a question the rules place needs no key."""
    if llm is None and client is None:
        return None
    built: list[LLMClient] = []

    def get() -> LLMClient:
        if not built:
            built.append(client if client is not None else build_client(llm))
        return built[0]

    def name() -> str:
        return llm.model if llm is not None and get().name == "hcx" else get().name

    return ModelAccess(client=get, model_name=name, retry_interval=retry_interval,
                       max_input_tokens=llm.max_input_tokens if llm is not None else None)


def _one(question: str, res: StatResources, llm: ModelAccess | None) -> CalcResult:
    index = res.index
    parsed = parse(index, question)
    resolved = resolve(index, parsed)
    applied = defaults(parsed, index, resolved, targets(index, parsed, resolved))
    common: dict[str, Any] = {"default_applied": applied, "tables": [res.table.meta.tbl_id]}

    rules = step_plan_rules(PlanRulesIn(question=question, default_applied=applied), res)
    rule = plan = rules.rule
    built, origin = rules.built, "rules"
    if rules.route != "built":
        unresolved = list(rules.unresolved or [])
        if llm is None:
            return CalcResult(status="no_plan", reason=RULES_UNRESOLVED, detail=unresolved, rule=rule, **common)
        filled = step_plan_fill_llm(PlanFillIn(rule=rule, unresolved=unresolved, called=False), res, llm=llm)
        failed = [c.status for c in filled.calls if c.status in (STATUS_TIMEOUT, STATUS_API_ERROR)]
        if failed:  # the model could not be reached: an error of the run, not an answer of the path
            raise RuntimeError(f"the fill's model call failed: {failed[-1]}")
        common |= {"calls": list(filled.calls), "meta": dict(filled.meta), "prompt_version": fill.VERSION}
        if filled.route != "built":
            left = list(filled.meta.get("unresolved_after_fill") or unresolved)
            return CalcResult(status="no_plan", reason=FILL_UNPLACED, detail=left, rule=rule, plan=filled.plan, **common)
        plan, built = filled.plan, filled.built
        origin = origin_of(rule, plan)

    done = step_compute(ComputeIn(plan=plan, built=built, default_applied=applied), res)
    if done.route == "refused":
        return CalcResult(status="refused", origin=origin, reason=done.refusal.reason, detail=[done.refusal.message],
                          rule=rule, plan=plan, **common)
    return CalcResult(status="ok", origin=origin, computed=done.computed,
                      answer=_answer(question, done.computed, res), rule=rule, plan=plan, **common)


def _multi(question: str, res: MultiResources) -> CalcResult:
    from kosis_chat.analyze.multi_rules import PlanMultiRulesIn, step_plan_multi_rules  # the defender's file

    common: dict[str, Any] = {"tables": [r.table.meta.tbl_id for r in res.tables]}
    planned = step_plan_multi_rules(PlanMultiRulesIn(question=question), res)
    if planned.route != "built" or planned.built is None:
        return CalcResult(status="no_plan", reason=RULES_UNRESOLVED, detail=list(planned.unresolved), **common)
    try:
        computed = compute(res.table, planned.built, None, tables=res.table_inputs())
    except CalcRefused as exc:
        return CalcResult(status="refused", origin="rules", reason=exc.reason, detail=[exc.message], **common)
    except NeedsClarify as exc:
        return CalcResult(status="no_plan", reason=MULTI_AMBIGUOUS, detail=[exc.message], **common)
    except PlanUnsupported as exc:
        return CalcResult(status="no_plan", reason=MULTI_UNSUPPORTED, detail=[str(exc)], **common)
    return CalcResult(status="ok", origin="rules", computed=computed, answer=_answer(question, computed, res), **common)


def calculate(question: str, tables: Sequence[StatResources], *, llm: LLMConfig | None = None,
              client: LLMClient | None = None, retry_interval: float = 0.0) -> CalcResult:
    """The numbers for one question over one to three prepared tables (load_tables, or schemas.resources.
    load_resources).

    llm: the model's settings, given by the caller (never read from .env here); client: a ready client, which wins
    over llm; neither: the rules alone. retry_interval: seconds before a second model call for the question.

    Raises RuntimeError when the fill's model call failed (timeout, API error), LLMKeyMissing when the model is
    needed and llm has no key, and ValueError when step_answer_facts names a fact compute did not make. A question
    the path cannot answer is not an error: status "no_plan" or "refused"."""
    tables = list(tables)
    if len(tables) == 1:
        return _one(question, tables[0], _access(llm, client, retry_interval))
    return _multi(question, MultiResources(tuple(tables)))


def load_tables(table_ids: Sequence[str]) -> list[StatResources]:
    """The tables of tables.csv keys, read and prepared as api.summarize_stats_run reads them."""
    from kosis_chat.adapters.parquet import load_table
    from kosis_chat.api import table_path
    from kosis_chat.render.source import org_id_from_table_path

    out = []
    for table_id in table_ids:
        path = table_path(table_id)
        table = load_table(path)
        out.append(load_resources(table, org_id_from_table_path(path, table.meta.tbl_id)))
    return out


__all__ = ["CalcResult", "FILL_UNPLACED", "MULTI_AMBIGUOUS", "MULTI_UNSUPPORTED", "Origin", "RULES_UNRESOLVED", "Status",
           "calculate", "load_tables", "origin_of"]
