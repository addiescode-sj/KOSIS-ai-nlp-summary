"""compute node (T1-28 S3a): the numbers, by code, from the built plan.

A pure step: (ComputeIn, StatResources) -> ComputeOut. StatPipeline.ask calls it and records the "compute" span.
"""

from __future__ import annotations

from kosis_chat.analyze.rules import to_query
from kosis_chat.compute.context import scenario_alternatives, with_alternatives
from kosis_chat.compute.engine import CalcRefused, compute
from kosis_chat.compute.schemas import ComputeIn, ComputeOut
from kosis_chat.schemas.resources import StatResources

from kosis_ai.schemas import Refusal  # isort: skip


def step_compute(inp: ComputeIn, res: StatResources) -> ComputeOut:
    """The query slots (to_query), the other scenarios shown beside the answer, and the computed facts.

    In: ComputeIn(plan, built, default_applied, computed: already computed by the model's plan, else None)
    Out: ComputeOut(query, built with the alternatives as context, alternatives, computed | refusal, route "ok" | "refused")
    Side effects: none (reads resources.table and resources.index; fills the table's lookup caches)
    Node: compute (span "compute")
    """
    query = to_query(inp.plan, res.index, inp.built)
    alternatives = [] if inp.computed else scenario_alternatives(res.table, res.index, inp.built, inp.default_applied)
    built = with_alternatives(inp.built, alternatives)
    try:
        computed = inp.computed or compute(res.table, built)
    except CalcRefused as exc:  # a refusal is an answer, not a failure
        return ComputeOut(query=query, built=built, alternatives=alternatives,
                          refusal=Refusal(reason=exc.reason, message=exc.message), route="refused")
    return ComputeOut(query=query, built=built, alternatives=alternatives, computed=computed, route="ok")
