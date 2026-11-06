"""The stat graph (T1-28 S4): the stage steps as LangGraph nodes, assembled from a node table (E4).

    START -> judge -(final)-> answer_final
                   -(plan)-> plan_rules -(built)-> compute
                                        -(unresolved)-> plan_fill_llm -(built)-> compute
                                                                      -(unplaced)-> plan_write_llm
             plan_write_llm -(placed)-> compute  -(clarify)-> answer_final  -(failed)-> fallback_chatbot -> END
             compute -(ok)-> caveats  -(refused)-> answer_final -> END
             caveats -> compose_llm -(verify)-> verify -(ok)-> compose_done
                                                       -(retry)-> compose_llm   (the one loop, D3: COMPOSE_MAX_ATTEMPTS)
                                                       -(template)-> template
                        compose_llm -(template)-> template -> compose_done
             caveats -> chart                     (beside the compose path)
             [compose_done, chart] -> render -> END   (render waits for both)

Seven conditional edges (judge, plan_rules, plan_fill_llm, plan_write_llm, compute, compose_llm, verify). The model's
own plan (plan_write_llm) is computed by code inside that step (D12 a: analyze's place() calls compute); its compute
node only adds the query (step_compute with `computed` set skips the calculation), as StatPipeline.ask did.

langgraph is imported only in this package (tests/test_layout.py).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from langgraph.graph import END, START, StateGraph

from kosis_chat.analyze.node import step_plan_fill_llm, step_plan_rules, step_plan_write_llm
from kosis_chat.chart.node import step_chart
from kosis_chat.compose.node import step_caveats, step_compose_llm, step_template, step_verify
from kosis_chat.compose.schemas import COMPOSE_MAX_ATTEMPTS
from kosis_chat.compute.node import step_compute
from kosis_chat.graph import adapters
from kosis_chat.graph.adapters import StatContext
from kosis_chat.judge.node import step_judge
from kosis_chat.graph.legacy import step_fallback_chatbot  # legacy/fallback.py, through graph/legacy.py
from kosis_chat.render.node import step_answer_final, step_render
from kosis_chat.schemas.resources import StatResources
from kosis_chat.schemas.state import StatState

# Node name -> step function (E4). node_overrides replaces an entry with a function of the same signature
# (step(inp, res, **keywords) -> Out: the same In / Out models, the keywords the adapter passes).
NODE_TABLE: dict[str, Callable[..., Any]] = {
    "judge": step_judge,
    "plan_rules": step_plan_rules,
    "plan_fill_llm": step_plan_fill_llm,
    "plan_write_llm": step_plan_write_llm,
    "fallback_chatbot": step_fallback_chatbot,
    "compute": step_compute,
    "caveats": step_caveats,
    "compose_llm": step_compose_llm,
    "verify": step_verify,
    "template": step_template,
    "chart": step_chart,
    "render": step_render,
    "answer_final": step_answer_final,
}
_ADAPTERS = {
    "judge": adapters.judge_node, "plan_rules": adapters.plan_rules_node, "plan_fill_llm": adapters.plan_fill_llm_node,
    "plan_write_llm": adapters.plan_write_llm_node, "fallback_chatbot": adapters.fallback_node,
    "compute": adapters.compute_node, "caveats": adapters.caveats_node, "compose_llm": adapters.compose_llm_node,
    "verify": adapters.verify_node, "template": adapters.template_node, "chart": adapters.chart_node,
    "render": adapters.render_node, "answer_final": adapters.answer_final_node,
}
# The longest path is 9 + 2 * COMPOSE_MAX_ATTEMPTS supersteps (judge plan_rules plan_fill_llm plan_write_llm compute
# caveats, compose_llm + verify per attempt, template, compose_done, render). The limit is twice that: never reached
# by the graph as built (the verify edge caps the loop); a node_overrides loop that never ends stops here.
LONGEST_PATH = 9 + 2 * COMPOSE_MAX_ATTEMPTS
RECURSION_LIMIT = 2 * LONGEST_PATH


def _route(state: StatState) -> str:
    return state.step_route


def build_graph(resources: StatResources, node_overrides: Mapping[str, Callable[..., Any]] | None = None):
    """The compiled graph for one table's resources (the adapters close over them; one graph per StatPipeline).

    node_overrides: node name -> step function to use in place of NODE_TABLE's (E4); an unknown name raises KeyError.
    Run it with run(graph, ...)."""
    overrides = dict(node_overrides or {})
    unknown = sorted(set(overrides) - set(NODE_TABLE))
    if unknown:
        raise KeyError(f"unknown node(s): {', '.join(unknown)} (known: {', '.join(NODE_TABLE)})")
    steps = NODE_TABLE | overrides
    g = StateGraph(StatState, context_schema=StatContext)
    for name, make in _ADAPTERS.items():
        g.add_node(name, make(steps[name], resources))
    g.add_node("compose_done", adapters.compose_done_node(resources))

    g.add_edge(START, "judge")
    g.add_conditional_edges("judge", _route, {"final": "answer_final", "plan": "plan_rules"})
    g.add_conditional_edges("plan_rules", _route, {"built": "compute", "unresolved": "plan_fill_llm"})
    g.add_conditional_edges("plan_fill_llm", _route, {"built": "compute", "unplaced": "plan_write_llm"})
    g.add_conditional_edges("plan_write_llm", _route,
                            {"placed": "compute", "clarify": "answer_final", "failed": "fallback_chatbot"})
    g.add_edge("fallback_chatbot", END)
    g.add_conditional_edges("compute", _route, {"ok": "caveats", "refused": "answer_final"})
    g.add_edge("caveats", "compose_llm")
    g.add_edge("caveats", "chart")  # the chart branch, beside the compose path
    g.add_conditional_edges("compose_llm", _route, {"verify": "verify", "template": "template"})
    g.add_conditional_edges("verify", _route, {"ok": "compose_done", "retry": "compose_llm", "template": "template"})
    g.add_edge("template", "compose_done")
    g.add_edge(["compose_done", "chart"], "render")  # the join: render runs once both have finished
    g.add_edge("render", END)
    g.add_edge("answer_final", END)
    return g.compile()


def run(graph, question: str, *, context: StatContext, meta: dict[str, Any], started: float,
        golden_scope: dict | None = None) -> StatState:
    """One question through the graph: the final StatState (records, meta, stat or fallback, chart).

    meta: the decision_metadata the answer starts with; started: time.perf_counter() at the question's start."""
    out = graph.invoke({"question": question, "golden_scope": golden_scope, "started": started, "meta": meta},
                       {"recursion_limit": RECURSION_LIMIT}, context=context)
    return out if isinstance(out, StatState) else StatState(**out)


__all__ = ["LONGEST_PATH", "NODE_TABLE", "RECURSION_LIMIT", "StatContext", "build_graph", "run"]
