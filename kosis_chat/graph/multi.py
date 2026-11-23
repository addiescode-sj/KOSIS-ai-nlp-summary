"""The multi-table graph (T1-30): a question over two or three tables, beside the one-table graph (graph/build.py).

    START -> judge -(final)-> answer_final -> END
                   -(plan)-> plan_multi_llm -(placed)-> compute -> caveats -> compose_llm <-> verify -> compose_done
                                            -(clarify | refused | failed)-> answer_final         -> template -> compose_done
                                                                          caveats -> chart
             [compose_done, chart] -> render -> END

The one-table graph is not changed: a question with one table never reaches this module. judge runs only the
question-level rules (judge/rules.py judge_multi); the plan is the model's (analyze/multi.py, no rules' plan, no
compact-v2 fallback); compose_llm, verify, template and chart are the one-table nodes as they are (they read the
computed facts only). The answer names every table: StatAnswer.sources (source = the first) and one '출처:' line.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

from langgraph.graph import END, START, StateGraph

from kosis_chat.analyze.node import step_plan_multi_llm
from kosis_chat.analyze.schemas import Plan, PlanMultiIn
from kosis_chat.chart.node import step_chart
from kosis_chat.compose.caveats import caveats_of
from kosis_chat.compose.node import step_compose_llm, step_template, step_verify
from kosis_chat.compose.schemas import COMPOSE_MAX_ATTEMPTS
from kosis_chat.graph import adapters
from kosis_chat.graph.adapters import StatContext, _ctx
from kosis_chat.graph.spans import step_span
from kosis_chat.judge.node import step_judge_multi
from kosis_chat.prepare.schemas import Cell
from kosis_chat.prepare.store import store_of
from kosis_chat.render.node import step_answer_final, step_render
from kosis_chat.render.schemas import AnswerFinalIn, RenderIn
from kosis_chat.render.source import to_source
from kosis_chat.render.text import final_text_multi
from kosis_chat.schemas.resources import MultiResources
from kosis_chat.schemas.state import StatState, called_before

from kosis_ai.schemas import Refusal, StatAnswer, StatQuery  # isort: skip

# The longest path: judge plan_multi_llm compute caveats, compose_llm + verify per attempt, template, compose_done, render.
LONGEST_PATH = 8 + 2 * COMPOSE_MAX_ATTEMPTS
RECURSION_LIMIT = 2 * LONGEST_PATH
FAILED_MESSAGE = "주어진 표들을 함께 읽어 이 질문을 계산하는 방법을 정하지 못했습니다."


def sources_of(res: MultiResources) -> list:
    return [to_source(r.table.meta, org_id=r.org_id) for r in res.tables]


def with_sources(stat: StatAnswer, res: MultiResources) -> StatAnswer:
    """The answer naming every table: sources, and the '출처:' line the one-table render wrote replaced by one for all."""
    if stat.source is None:  # out of scope: no table
        return stat
    sources = sources_of(res)
    body = stat.text.rsplit("\n\n출처: ", 1)[0]
    return StatAnswer.model_validate(stat.model_dump() | {"sources": [s.model_dump() for s in sources],
                                                          "text": final_text_multi(body, sources)})


def _route(state: StatState) -> str:
    return state.step_route


def plan_multi_node(res: MultiResources) -> Callable:
    def plan_multi_llm(state: StatState, runtime) -> dict:
        written = step_plan_multi_llm(PlanMultiIn(question=state.question, called=called_before(state.records)), res,
                                      llm=_ctx(runtime).llm)
        out = {"records": list(written.calls), "meta": written.meta, "step_route": written.route}
        if written.route == "placed":
            placed = written.placed
            kind = None if placed.plan.present == "table" else placed.plan.present
            out |= {"plan": Plan(question=state.question, type=kind, codes={}, periods=[]), "built": placed.built,
                    "computed": placed.computed}
        elif written.route == "clarify":
            out |= {"clarify": written.clarify, "meta": written.meta | {"route": "llm_plan_multi+clarify"}}
        elif written.route == "refused":
            exc = written.refusal
            out |= {"refusal": Refusal(reason=exc.reason, message=exc.message),
                    "meta": written.meta | {"route": "llm_plan_multi+calc_refused"}}
        else:
            problems = written.meta.get("plan_retry_problems") or written.meta.get("plan_problems") or []
            out |= {"refusal": Refusal(reason="class_mismatch", message=FAILED_MESSAGE, available="; ".join(problems) or None),
                    "meta": written.meta | {"route": "llm_plan_multi+failed"}, "step_route": "refused"}
        return out
    return plan_multi_llm


def compute_node(res: MultiResources) -> Callable:
    """The plan was computed by plan_multi_llm (as plan_write_llm does): the query and the source are added here."""
    def compute(state: StatState) -> dict:
        started_meta = {"type": state.plan.type, "pattern": state.built.pattern,
                        "plan": state.built.plan.model_dump(mode="json", exclude_none=True)}
        computed = state.computed
        periods = sorted({f.period for f in computed.result.facts if f.source_ref and f.period})
        query = StatQuery(question=state.question, type=state.plan.type, periods=periods, plan=state.built.plan)
        span = step_span("compute", state.started, facts=len(computed.result.facts), tables=len(res.tables))
        return {"source": to_source(res.table.meta, org_id=res.org_id), "query": query, "records": [span],
                "meta": started_meta, "step_route": "ok"}
    return compute


class _Cells:
    """Every table's cells seen as one, for caveats_of, which only looks up the cited cells by source_ref: each table's
    own store is asked once, and on a ref two tables share the later table's cell wins."""

    def __init__(self, res: MultiResources) -> None:
        self.stores = [store_of(r.table) for r in res.tables]

    def by_refs(self, refs: Iterable[str]) -> dict[str, Cell]:
        refs = list(refs)
        found: dict[str, Cell] = {}
        for store in self.stores:
            found.update(store.by_refs(refs))
        return found


def caveats_node(res: MultiResources) -> Callable:
    def caveats(state: StatState) -> dict:
        found = caveats_of(state.computed, _Cells(res), [], [])
        shown = [c.text for c in found]
        return {"caveats_out": found, "shown": shown, "context_ids": []}
    return caveats


def render_node(res: MultiResources) -> Callable:
    def render(state: StatState) -> dict:
        chart = state.chart_out
        stat = step_render(RenderIn(question=state.question, query=state.query, computed=state.computed, parts=state.parts,
                                    verified=state.verified, source=state.source, default_applied={},
                                    caveats=state.caveats_out, shown=state.shown, context_ids=state.context_ids), res).stat
        return {"stat": with_sources(stat, res),
                "meta": {"chart": chart.kind if chart else None, "caveats": [c.kind for c in state.caveats_out]}}
    return render


def answer_final_node(res: MultiResources) -> Callable:
    def answer_final(state: StatState) -> dict:
        if state.step_route == "refused":
            source = to_source(res.table.meta, org_id=res.org_id)
            inp = AnswerFinalIn(kind="refused", question=state.question, refusal=state.refusal,
                                query=StatQuery(question=state.question), source=source)
            return {"stat": with_sources(step_answer_final(inp, res).stat, res)}
        if state.step_route == "clarify":
            inp = AnswerFinalIn(kind="clarify", question=state.question, clarify=state.clarify)
            return {"stat": with_sources(step_answer_final(inp, res).stat, res)}
        stat = step_answer_final(AnswerFinalIn(kind="verdict", question=state.question, verdict=state.verdict), res).stat
        return {"stat": with_sources(stat, res), "meta": {"route": "rules_final"}}
    return answer_final


def build_multi_graph(res: MultiResources):
    """The compiled graph for one set of two or three tables (one graph per MultiStatPipeline)."""
    g = StateGraph(StatState, context_schema=StatContext)
    g.add_node("judge", adapters.judge_node(step_judge_multi, res))
    g.add_node("plan_multi_llm", plan_multi_node(res))
    g.add_node("compute", compute_node(res))
    g.add_node("caveats", caveats_node(res))
    g.add_node("compose_llm", adapters.compose_llm_node(step_compose_llm, res))
    g.add_node("verify", adapters.verify_node(step_verify, res))
    g.add_node("template", adapters.template_node(step_template, res))
    g.add_node("compose_done", adapters.compose_done_node(res))
    g.add_node("chart", adapters.chart_node(step_chart, res))
    g.add_node("render", render_node(res))
    g.add_node("answer_final", answer_final_node(res))

    g.add_edge(START, "judge")
    g.add_conditional_edges("judge", _route, {"final": "answer_final", "plan": "plan_multi_llm"})
    g.add_conditional_edges("plan_multi_llm", _route,
                            {"placed": "compute", "clarify": "answer_final", "refused": "answer_final"})
    g.add_edge("compute", "caveats")
    g.add_edge("caveats", "compose_llm")
    g.add_edge("caveats", "chart")
    g.add_conditional_edges("compose_llm", _route, {"verify": "verify", "template": "template"})
    g.add_conditional_edges("verify", _route, {"ok": "compose_done", "retry": "compose_llm", "template": "template"})
    g.add_edge("template", "compose_done")
    g.add_edge(["compose_done", "chart"], "render")
    g.add_edge("render", END)
    g.add_edge("answer_final", END)
    return g.compile()


def run_multi(graph, question: str, *, context: StatContext, meta: dict[str, Any], started: float) -> StatState:
    out = graph.invoke({"question": question, "started": started, "meta": meta}, {"recursion_limit": RECURSION_LIMIT},
                       context=context)
    return out if isinstance(out, StatState) else StatState(**out)


__all__ = ["FAILED_MESSAGE", "LONGEST_PATH", "RECURSION_LIMIT", "build_multi_graph", "run_multi", "sources_of",
           "with_sources"]
