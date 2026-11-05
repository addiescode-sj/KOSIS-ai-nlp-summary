"""T1-28 S4: StatPipeline.ask runs a LangGraph graph (kosis_chat/graph/build.py); the Answers are the S3b router's.

The reference is the manual router StatPipeline.ask was until S4, kept here as `sequential_ask` (copied from the S3b
pipeline.py: the same steps, in the same order, with the same records and decision_metadata writes). Each case runs a
fresh pipeline twice with the same scripted MockClient, once through the graph and once through the reference, and
compares the whole Answer (volatile times aside), the decision_metadata key order, the spans (names, status,
attributes, order), the prompts sent to the model in order, and the time.sleep calls (retry_interval decisions).
Plus: the compose <-> verify loop cap, node_overrides (E4), a chart branch that ends late, the reducers.
"""

from __future__ import annotations

import hashlib
import json
import re
import threading
import time
from dataclasses import replace
from typing import Any

import pytest
from langgraph.channels.binop import BinaryOperatorAggregate
from langgraph.graph import END, START, StateGraph

from kosis_chat import pipeline as pipeline_module
from kosis_chat.analyze.node import step_plan_fill_llm, step_plan_rules, step_plan_write_llm
from kosis_chat.analyze.rules import Unresolved
from kosis_chat.analyze.schemas import PlanFillIn, PlanRulesIn, PlanWriteIn
from kosis_chat.chart.node import step_chart
from kosis_chat.chart.schemas import ChartIn
from kosis_chat.compose.node import step_caveats, step_compose_llm, step_template, step_verify
from kosis_chat.compose.schemas import COMPOSE_MAX_ATTEMPTS, CaveatsIn, ComposeLLMIn, TemplateIn, VerifyIn, VerifyOut
from kosis_chat.compose.verify import label_digits
from kosis_chat.compute.node import step_compute
from kosis_chat.compute.schemas import ComputeIn
from kosis_chat.data.loader import load_table
from kosis_chat.graph import build
from kosis_chat.graph.build import NODE_TABLE, RECURSION_LIMIT, build_graph
from kosis_chat.graph.legacy import fallback_answer, to_answer
from kosis_chat.graph.spans import step_span
from kosis_chat.judge.node import step_judge
from kosis_chat.judge.schemas import JudgeIn
from kosis_chat.legacy.fallback import FALLBACK_PROMPT, FallbackIn, step_fallback_chatbot
from kosis_chat.llm.model import LLMResult, MockClient
from kosis_chat.pipeline import StatPipeline
from kosis_chat.render.node import step_answer_final, step_render
from kosis_chat.render.schemas import AnswerFinalIn, RenderIn
from kosis_chat.render.source import to_source
from kosis_chat.schemas.state import StatState, StepSpan, append, called_before, merge

REAL_SLEEP = time.sleep  # before any test replaces time.sleep


# ---------------------------------------------------------------- the reference: the S3b manual router


def _compose_seq(bot, question, computed, cautions, context_ids, records, meta, llm):
    res = bot.resources
    prompt = None
    problems: list[str] = []
    for attempt in range(COMPOSE_MAX_ATTEMPTS):
        written = step_compose_llm(ComposeLLMIn(question=question, computed=computed, cautions=cautions,
                                                context_ids=context_ids, attempt=attempt, problems=problems,
                                                prompt=prompt, called=called_before(records)), res, llm=llm)
        records.append(written.call)
        meta.update(written.meta)
        prompt = written.prompt
        if written.route == "template":
            break
        started = time.perf_counter()
        checked = step_verify(VerifyIn(parts=written.parts, computed=computed, attempt=attempt), res)
        problems = checked.problems
        records.append(step_span("verify", started, "error" if problems else "ok", attempt=attempt,
                                 problems=len(problems)))
        meta.setdefault("verify_problems", []).append(problems)
        if checked.route == "ok":
            meta["compose"] = "model" if attempt == 0 else "model_retry"
            return written.parts, True
        if checked.route == "template":
            break
    started = time.perf_counter()
    meta["compose"] = "template"
    templated = step_template(TemplateIn(type=meta["type"], computed=computed), res)
    if templated.problems:
        meta["template_problems"] = templated.problems
    records.append(step_span("template", started, "error" if templated.problems else "ok"))
    return templated.parts, templated.verified


def sequential_ask(bot: StatPipeline, question: str, golden_scope: dict | None = None):
    """StatPipeline.ask as of T1-28 S3b (the manual router), for comparison only."""
    res = bot.resources
    llm = bot._model_access()
    started = time.perf_counter()
    records: list = []

    def done(stat, verdict, meta):
        return to_answer(stat, verdict, meta, records, started, table_meta=bot.table.meta, prompt_version=bot.prompt_version)

    judged = step_judge(JudgeIn(question=question), res)
    verdict = judged.verdict
    records.append(step_span("judge", started, outcome=verdict.outcome, rule=verdict.rule, final=verdict.final))
    meta: dict[str, Any] = {**bot.decision_metadata, "route": "rules"}
    if judged.route == "final":
        stat = step_answer_final(AnswerFinalIn(kind="verdict", question=question, verdict=verdict), res).stat
        return done(stat, verdict, meta | {"route": "rules_final"})
    rules_started = time.perf_counter()
    rules = step_plan_rules(PlanRulesIn(question=question, default_applied=verdict.default_applied), res,
                            build=pipeline_module.build_plan)
    rule = plan = rules.rule
    built = rules.built
    if rules.route == "built":
        records.append(step_span("analyze.rules", rules_started, type=plan.type))
    else:
        meta["unresolved"] = rules.unresolved
        records.append(step_span("analyze.rules", rules_started, "error", type=plan.type, unresolved=rules.unresolved))
        filled = step_plan_fill_llm(PlanFillIn(rule=rule, unresolved=rules.unresolved, called=called_before(records)),
                                    res, llm=llm, build=pipeline_module.build_plan)
        records += filled.calls
        meta.update(filled.meta)
        plan, built = filled.plan, filled.built
    computed = None
    if built is None:
        written = step_plan_write_llm(PlanWriteIn(question=question, rule=rule, called=called_before(records)), res, llm=llm)
        records += written.calls
        meta.update(written.meta)
        if written.route == "clarify":
            stat = step_answer_final(AnswerFinalIn(kind="clarify", question=question, clarify=written.clarify), res).stat
            return done(stat, verdict, meta | {"route": "llm_plan+clarify"})
        if written.route == "placed":
            plan, built, computed = written.placed.query_plan, written.placed.built, written.placed.computed
    if built is None:
        fell = step_fallback_chatbot(FallbackIn(question=question, golden_scope=golden_scope), res,
                                     chatbot=lambda: bot.fallback, llm=llm)
        records.append(fell.call)
        return fallback_answer(fell.answer, meta, records, started, fallback_prompt=FALLBACK_PROMPT)
    source = to_source(bot.table.meta, org_id=bot.org_id)
    meta["type"] = plan.type
    meta["pattern"] = built.pattern
    meta["plan"] = built.plan.model_dump(mode="json", exclude_none=True)
    compute_started = time.perf_counter()
    out = step_compute(ComputeIn(plan=plan, built=built, default_applied=verdict.default_applied, computed=computed), res)
    query, built = out.query, out.built
    if out.route == "refused":
        records.append(step_span("compute", compute_started, outcome="refused", reason=out.refusal.reason))
        stat = step_answer_final(AnswerFinalIn(kind="refused", question=question, refusal=out.refusal, query=query,
                                               source=source), res).stat
        return done(stat, verdict, meta | {"route": f"{meta['route']}+calc_refused"})
    computed = out.computed
    records.append(step_span("compute", compute_started, facts=len(computed.result.facts)))
    noted = step_caveats(CaveatsIn(computed=computed, built=built, alternatives=out.alternatives,
                                   default_applied=verdict.default_applied), res)
    caveats, shown, context_ids = noted.caveats, noted.shown, noted.context_ids
    parts, verified = _compose_seq(bot, question, computed, shown, context_ids, records, meta, llm)
    meta["label_digits"] = label_digits(parts, computed)
    meta["numbers_verified"] = verified
    chart = step_chart(ChartIn(type=plan.type, computed=computed), res).chart
    stat = step_render(RenderIn(question=question, query=query, computed=computed, parts=parts, verified=verified,
                                source=source, default_applied=verdict.default_applied, caveats=caveats, shown=shown,
                                context_ids=context_ids, chart=chart), res).stat
    meta["chart"] = chart.kind if chart else None
    meta["caveats"] = [c.kind for c in caveats]
    return replace(done(stat, verdict, meta), chart=chart.spec if chart else None)


# ---------------------------------------------------------------- a scripted model (by a hash of the prompt)


def _h(text: str) -> int:
    return int(hashlib.md5(text.encode("utf-8")).hexdigest(), 16)


def _menu(user: str, header: str) -> dict[str, list[str]]:
    if header not in user:
        return {}
    part = user.split(header, 1)[1].split("\n\n", 1)[0]
    axes = {}
    for line in part.splitlines():
        m = re.match(r"- ([^:]+): (.*)", line)
        if not m or line.startswith("- 시점"):
            continue
        codes = [c.split("=", 1)[0] for c in m.group(2).split(", ") if "=" in c]
        if codes:
            axes[m.group(1)] = codes
    return axes


def scripted(salt: str):
    """A model whose reply depends on the prompt (and salt): right, wrong, correctable and broken replies per step."""

    def reply(system: str, user: str) -> LLMResult:
        k = _h(salt + system + user)
        usage = {"promptTokens": len(user) % 997, "completionTokens": k % 50}
        if system.startswith("당신은 통계 질문을 통계표의 칸으로"):  # analyze-fill-v2
            axes = _menu(user, "[채울 분류 목록]\n")
            if not axes or k % 4 == 0:
                text = "not json"
            else:
                name, codes = next(iter(axes.items()))
                text = json.dumps([{"type": "value", "codes": {name: codes[:1]}},
                                   {"codes": {name: ["없는코드"]}},
                                   {"type": "compare", "codes": {name: codes[:2]}}][k % 3], ensure_ascii=False)
        elif system.startswith("당신은 통계 질문을 계산 순서"):  # analyze-plan-v1
            axes = _menu(user, "(항목은 item, 나머지는 dims)\n")
            item = axes.get("항목", ["T1"])
            dims = {n: {"codes": c[:1]} for n, c in axes.items() if n != "항목"}
            periods = [{"kind": "recent", "n": 1}, {"kind": "anchor", "event": f"가상사건{k % 2}", "offset": 0},
                       {"kind": "recent", "n": 2}][k % 3]
            step = {"id": "a", "op": "select", "item": {"codes": item[:1]}, "dims": dims, "periods": periods}
            text = "broken" if k % 5 == 0 else json.dumps({"steps": [step], "show": ["a"], "present": "value"},
                                                          ensure_ascii=False)
        elif system.startswith("당신은 사건이 언제"):  # event-lookup-v1
            text = ['{"start": "2020-03"}', '{"start": null}', "bad"][k % 3]
        elif system.startswith("당신은 KOSIS 통계표의 계산 결과"):  # compose-v2
            main = re.findall(r"\{(f\d+)\} \[반드시 인용\]", user)
            text = ["not json", json.dumps({"answer": "값은 3102가구입니다."}, ensure_ascii=False),
                    json.dumps({"answer": "답은 " + ", ".join("{%s}" % f for f in main) + "입니다."}, ensure_ascii=False)][k % 3]
        else:  # the compact-v2 chatbot
            text = ['{"outcome":"answered","answer":"표에 따르면 그렇습니다."}', "x"][k % 2]
        return LLMResult(text=text, usage=usage)

    return reply


def reply(answer: str) -> str:
    return json.dumps({"answer": answer}, ensure_ascii=False)


GOOD_FILL = json.dumps({"codes": {"항목": ["T1"]}})
BAD_FILL = json.dumps({"codes": {"항목": ["T9"]}})
GOOD_PLAN = json.dumps({"steps": [{"id": "m", "op": "select", "item": {"codes": ["T1"]}, "dims": {"성별": {"codes": ["1"]}},
                                   "periods": {"kind": "list", "periods": ["2023"]}}], "show": ["m"], "present": "value"},
                       ensure_ascii=False)


# ---------------------------------------------------------------- running a case both ways


QUESTIONS = {
    "household": ["2023년 서울특별시 1인 가구 수는?", "서울특별시 1인 가구 추이는?", "2030년 서울 1인 가구 수는?",
                  "2023년 중구 1인 가구는?", "2023년 서울 1인 가구는?", "2023년 도심 1인 가구는?",
                  "2015~2024년 서울특별시 1인 가구를 모두 합치면 몇 가구야?",
                  "2015년부터 2024년까지 서울특별시 1인 가구의 연평균은?", "2023년 서울특별시와 부산광역시 1인 가구를 비교해줘"],
    "pop": ["2023년 여자 가상 인구는?", "2023년 여자 인구는?", "2023년 남자는?", "2019년 남자 가상 인구는?", "여자 인구는?",
            "2030년 여자 인구는?", "2022년부터 2023년까지 남자 가상 인구 추이는?"],
}


def _with_mode(monkeypatch, mode: str):
    """plain: the rules as they are; forced: build_plan always fails (the model routes for every question);
    first: only the first build_plan call of a question fails (the fill can place it). Returns the per-question reset."""
    calls = {"n": 0}
    if mode != "plain":
        real = pipeline_module.build_plan

        def unplaced(plan, index, table):
            calls["n"] += 1
            if mode == "forced" or calls["n"] == 1:
                raise Unresolved(["셀 선택"])
            return real(plan, index, table)

        monkeypatch.setattr(pipeline_module, "build_plan", unplaced)  # both ways read pipeline.build_plan at each ask
    return lambda: calls.update(n=0)


def _normal(answer) -> dict:
    """The Answer with its volatile times left out (as eval/parity.py does), plus what dict equality would not see."""
    meta = answer.decision_metadata
    spans = [{k: v for k, v in s.items() if k not in ("start_ms", "duration_ms")} for s in meta.get("spans", [])]
    return {
        "answer": replace(answer, latency_seconds=0.0, retry_wait_seconds=0.0, decision_metadata={**meta, "spans": spans}),
        "meta_keys": list(meta),
        "spans": spans,
        "stat_json": answer.stat_answer.model_dump_json() if answer.stat_answer is not None else None,
        "waited": answer.retry_wait_seconds > 0,
    }


def _run_both(path, question, replies, monkeypatch, *, retry_interval=1.0, reset=lambda: None):
    sleeps: dict[str, list] = {"graph": [], "seq": []}
    out = {}
    for way in ("graph", "seq"):
        monkeypatch.setattr(time, "sleep", sleeps[way].append)
        client = MockClient(list(replies) if isinstance(replies, list) else replies)
        bot = StatPipeline(load_table(path), client=client, org_id="101", retry_interval=retry_interval)
        reset()
        answer = bot.ask(question) if way == "graph" else sequential_ask(bot, question)
        out[way] = (_normal(answer), client.calls, answer)
    monkeypatch.setattr(time, "sleep", REAL_SLEEP)
    return out, sleeps


def _assert_same(out, sleeps):
    (graph, graph_calls, _), (seq, seq_calls, _) = out["graph"], out["seq"]
    assert graph_calls == seq_calls  # the same prompts, in the same order
    assert sleeps["graph"] == sleeps["seq"]
    assert graph["meta_keys"] == seq["meta_keys"]
    assert graph["spans"] == seq["spans"]
    assert graph["stat_json"] == seq["stat_json"]
    assert graph["waited"] == seq["waited"]
    assert graph["answer"] == seq["answer"]


CASES = [(which, q, mode, salt) for which, qs in QUESTIONS.items() for q in qs for mode in ("plain", "forced", "first")
         for salt in ("a", "b", "c")]
SEEN_ROUTES: set[str] = set()
SEEN_CASES: set[tuple] = set()


@pytest.mark.parametrize(("which", "question", "mode", "salt"), CASES, ids=[f"{w}-{q}-{m}-{s}" for w, q, m, s in CASES])
def test_graph_matches_the_router(household_path, pop_path, monkeypatch, which, question, mode, salt):
    reset = _with_mode(monkeypatch, mode)
    out, sleeps = _run_both(household_path if which == "household" else pop_path, question, scripted(salt), monkeypatch,
                            reset=reset)
    _assert_same(out, sleeps)
    SEEN_ROUTES.add(out["graph"][2].decision_metadata["route"])
    SEEN_CASES.add((which, question, mode, salt))


QUEUED = [  # fixed reply queues (tests/test_nodes_llm.py, tests/test_nodes_pure.py routes)
    ("pop", "2023년 남자는?", [GOOD_FILL, reply("{f1}입니다.")], "llm_fill"),
    ("pop", "2023년 남자는?", [BAD_FILL, GOOD_FILL, reply("{f1}입니다.")], "llm_fill"),
    ("pop", "2023년 남자는?", [BAD_FILL, BAD_FILL, GOOD_PLAN, reply("{f1}입니다.")], "llm_plan"),
    ("pop", "2023년 여자 인구는?", ["not json"] * 8, "fallback_chatbot"),
    ("household", "2023년 서울특별시 1인 가구 수는?", [reply("서울은 {f1}입니다.")], "rules"),
    ("household", "2023년 서울특별시 1인 가구 수는?", [reply("3102가구"), reply("서울은 {f1}입니다.")], "rules"),
    ("household", "2023년 서울특별시 1인 가구 수는?", [reply("3102가구")] * 5, "rules"),
    ("household", "2030년 서울 1인 가구 수는?", [], "rules_final"),
]


@pytest.mark.parametrize(("which", "question", "replies", "route"), QUEUED, ids=[f"{c[0]}-{c[1]}-{len(c[2])}" for c in QUEUED])
@pytest.mark.parametrize("interval", [0.0, 1.0])
def test_graph_matches_the_router_on_fixed_replies(household_path, pop_path, monkeypatch, which, question, replies,
                                                   route, interval):
    out, sleeps = _run_both(household_path if which == "household" else pop_path, question, replies or "x", monkeypatch,
                            retry_interval=interval)
    _assert_same(out, sleeps)
    assert out["graph"][2].decision_metadata["route"] == route


def test_the_cases_cover_the_routes():
    """Runs after the parametrized cases above (file order): the scripted cases reached the rules' routes and, in the
    forced mode, at least one model route."""
    if len(SEEN_CASES) < len(CASES):  # a -k selection or another order: only the full set must reach these routes
        pytest.skip("run with all the cases above")
    assert {"rules_final", "rules"} <= SEEN_ROUTES and len(SEEN_ROUTES) >= 3


# ---------------------------------------------------------------- the compose <-> verify loop


def _names(answer):
    return [s["name"] for s in answer.decision_metadata["spans"]]


def test_compose_loop_stops_at_the_cap(household_path):
    client = MockClient([reply("3102가구")] * 5)
    answer = StatPipeline(load_table(household_path), client=client, org_id="101").ask("2023년 서울특별시 1인 가구 수는?")
    assert _names(answer).count("compose") == COMPOSE_MAX_ATTEMPTS == len(client.calls)
    assert _names(answer)[-1] == "template" and answer.decision_metadata["compose"] == "template"
    assert len(answer.decision_metadata["verify_problems"]) == COMPOSE_MAX_ATTEMPTS


def test_compose_loop_cap_holds_against_a_verify_that_always_retries(household_path):
    """The verify edge caps the loop itself (as range(COMPOSE_MAX_ATTEMPTS) did), whatever the verify step says."""
    def always_retry(inp, res):
        return VerifyOut(problems=["가짜 문제"], route="retry")

    client = MockClient(reply("서울은 {f1}입니다."))
    bot = StatPipeline(load_table(household_path), client=client, org_id="101")
    bot._graph = build_graph(bot.resources, node_overrides={"verify": always_retry})
    answer = bot.ask("2023년 서울특별시 1인 가구 수는?")
    assert _names(answer) == ["judge", "analyze.rules", "compute", *["compose", "verify"] * COMPOSE_MAX_ATTEMPTS, "template"]
    assert len(client.calls) == COMPOSE_MAX_ATTEMPTS and answer.decision_metadata["compose"] == "template"


def test_recursion_limit_is_explicit_and_above_the_longest_path():
    assert build.LONGEST_PATH == 9 + 2 * COMPOSE_MAX_ATTEMPTS and RECURSION_LIMIT > build.LONGEST_PATH


# ---------------------------------------------------------------- node_overrides (E4)


def test_node_overrides_swap_in_a_fake_judge(pop_path):
    seen = []

    def fake_judge(inp, res):
        seen.append(inp.question)
        return step_judge(JudgeIn(question="2030년 여자 인구는?"), res)  # a year the table lacks: refused by the rules

    bot = StatPipeline(load_table(pop_path), client=MockClient("x"), org_id="101")
    bot._graph = build_graph(bot.resources, node_overrides={"judge": fake_judge})
    answer = bot.ask("2023년 여자 가상 인구는?")
    assert seen == ["2023년 여자 가상 인구는?"]
    assert answer.decision_metadata["route"] == "rules_final" and _names(answer) == ["judge"]
    assert answer.stat_answer.outcome != "answered"
    plain = StatPipeline(load_table(pop_path), client=MockClient("x"), org_id="101").ask("2023년 여자 가상 인구는?")
    assert plain.decision_metadata["route"] == "rules" and plain.stat_answer.outcome == "answered"


def test_node_overrides_reject_an_unknown_name(pop_path):
    bot = StatPipeline(load_table(pop_path), client=MockClient("x"), org_id="101")
    with pytest.raises(KeyError, match="unknown node"):
        build_graph(bot.resources, node_overrides={"judges": step_judge})
    assert set(NODE_TABLE) == {"judge", "plan_rules", "plan_fill_llm", "plan_write_llm", "fallback_chatbot", "compute",
                               "caveats", "compose_llm", "verify", "template", "chart", "render", "answer_final"}


def test_a_step_error_reaches_the_caller_as_itself(pop_path):
    """run_eval isolates a pipeline error per question by its type and message (call_error "KeyError: 'dims'")."""
    def broken(inp, res):
        raise KeyError("dims")

    bot = StatPipeline(load_table(pop_path), client=MockClient("x"), org_id="101")
    bot._graph = build_graph(bot.resources, node_overrides={"compute": broken})
    with pytest.raises(KeyError) as caught:
        bot.ask("2023년 여자 가상 인구는?")
    assert f"{type(caught.value).__name__}: {caught.value}" == "KeyError: 'dims'"


# ---------------------------------------------------------------- the chart branch beside compose


@pytest.mark.parametrize("replies", [[reply("서울은 {f1}입니다.")], [reply("3102가구"), reply("서울은 {f1}입니다.")],
                                     ["not json"]])
def test_a_late_chart_keeps_the_spans_and_the_answer(household_path, replies):
    events: list[str] = []
    lock = threading.Lock()

    def slow_chart(inp, res):
        REAL_SLEEP(1.0)
        with lock:
            events.append("chart_end")
        return step_chart(inp, res)

    inner = MockClient(list(replies))  # the same replies (and the same end of queue) as the plain runs below

    def model(system, user):
        with lock:
            events.append("model")
        return inner.generate(system, user)

    question = "2023년 서울특별시 1인 가구 수는?"
    slow = StatPipeline(load_table(household_path), client=MockClient(model), org_id="101")
    slow._graph = build_graph(slow.resources, node_overrides={"chart": slow_chart})
    late = slow.ask(question)
    plain = StatPipeline(load_table(household_path), client=MockClient(list(replies)), org_id="101").ask(question)
    reference = sequential_ask(StatPipeline(load_table(household_path), client=MockClient(list(replies)), org_id="101"),
                               question)
    for other in (plain, reference):
        assert _normal(late)["spans"] == _normal(other)["spans"]
        assert _normal(late)["meta_keys"] == _normal(other)["meta_keys"]
        assert _normal(late)["answer"] == _normal(other)["answer"]
    assert late.chart is not None and late.chart == reference.chart
    assert events[0] == "model" and "chart_end" in events  # the first compose call did not wait for the chart


# ---------------------------------------------------------------- the state and its reducers


def test_reducers_append_and_merge_like_the_router():
    old_records = [StepSpan(name="judge", started=0.0, ended=1.0)]
    new = [StepSpan(name="compute", started=1.0, ended=2.0)]
    together = append(old_records, new)
    assert [r.name for r in together] == ["judge", "compute"] and together is not old_records and len(old_records) == 1
    meta = {"pipeline": "stat", "route": "rules", "plan": {"a": 1}}
    merged = merge(meta, {"type": "value", "pattern": None, "plan": {"b": 2}})
    expected = dict(meta)
    expected.update({"type": "value", "pattern": None, "plan": {"b": 2}})
    assert list(merged) == list(expected) == ["pipeline", "route", "plan", "type", "pattern"] and merged == expected
    assert meta == {"pipeline": "stat", "route": "rules", "plan": {"a": 1}}


def test_the_compiled_graph_reduces_records_and_meta(pop_path):
    bot = StatPipeline(load_table(pop_path), client=MockClient("x"), org_id="101")
    channels = bot.graph.channels
    assert isinstance(channels["records"], BinaryOperatorAggregate) and channels["records"].operator is append
    assert isinstance(channels["meta"], BinaryOperatorAggregate) and channels["meta"].operator is merge


def test_pydantic_state_with_annotated_reducers_in_langgraph():
    """The LangGraph 1.x points the plan listed as unverified: a Pydantic state, Annotated reducers, dict returns,
    a node's input built as the model, objects kept as the same object (no copy), and a join of two branches."""
    marker = object()
    seen = {}

    def first(state: StatState) -> dict:
        return {"records": [StepSpan(name="a", started=0.0, ended=0.0)], "meta": {"x": 1, "route": "a"}, "stat": marker}

    def left(state: StatState) -> dict:
        REAL_SLEEP(0.05)
        return {"chart_out": "left"}

    def right(state: StatState) -> dict:
        seen["type"] = type(state)
        seen["stat"] = state.stat
        return {"records": [StepSpan(name="b", started=0.0, ended=0.0)], "meta": {"route": "b", "y": 2}}

    def join(state: StatState) -> dict:
        seen["join"] = (state.chart_out, [r.name for r in state.records])
        return {}

    g = StateGraph(StatState)
    for name, fn in (("first", first), ("left", left), ("right", right), ("join", join)):
        g.add_node(name, fn)
    g.add_edge(START, "first")
    g.add_edge("first", "left")
    g.add_edge("first", "right")
    g.add_edge(["left", "right"], "join")
    g.add_edge("join", END)
    out = g.compile().invoke({"question": "q", "meta": {"pipeline": "stat", "route": "rules"}},
                             {"recursion_limit": RECURSION_LIMIT})
    state = out if isinstance(out, StatState) else StatState(**out)
    assert seen["type"] is StatState and seen["stat"] is marker and state.stat is marker
    assert list(state.meta.items()) == [("pipeline", "stat"), ("route", "b"), ("x", 1), ("y", 2)]
    assert [r.name for r in state.records] == ["a", "b"] and seen["join"] == ("left", ["a", "b"])
