"""C pipeline on the made-up tables with a mock model: compose, verify, retry, template, fill, fallback, run_eval."""

import json

import pytest

from eval import grade_run, run_eval, scorer
from kosis_chat.answer.render import final_text, format_fact, josa
from kosis_chat.answer.template import template_parts
from kosis_chat.answer.verify import label_digits, verify
from kosis_chat.compute import build, compute
from kosis_chat.contract import to_stat_answer
from kosis_chat.data.loader import load_table, sha256_of
from kosis_chat.data.preprocessing import clean
from kosis_chat.llm import compose
from kosis_chat.llm.model import LLMResult, MockClient
from kosis_chat.pipeline import StatPipeline

from kosis_ai.schemas import Fact, FactPart, PlanV3, TextPart  # isort: skip


def ref(item, region, period):
    return f"FAKE_TBL_002|{item}|{region}|{period}"


def reply(answer):
    return json.dumps({"answer": answer}, ensure_ascii=False)


@pytest.fixture
def households(household_path):
    return load_table(household_path)


def pipeline(table, replies, **kw):
    return StatPipeline(table, client=MockClient(replies), org_id="101", **kw)


# ---------------------------------------------------------------- render, verify, template


def test_format_fact_and_josa():
    cell = Fact(id="f1", value=1234567.0, unit="가구", period="2023", source_ref="T|1|2023")
    diff = Fact(id="f3", value=-11.7, unit="%p", derivation="f2 - f1", round=1)
    growth = Fact(id="f4", value=-24.66, unit="%", derivation="(f2 / f1 - 1) * 100", round=1)
    symbol = Fact(id="f2", raw="x", period="2023", source_ref="T|1|2024")
    assert [format_fact(f) for f in (cell, diff, growth, symbol)] == ["1,234,567가구", "11.7%p", "24.7%", "'x'(비공개)"]
    assert josa("서울특별시", "이") == "서울특별시가" and josa("경기도", "은") == "경기도는" and josa("1,234명", "으로") == "1,234명으로"
    assert josa("39.3%", "으로") == "39.3%로" and josa("서울", "으로") == "서울로"


def test_verify_flags_typed_numbers_missing_facts_and_wrong_direction(households):
    table = clean(households)
    computed = compute(table, build.trend([ref("T1", "21", p) for p in ("2015", "2020", "2023")]))
    good = compose.to_parts("부산 1인 가구는 2015년 {f1}에서 2023년 {f3}으로 {f4} 늘었습니다(증가율 {f5}).", computed)
    assert verify(good, computed) == [] and label_digits(good, computed) == 1  # '1인'
    typed = compose.to_parts("2015년 {f1}에서 {f3}으로 3가구 늘었습니다.", computed)
    problems = verify(typed, computed)
    assert any("직접 쓴 숫자: 3" in p for p in problems) and any("빠진 사실" in p for p in problems)
    wrong = compose.to_parts("{f1}에서 {f3}으로 {f4} 줄었습니다({f5}).", computed)
    assert any("증가인데 감소" in p for p in verify(wrong, computed))


def test_to_parts_drops_a_repeated_unit(households):
    computed = compute(clean(households), build.value(ref("T1", "11", "2023")))
    parts = compose.to_parts("서울은 {f1}가구입니다.", computed)
    assert parts == [TextPart(text="서울은 "), FactPart(fact_id="f1"), TextPart(text="입니다.")]


@pytest.mark.parametrize("built", [
    build.value(ref("T1", "11", "2023")),
    build.trend([ref("T1", "21", p) for p in ("2015", "2020", "2023")]),
    build.compare([ref("T1", "11", "2023"), ref("T1", "00", "2023")], 1),
    build.share([ref("T1", "11", "2023")], ref("T0", "11", "2023")),
    build.extreme([ref("T1", r, "2023") for r in ("11", "21", "39")], "max"),
])
def test_templates_pass_verification(households, built):
    computed = compute(clean(households), built)
    assert verify(template_parts(built.plan.present, computed), computed) == []


def test_templates_cite_a_zero_difference():
    from kosis_chat.compute.engine import Computed
    from kosis_ai.schemas import StatResult

    def facts(period_b):
        dims = {"항목": "1인 (가구)", "행정구역별": "서울특별시"}
        a = Fact(id="f1", value=100.0, unit="가구", period="2023", dims=dims, source_ref="T|T1|11|2023")
        b = Fact(id="f2", value=100.0, unit="가구", period=period_b, dims=dims if period_b != "2023" else dims | {"행정구역별": "경기도"},
                 source_ref=f"T|T1|{'11' if period_b != '2023' else '31'}|{period_b}")
        diff = Fact(id="f3", value=0, unit="가구", derivation="f2 - f1", round=0)
        return [a, b, diff]

    trend = Computed(result=StatResult(facts=facts("2024")), main=["f1", "f2", "f3"])
    parts = template_parts("trend", trend)
    assert verify(parts, trend) == [] and "변함이 없습니다(차이 " in "".join(p.text for p in parts if isinstance(p, TextPart))
    compare = Computed(result=StatResult(facts=facts("2023")), main=["f1", "f2", "f3"])
    parts = template_parts("compare", compare)
    assert verify(parts, compare) == [] and "두 값은 같습니다(차이 " in "".join(p.text for p in parts if isinstance(p, TextPart))


def test_final_text_adds_default_note_and_source():
    from kosis_ai.schemas import Source
    source = Source(org_id="101", tbl_id="T", survey_nm="조사", tbl_nm="표")
    text = final_text("본문입니다.", source, {"period": "latest (2024)"}, ["2022년은 표에 없음", None])
    assert text == "질문에 시점이 없어 가장 최근 시점인 2024년 기준으로 답합니다.\n본문입니다.\n참고: 2022년은 표에 없음\n\n출처: KOSIS / 조사 / 표"


# ---------------------------------------------------------------- pipeline


def test_model_sentence_with_code_numbers(households):
    bot = pipeline(households, [reply("2023년 서울특별시의 1인 가구는 {f1}입니다.")])
    answer = bot.ask("2023년 서울특별시 1인 가구 수는?")
    stat = answer.stat_answer
    assert answer.ok and answer.decision_metadata["compose"] == "model" and answer.attempts == 1
    assert stat.parts[1] == FactPart(fact_id="f1") and stat.result.facts[0].source_ref == ref("T1", "11", "2023")
    assert stat.text.startswith("2023년 서울특별시의 1인 가구는 3,102가구입니다.") and "출처: KOSIS" in stat.text
    assert stat.query.type == "value" and stat.query.filters == {"행정구역별(읍면동)": "서울특별시"}
    assert to_stat_answer("q", answer) is stat
    _, user = bot.client.calls[0]
    assert "{f1} [반드시 인용] 2023년 서울특별시 1인 가구 = 3,102가구" in user


def test_one_retry_after_a_typed_number(households):
    bot = pipeline(households, [reply("서울은 3102가구입니다."), reply("서울은 {f1}입니다.")])
    answer = bot.ask("2023년 서울특별시 1인 가구 수는?")
    assert answer.decision_metadata["compose"] == "model_retry" and len(bot.client.calls) == 2
    assert "[이전 답변의 문제]" in bot.client.calls[1][1] and "직접 쓴 숫자: 3102" in bot.client.calls[1][1]


def test_template_after_two_failures(households):
    bot = pipeline(households, [reply("3102가구"), reply("3102가구")])
    answer = bot.ask("2023년 서울특별시 1인 가구 수는?")
    assert answer.decision_metadata["compose"] == "template"
    assert answer.stat_answer.text.startswith("2023년 서울특별시 1인 가구는 3,102가구입니다.")


def test_rule_refusal_sends_no_call(households):
    bot = pipeline(households, [])
    answer = bot.ask("2030년 서울 1인 가구 수는?")
    assert answer.stat_answer.outcome == "refused" and answer.stat_answer.parts == [] and bot.client.calls == []
    assert answer.decision_metadata["route"] == "rules_final"


def test_model_fills_what_the_rules_leave_open(pop_path):
    fill = json.dumps({"type": "value", "codes": {"항목": ["T1"]}})
    bot = pipeline(load_table(pop_path), [fill, reply("2023년 남자는 {f1}입니다.")])
    answer = bot.ask("2023년 남자는?")
    assert answer.decision_metadata["route"] == "llm_fill" and answer.decision_metadata["unresolved"] == ["항목"]
    assert answer.stat_answer.result.facts[0].source_ref == "FAKE_TBL_001|T1|1|2023"
    user = bot.client.calls[0][1]  # the menu lists only the category the rules could not settle
    assert "[채울 분류 목록]" in user and "T1=가상 인구" in user and "- 성별:" not in user and "남자" in user


def test_model_fill_label_is_turned_into_its_code(pop_path):
    fill = json.dumps({"type": "value", "codes": {"항목": ["가상 인구"]}}, ensure_ascii=False)
    bot = pipeline(load_table(pop_path), [fill, reply("2023년 남자는 {f1}입니다.")])
    answer = bot.ask("2023년 남자는?")
    assert answer.stat_answer.result.facts[0].source_ref == "FAKE_TBL_001|T1|1|2023" and len(bot.client.calls) == 2
    assert "fill_retry" not in answer.decision_metadata and "ignored" not in answer.decision_metadata["fill"]


def test_model_fill_filed_under_a_wrong_category_name_is_moved(pop_path):
    fill = json.dumps({"codes": {"지표": ["T1"]}}, ensure_ascii=False)
    answer = pipeline(load_table(pop_path), [fill, reply("2023년 남자는 {f1}입니다.")]).ask("2023년 남자는?")
    assert answer.stat_answer.result.facts[0].source_ref == "FAKE_TBL_001|T1|1|2023"


def test_model_fill_problems_are_said_back_once(pop_path):
    first = json.dumps({"codes": {"시점": ["2023"], "항목": ["인구"]}}, ensure_ascii=False)  # a period, and no such label
    second = json.dumps({"codes": {"항목": ["T1"]}})
    bot = pipeline(load_table(pop_path), [first, second, reply("2023년 남자는 {f1}입니다.")])
    answer = bot.ask("2023년 남자는?")
    meta = answer.decision_metadata
    assert meta["route"] == "llm_fill" and meta["fill"]["ignored"] == ["분류 없음 시점", "코드 없음 항목=인구"]
    assert meta["fill_retry"] == {"codes": {"항목": ["T1"]}} and answer.attempts == 3
    retry = bot.client.calls[1][1]
    assert "[이전 답변의 문제]" in retry and "시점은 codes에 넣지 않습니다." in retry and "'인구'" in retry
    assert answer.stat_answer.result.facts[0].source_ref == "FAKE_TBL_001|T1|1|2023"


# A plan the model writes over a code the table does not have: said back once, then given up (S5b).
BAD_PLAN = json.dumps({"steps": [{"id": "a", "op": "select", "item": {"codes": ["T9"]},
                                  "periods": {"kind": "list", "periods": ["2023"]}}], "show": ["a"], "present": "value"})


def test_unplaced_question_goes_to_the_chatbot(pop_path):
    fill = json.dumps({"codes": {"항목": ["T9"]}})  # not in the table: ignored, said back once, then the model's plan
    chat = json.dumps({"outcome": "answered", "message": "남자는 500명입니다."}, ensure_ascii=False)
    answer = pipeline(load_table(pop_path), [fill, fill, BAD_PLAN, BAD_PLAN, chat]).ask("2023년 남자는?")
    assert answer.stat_answer is None and answer.prompt_version == "compact-v2"
    meta = answer.decision_metadata
    assert meta["route"] == "fallback_chatbot" and meta["fill"]["ignored"] and answer.attempts == 5
    assert meta["fill_retry"]["ignored"] == ["코드 없음 항목=T9"]
    assert meta["plan_problems"] == meta["plan_retry_problems"] == ["'T9'은(는) 항목 목록에 없는 값입니다. 목록에서 '=' 앞의 코드를 씁니다."]
    assert to_stat_answer("2023년 남자는?", answer, org_id="101").text.startswith("남자는 500명입니다.")


def test_run_eval_stat_pipeline(household_path, mock_settings, tmp_path):
    items = [{"id": "q1", "kind": "answerable", "type": "value", "question": "2023년 서울특별시 1인 가구 수는?",
              "table": str(household_path), "table_sha256": sha256_of(household_path), "scope": {},
              "required_facts": [{"id": "f1", "value": 3102, "unit": "가구", "period": "2023", "source_ref": ref("T1", "11", "2023")}]}]
    gdir = tmp_path / "golden"
    gdir.mkdir()
    (gdir / "g.jsonl").write_text("".join(json.dumps(i, ensure_ascii=False) + "\n" for i in items), encoding="utf-8")
    run_dir = run_eval.run(gdir, "C", mock_settings, tmp_path / "results", client=MockClient([reply("서울은 {f1}입니다.")]),
                           pipeline="stat")
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["pipeline"] == "stat" and manifest["prompt_version"] == "compose-v2" and manifest["judge_mode"] == "rules"
    (row,) = scorer.read_jsonl(run_dir / "results.jsonl")
    assert row["stat_answer"]["parts"][1] == {"kind": "fact", "fact_id": "f1"} and row["call_status"] == "ok"
    assert grade_run.main(["--run", str(run_dir)]) == 0
    (graded,) = scorer.read_jsonl(next((run_dir / "grades").iterdir()) / "grades.jsonl")
    assert graded["verdict"] == "PASS"


# ---------------------------------------------------------------- charts and the result page


@pytest.mark.parametrize("built, kind, rows, highlighted", [
    (build.value(ref("T1", "11", "2023")), "value", 4, ["2023"]),  # every numeric period (2025 is blank)
    (build.trend([ref("T1", "21", p) for p in ("2015", "2020", "2023")]), "trend", 3, ["2015", "2023"]),
    (build.compare([ref("T1", "11", "2023"), ref("T1", "00", "2023")], 1), "compare", 2, ["서울특별시"]),
    (build.share([ref("T1", "11", "2023")], ref("T0", "11", "2023")), "share", 2, ["1인 가구"]),  # the part and a '그 외' slice
    (build.extreme([ref("T1", r, "2023") for r in ("11", "21", "39")], "max"), "extreme", 3, ["제주특별자치도"]),
])  # gap and grw are drawn as a grid of targets by periods (S6, tests/test_answer_contract.py)
def test_chart_for_each_type(households, built, kind, rows, highlighted):
    from kosis_chat.answer.chart import SCHEMA, chart_spec
    chart = chart_spec(kind, compute(clean(households), built))
    spec, values = chart.spec, chart.spec["data"]["values"]
    assert spec["$schema"] == SCHEMA and len(values) == rows
    assert [v["이름"] for v in values if v["강조"] == "예"] == highlighted
    assert chart.fact_ids and chart.model_dump()  # the shared Chart contract: kind, spec, the facts drawn
    json.dumps(spec)  # plain JSON, ready for any Vega-Lite viewer


def test_pipeline_answer_carries_a_chart_and_the_page_shows_it(household_path, mock_settings, tmp_path):
    from eval import view_run
    items = [{"id": "q1", "kind": "answerable", "type": "extreme", "question": "2023년 1인 가구가 가장 많은 시도는?",
              "table": str(household_path), "table_sha256": sha256_of(household_path), "scope": {}}]
    gdir = tmp_path / "golden"
    gdir.mkdir()
    (gdir / "g.jsonl").write_text(json.dumps(items[0], ensure_ascii=False) + "\n", encoding="utf-8")
    run_dir = run_eval.run(gdir, "C", mock_settings, tmp_path / "results", client=MockClient("x"), pipeline="stat")
    (row,) = scorer.read_jsonl(run_dir / "results.jsonl")
    assert row["chart"]["data"]["values"][0]["이름"] == "제주특별자치도"
    page = view_run.build(run_dir).read_text(encoding="utf-8")
    assert "vega-embed@6.26.0" in page and "제주특별자치도" in page and "2023년 1인 가구가 가장 많은 시도는?" in page
    assert PlanV3.model_validate(row["decision_metadata"]["plan"]).present == "extreme"  # the plan is saved
    assert '"plan": {' in page and '"op": "rank"' in page and "function planHtml" in page  # and handed to the page


def test_plan_is_saved_only_when_code_computed(households):
    answered = pipeline(households, [reply("서울은 {f1}입니다.")]).ask("2023년 서울특별시 1인 가구 수는?")
    plan = PlanV3.model_validate(answered.decision_metadata["plan"])
    assert plan.steps[0].op == "select" and plan.show
    json.dumps(answered.decision_metadata["plan"])  # plain JSON for results.jsonl
    refused = pipeline(households, []).ask("2030년 서울 1인 가구 수는?")
    assert refused.stat_answer.outcome == "refused" and "plan" not in refused.decision_metadata


# ---------------------------------------------------------------- T0-03 v2: who wrote the numbers


def test_answered_records_code_numbers_as_verified(households):
    answer = pipeline(households, [reply("서울은 {f1}입니다.")]).ask("2023년 서울특별시 1인 가구 수는?")
    stat = answer.stat_answer
    assert stat.numbers_origin == "code" and stat.numbers_verified and stat.defaults_verified
    assert answer.decision_metadata["numbers_origin"] == "code" and answer.decision_metadata["numbers_verified"] is True


def test_template_fallback_is_verified_too(households):
    answer = pipeline(households, [reply("3102가구"), reply("3102가구")]).ask("2023년 서울특별시 1인 가구 수는?")
    assert answer.stat_answer.numbers_verified and "template_problems" not in answer.decision_metadata


def test_rule_refusal_has_no_numbers(households):
    stat = pipeline(households, []).ask("2030년 서울 1인 가구 수는?").stat_answer
    assert stat.numbers_origin == "none" and not stat.numbers_verified and not stat.defaults_verified


def test_chatbot_fallback_records_model_numbers_and_sums_usage(pop_path):
    fill = LLMResult(text=json.dumps({"codes": {"항목": ["T9"]}}), usage={"promptTokens": 10, "completionTokens": 2})
    plan = LLMResult(text=BAD_PLAN, usage={"promptTokens": 20, "completionTokens": 4})
    chat = LLMResult(text=json.dumps({"outcome": "answered", "message": "남자는 500명입니다."}, ensure_ascii=False),
                     usage={"promptTokens": 30, "completionTokens": 5, "model": "mock"})  # only the counts are summed
    # fill and its retry, the model's plan and its retry, the chatbot
    answer = pipeline(load_table(pop_path), [fill, fill, plan, plan, chat]).ask("2023년 남자는?")
    meta = answer.decision_metadata
    assert meta["route"] == "fallback_chatbot" and meta["numbers_origin"] == "model" and meta["numbers_verified"] is False
    assert answer.usage == {"promptTokens": 90, "completionTokens": 17} and answer.attempts == 5
    stat = to_stat_answer("2023년 남자는?", answer, org_id="101")
    assert stat.numbers_origin == "model" and stat.result.facts == [] and not stat.numbers_verified


def test_run_eval_isolates_a_pipeline_error_per_question(household_path, mock_settings, tmp_path, monkeypatch):
    items = [{"id": f"q{i}", "kind": "answerable", "type": "value", "question": q, "table": str(household_path),
              "table_sha256": sha256_of(household_path), "scope": {},
              "required_facts": [{"id": "f1", "value": 3102, "unit": "가구", "period": "2023", "source_ref": ref("T1", "11", "2023")}]}
             for i, q in enumerate(["2023년 서울특별시 1인 가구 수는?", "2023년 서울특별시 1인 가구 수는?"], start=1)]
    gdir = tmp_path / "golden"
    gdir.mkdir()
    (gdir / "g.jsonl").write_text("".join(json.dumps(i, ensure_ascii=False) + "\n" for i in items), encoding="utf-8")
    original = StatPipeline.ask
    calls = []

    def flaky(self, question, scope=None):
        calls.append(question)
        if len(calls) == 1:
            raise KeyError("dims")  # a bug in our own code on the first question
        return original(self, question, scope)

    monkeypatch.setattr(StatPipeline, "ask", flaky)
    run_dir = run_eval.run(gdir, "C", mock_settings, tmp_path / "results", client=MockClient([reply("서울은 {f1}입니다.")]),
                           pipeline="stat")
    first, second = scorer.read_jsonl(run_dir / "results.jsonl")
    assert first["call_status"] == "pipeline_error" and first["call_error"] == "KeyError: 'dims'" and first["stat_answer"] is None
    assert second["call_status"] == "ok" and second["stat_answer"]["numbers_origin"] == "code"
    events = scorer.read_jsonl(run_dir / "events.jsonl")
    (error_event,) = [e for e in events if e.get("type") == "pipeline_error"]
    assert "KeyError" in error_event["traceback"]
    assert grade_run.main(["--run", str(run_dir)]) == 0
    graded = scorer.read_jsonl(next((run_dir / "grades").iterdir()) / "grades.jsonl")
    assert [g["verdict"] for g in graded] == ["FAIL", "PASS"] and graded[0]["error_types"] == ["run_failure"]


# ---------------------------------------------------------------- patterns answered end to end, charts in the contract


def test_average_question_is_answered_with_a_chart_in_the_contract(households):
    bot = pipeline(households, [reply("2015년부터 2024년까지 서울특별시 1인 가구는 연평균 {f5}입니다.")])
    answer = bot.ask("2015년부터 2024년까지 서울특별시 1인 가구의 연평균은?")
    stat = answer.stat_answer
    assert answer.decision_metadata["pattern"] == "avg" and stat.query.periods == ["2015", "2020", "2023", "2024"]
    assert [s.op for s in stat.query.plan.steps] == ["select", "aggregate"]
    assert stat.result.facts[4].derivation == "(f1 + f2 + f3 + f4) / 4" and stat.text.startswith("2015년부터 2024년까지 서울특별시 1인 가구는 연평균 3,101.5가구입니다.")
    assert stat.chart.kind == "timeline" and stat.chart.fact_ids == ["f1", "f2", "f3", "f4", "f5"] and answer.chart == stat.chart.spec
    assert stat.numbers_origin == "code" and stat.numbers_verified and answer.decision_metadata["chart"] == "timeline"
    _, user = bot.client.calls[0]
    assert "{f5} [반드시 인용] 평균 ((f1 + f2 + f3 + f4) / 4) = 3,101.5가구 — 2015년~2024년 4개 시점 평균" in user


@pytest.mark.parametrize("question, pattern, text", [
    ("2015년부터 2024년까지 서울특별시 1인 가구의 연평균은?", "avg", "2015년~2024년 서울특별시 1인 가구 평균은 3,101.5가구입니다."),
    ("서울특별시 1인 가구는 2015~2020년 평균보다 2023~2024년 평균이 얼마나 많아?", "pavg",
     "서울특별시 1인 가구는 2015년~2020년 평균 3,100.5가구에서 2023년~2024년 평균 3,102.5가구로 2.0가구 늘었습니다(증가율 0.1%)."),
    ("2024년 서울특별시 1인 가구는 2015~2023년 평균보다 얼마나 많아?", "vsa",
     "2024년 서울특별시 1인 가구는 3,103가구로 2015년~2023년 평균 3,101.0가구보다 2.0가구 많습니다(증가율 0.1%)."),
    ("전국 일반가구 중 1인 가구 비중은 2015년보다 2024년에 몇 %p 늘었어?", "shr",
     "전국 일반가구 전체 가운데 1인 가구 비중은 2015년 11.0%에서 2024년 11.0%로 변함이 없습니다(차이 0.0%p)."),  # made-up values: equal shares
    ("서울과 부산의 1인 가구 격차는 2015년보다 2024년에 얼마나 변했어?", "gap",
     ("서울특별시와 부산광역시의 1인 가구 격차는 2015년 1,000가구(서울특별시 3,100가구, 부산광역시 4,100가구)에서 2024년 "
      "1,000가구(서울특별시 3,103가구, 부산광역시 4,103가구)로 변함이 없습니다(차이 0가구).")),  # made-up values: equal gaps
    ("2015년 대비 2024년 1인 가구 증가율은 서울과 부산 중 어디가 더 높아?", "grw",
     "2015년~2024년 서울특별시는 3,100가구에서 3,103가구로 0.1% 증가했고, 부산광역시는 4,100가구에서 4,103가구로 0.1% 증가했습니다."),
])
def test_pattern_templates_pass_verification(households, question, pattern, text):
    bot = pipeline(households, [reply("숫자 없이"), reply("숫자 없이")])  # the model never cites: the template answers
    answer = bot.ask(question)
    stat = answer.stat_answer
    assert answer.decision_metadata["compose"] == "template" and answer.decision_metadata["pattern"] == pattern
    assert stat.numbers_verified
    assert stat.text.startswith(text), stat.text
    assert stat.chart is not None and all(stat.result.get(i) for i in stat.chart.fact_ids)


def test_stock_sum_is_refused_before_any_call(households):
    bot = pipeline(households, [])
    answer = bot.ask("2015~2024년 서울특별시 1인 가구를 모두 합치면 몇 가구야?")
    assert answer.stat_answer.outcome == "refused" and answer.stat_answer.refusal.reason == "not_aggregatable" and bot.client.calls == []
    assert answer.stat_answer.chart is None and answer.chart is None
