"""Per-step spans of StatPipeline (T1-22) and their per-step summary in eval.grade_run."""

import pytest

from eval.grade_run import stage_summary
from kosis_chat.data.loader import load_table
from kosis_chat.llm.model import LLMResult, MockClient
from kosis_chat.pipeline import StatPipeline


@pytest.fixture
def table(pop_path):
    return load_table(pop_path)


def _spans(answer):
    return answer.decision_metadata["spans"]


def test_template_fallback_records_each_step(table, mock_settings):
    reply = LLMResult(text="not json", usage={"promptTokens": 120, "completionTokens": 7})
    answer = StatPipeline(table, client=MockClient(reply), settings=mock_settings, org_id="101").ask("2023년 여자 가상 인구는?")
    spans = _spans(answer)
    assert [s["name"] for s in spans] == ["judge", "analyze.rules", "compute", "compose", "template"]
    compose = spans[3]
    assert compose["status"] == "error"
    assert compose["attributes"]["gen_ai.request.model"] == "mock"
    assert compose["attributes"]["gen_ai.usage.input_tokens"] == 120 * compose["attributes"]["llm.attempts"]  # summed over format retries
    assert all(s["duration_ms"] >= 0 and s["start_ms"] >= 0 for s in spans)


def test_fallback_keeps_fill_tokens_and_marks_failed_steps(table, mock_settings):
    # two items in the table: the rules cannot pick one, the fill reply is not JSON, so the chatbot answers
    reply = LLMResult(text="not json", usage={"promptTokens": 120, "completionTokens": 7})
    answer = StatPipeline(table, client=MockClient(reply), settings=mock_settings, org_id="101").ask("2023년 여자 인구는?")
    assert [(s["name"], s["status"]) for s in _spans(answer)][:3] == [
        ("judge", "ok"), ("analyze.rules", "error"), ("analyze.llm_fill", "error")]
    assert _spans(answer)[-1]["name"] == "fallback_chatbot"
    # every model step before the chatbot (the fill, then the model's own plan, S5b) keeps its tokens in the total
    spent = [s["attributes"]["gen_ai.usage.input_tokens"] for s in _spans(answer) if "gen_ai.usage.input_tokens" in s["attributes"]]
    assert len(spent) >= 2 and answer.usage["promptTokens"] == sum(spent)


def test_final_verdict_is_one_ok_judge_span(table, mock_settings):
    answer = StatPipeline(table, client=MockClient("x"), settings=mock_settings, org_id="101").ask("2030년 여자 인구는?")
    assert answer.decision_metadata["route"] == "rules_final"
    assert [(s["name"], s["status"]) for s in _spans(answer)] == [("judge", "ok")]


def test_stage_summary_counts_failures_and_tokens():
    span = lambda name, status, ms, tokens=None: {  # noqa: E731
        "name": name, "status": status, "duration_ms": ms,
        "attributes": {"gen_ai.usage.input_tokens": tokens} if tokens else {}}
    rows = [{"decision_metadata": {"spans": [span("judge", "ok", 1), span("compose", "error", 900, 100)]}},
            {"decision_metadata": {"spans": [span("judge", "ok", 3), span("compose", "ok", 500, 50)]}},
            {"decision_metadata": {}}]
    summary = stage_summary(rows)
    assert list(summary) == ["judge", "compose"]
    assert summary["compose"] == {"runs": 2, "errors": 1, "p50_ms": 500, "p95_ms": 900, "input_tokens": 150, "output_tokens": 0}
