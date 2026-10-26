"""Raw mode (--pipeline raw): one call with table text + question, the reply kept as written, read by people."""

from __future__ import annotations

import json

import pytest

from eval import compare_runs, grade_run, run_eval, scorer, view_raw
from kosis_chat.chatbot import Chatbot
from kosis_chat.data.loader import load_table, sha256_of
from kosis_chat.llm.model import MockClient
from kosis_chat.llm.raw_prompt import SYSTEM_PROMPT
from kosis_chat.raw import RawRunner

REPLY = "```json\n{\"answer\": \"2023년 여자 가상 인구는 550명입니다.\"}\n```"  # any shape is kept as written
SCOPE = {"items": ["T1"], "periods": ["2023", "2023"]}


@pytest.fixture
def pop(pop_path):
    return load_table(pop_path)


def runner(table, settings, reply, **kwargs):
    client = MockClient(reply)
    return RawRunner(table, client=client, settings=settings, **kwargs), client


def test_one_call_with_the_table_and_the_question(pop, mock_settings):
    bot, client = runner(pop, mock_settings, REPLY)
    answer = bot.ask("2023년 여자 인구는?", SCOPE)
    assert len(client.calls) == 1 and client.calls[0][0] == SYSTEM_PROMPT
    user = client.calls[0][1]
    assert "[통계표]" in user and "가상 성별 인구" in user and user.rstrip().endswith("2023년 여자 인구는?")
    assert answer.ok and answer.attempts == 1 and answer.text == REPLY and answer.raw_responses == [REPLY]
    assert answer.stat_answer is None and answer.chart is None and answer.format_errors == []
    assert answer.decision_metadata["route"] == "raw"


def test_same_table_text_as_the_chatbot(pop, mock_settings):
    raw_text, record = runner(pop, mock_settings, REPLY)[0].prepare("q", SCOPE)
    chat_text, _ = Chatbot(pop, client=MockClient("x"), settings=mock_settings, scope_mode="oracle").prepare("q", SCOPE)
    assert raw_text == chat_text and len(record["table_text_sha256"]) == 64


def test_no_rules_run_before_the_model(pop, mock_settings):
    bot, client = runner(pop, mock_settings, "2019년 자료는 표에 없습니다.")
    answer = bot.ask("2019년 남자 가상 인구는?", SCOPE)  # the rules would refuse this without a call
    assert len(client.calls) == 1 and answer.judge == {} and answer.ok


def _golden(tmp_path, pop_path, items=None):
    items = items or [{"id": "q1", "kind": "answerable", "question": "2023년 여자 인구는?", "table": str(pop_path),
                       "table_sha256": sha256_of(pop_path), "scope": SCOPE}]
    gdir = tmp_path / "golden" / f"v{len(list(tmp_path.glob('golden/*')))}"
    gdir.mkdir(parents=True)
    (gdir / "a.jsonl").write_text("".join(json.dumps(i, ensure_ascii=False) + "\n" for i in items), encoding="utf-8")
    return gdir


def test_run_eval_raw_keeps_the_reply_and_view_raw_shows_it(pop_path, mock_settings, tmp_path):
    gdir = _golden(tmp_path, pop_path)
    run_dir = run_eval.run(gdir, "raw", mock_settings, tmp_path / "results", client=MockClient(REPLY), pipeline="raw")
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert (manifest["pipeline"], manifest["prompt_version"], manifest["judge_mode"]) == ("raw", "raw-v2", "off")
    row = scorer.read_jsonl(run_dir / "results.jsonl")[0]
    assert row["call_status"] == "ok" and row["raw_answer"] == REPLY and row["stat_answer"] is None
    assert row["scope"]["table_text_sha256"]
    page = view_raw.build(run_dir).read_text(encoding="utf-8")
    assert "550명" in page and "2023년 여자 인구는?" in page and "가상 성별 인구" in page  # reply, question, input table
    assert grade_run.main(["--run", str(run_dir)]) == 2  # raw runs are not graded
    with pytest.raises(run_eval.GoldenError, match="--form"):
        run_eval.run(gdir, "raw", mock_settings, tmp_path / "r2", client=MockClient(REPLY), pipeline="raw",
                     form_path=tmp_path / "form.txt")


def test_compare_runs_puts_both_answers_side_by_side(pop_path, mock_settings, tmp_path):
    gdir = _golden(tmp_path, pop_path)
    raw_dir = run_eval.run(gdir, "raw", mock_settings, tmp_path / "results", client=MockClient(REPLY), pipeline="raw")
    pipe_dir = run_eval.run(gdir, "chat", mock_settings, tmp_path / "results", client=MockClient("파이프라인 답 550명"))
    out = compare_runs.build(raw_dir, pipe_dir)
    page = out.read_text(encoding="utf-8")
    assert out.parent == raw_dir and "2023년 여자 가상 인구는" in page and "파이프라인 답 550명" in page
    with pytest.raises(ValueError, match="raw run"):
        compare_runs.build(raw_dir, raw_dir)


def test_compare_runs_stops_when_the_questions_differ(pop_path, mock_settings, tmp_path):
    raw_dir = run_eval.run(_golden(tmp_path, pop_path), "raw", mock_settings, tmp_path / "results",
                           client=MockClient(REPLY), pipeline="raw")
    other = [{"id": "q1", "kind": "answerable", "question": "2022년 남자 인구는?", "table": str(pop_path),
              "table_sha256": sha256_of(pop_path), "scope": SCOPE}]
    pipe_dir = run_eval.run(_golden(tmp_path, pop_path, other), "chat", mock_settings, tmp_path / "results",
                            client=MockClient("x"))
    with pytest.raises(ValueError, match="질문 다름"):
        compare_runs.build(raw_dir, pipe_dir)
    assert compare_runs.main(["--raw_path", str(raw_dir), "--pipe_path", str(pipe_dir)]) == 2
