"""Chatbot flow with MockClient: prompt content and every call status."""

from __future__ import annotations

from dataclasses import replace

import pytest

from kosis_chat import cli
from kosis_chat.chatbot import Chatbot
from kosis_chat.data.loader import load_table
from kosis_chat.llm.model import HCXClient, LLMResult, MockClient, build_client
from kosis_chat.llm.prompts import NUMBER_RULE


@pytest.fixture
def table(fixtures):
    return load_table(fixtures / "pop_a.csv")


def test_ask_sends_rules_table_and_question(table, mock_settings):
    client = MockClient("답변 출처: KOSIS / 가상 인구조사 / 가상 성별 인구")
    answer = Chatbot(table, client=client, settings=mock_settings).ask("2023년 여자 인구는?")
    assert answer.ok and answer.text.startswith("답변")
    assert answer.prompt_version == "baseline-v1"
    system, user = client.calls[0]
    assert NUMBER_RULE in system
    assert "× 100" in system and "2023년 대비 2024년" in system
    assert "가상 성별 인구" in user and "2023년 여자 인구는?" in user


def test_input_too_long_is_not_truncated_or_sent(table, mock_settings):
    client = MockClient("never")
    settings = replace(mock_settings, max_input_tokens=10)
    answer = Chatbot(table, client=client, settings=settings).ask("질문")
    assert answer.status == "input_too_long"
    assert client.calls == []


def test_timeout_and_api_error_become_statuses(table, mock_settings):
    def raise_timeout(system, user):
        raise TimeoutError("slow")

    assert Chatbot(table, client=MockClient(raise_timeout), settings=mock_settings).ask("q").status == "timeout"
    bad = MockClient([LLMResult(text="", status="api_error", error="500")])
    answer = Chatbot(table, client=bad, settings=mock_settings).ask("q")
    assert (answer.status, answer.error) == ("api_error", "500")


def test_hcx_backend_builds_hcx_client(mock_settings, monkeypatch):
    monkeypatch.setenv("HCX_API_KEY", "nv-dummy")
    settings = replace(mock_settings, backend="hcx", model_name="HCX-007", base_url="http://x", api_key_env="HCX_API_KEY")
    assert isinstance(build_client(settings), HCXClient)


def test_cli_answers_questions_with_mock(fixtures, capsys):
    code = cli.main(["--table", str(fixtures / "pop_a.csv"), "--backend", "mock", "-q", "질문1", "-q", "질문2"])
    out = capsys.readouterr().out
    assert code == 0
    assert out.count("[mock]") == 2


def test_cli_reports_contract_problems(fixtures, capsys):
    assert cli.main(["--table", str(fixtures / "wide.csv"), "--backend", "mock", "-q", "x"]) == 2
    assert "wide format" in capsys.readouterr().err


def test_cli_form_file_is_sent_with_each_question(fixtures, tmp_path, capsys):
    form = tmp_path / "form.txt"
    form.write_text("﻿표 형식으로 답하세요.\n", encoding="utf-8")
    assert cli.main(["--table", str(fixtures / "pop_a.csv"), "--backend", "mock", "--form", str(form), "-q", "x"]) == 0
    assert cli.main(["--table", str(fixtures / "pop_a.csv"), "--backend", "mock", "--form", str(tmp_path / "none.txt"), "-q", "x"]) == 2
    assert "--form" in capsys.readouterr().err
