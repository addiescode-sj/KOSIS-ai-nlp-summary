"""GeminiClient with a fake SDK client (no network, no key)."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest

from kosis_chat.chatbot import Chatbot
from kosis_chat.config import load_settings
from kosis_chat.data.loader import load_table
from kosis_chat.llm.model import GeminiClient, build_client

genai_errors = pytest.importorskip("google.genai.errors")


@pytest.fixture
def table(fixtures):
    return load_table(fixtures / "pop_a.csv")


@pytest.fixture
def gemini_settings(mock_settings):
    return replace(mock_settings, backend="gemini", model_name="gemini-test", thinking_level="low", max_tokens=8192)


class FakeModels:
    def __init__(self, reply=None, exc=None):
        self.reply, self.exc, self.calls = reply, exc, []

    def generate_content(self, model, contents, config):
        self.calls.append((model, contents, config))
        if self.exc:
            raise self.exc
        return self.reply


def _client(settings, **kw):
    fake = SimpleNamespace(models=FakeModels(**kw))
    return GeminiClient(settings, sdk_client=fake), fake.models


def _reply(text, finish="STOP"):
    return SimpleNamespace(text=text, usage_metadata=None, candidates=[SimpleNamespace(finish_reason=finish)])


def test_ok_reply_and_request_shape(table, gemini_settings):
    client, models = _client(gemini_settings, reply=_reply("2023년 여자 인구는 100명입니다."))
    answer = Chatbot(table, client=client, settings=gemini_settings).ask("여자 인구는?")
    assert answer.status == "ok" and "100명" in answer.text
    model, contents, config = models.calls[0]
    assert model == "gemini-test" and contents.startswith("") and "여자 인구는?" in contents
    assert config.system_instruction and config.max_output_tokens == 8192
    assert config.temperature is None  # Gemini 3.x: not sent
    assert str(config.thinking_config.thinking_level).lower().endswith("low")


def test_empty_reply_reports_finish_reason(table, gemini_settings):
    client, _ = _client(gemini_settings, reply=_reply(None, finish="MAX_TOKENS"))
    answer = Chatbot(table, client=client, settings=gemini_settings).ask("q")
    assert answer.status == "api_error" and "MAX_TOKENS" in answer.error


def test_api_errors_become_statuses(table, gemini_settings):
    err = genai_errors.ClientError(429, {"error": {"message": "quota exceeded", "status": "RESOURCE_EXHAUSTED"}})
    client, _ = _client(gemini_settings, exc=err)
    answer = Chatbot(table, client=client, settings=gemini_settings).ask("q")
    assert answer.status == "api_error" and "429" in answer.error

    class ReadTimeout(Exception):
        pass

    client, _ = _client(gemini_settings, exc=ReadTimeout("slow"))
    assert Chatbot(table, client=client, settings=gemini_settings).ask("q").status == "timeout"


def test_missing_key_is_a_config_error(gemini_settings, monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(ValueError, match="GEMINI_API_KEY"):
        build_client(replace(gemini_settings, api_key_env="GEMINI_API_KEY"))


def test_load_settings_uses_gemini_block(tmp_path):
    cfg = tmp_path / "m.yaml"
    cfg.write_text(
        "backend: gemini\nmodel: {name: HCX-X, max_tokens: 1024}\n"
        "gemini: {model: gemini-3.8-flash, thinking_level: low, max_tokens: 8192}\n",
        encoding="utf-8",
    )
    s = load_settings(cfg)
    assert (s.model_name, s.thinking_level, s.max_tokens, s.api_key_env) == ("gemini-3.8-flash", "low", 8192, "GEMINI_API_KEY")
    assert s.model_params()["temperature"] is None
    assert load_settings(cfg, backend="hcx").model_name == "HCX-X"
