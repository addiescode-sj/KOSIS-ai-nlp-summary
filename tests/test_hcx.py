"""HCXClient against a local fake CLOVA Studio server (no network, dummy key)."""

from __future__ import annotations

from dataclasses import replace

import pytest

from kosis_chat.chatbot import Chatbot
from kosis_chat.config import load_settings
from kosis_chat.data.loader import load_table
from kosis_chat.llm.model import HCXClient, build_client


@pytest.fixture
def table(fixtures):
    return load_table(fixtures / "pop_a.csv")


@pytest.fixture(autouse=True)
def hcx_key(monkeypatch):
    monkeypatch.setenv("HCX_API_KEY", "nv-dummy")


def no_sleep(_):
    pass


def _settings(mock_settings, base_url, **kw):
    values = {"backend": "hcx", "base_url": base_url, "model_name": "HCX-007", "thinking_level": "none"}
    return replace(mock_settings, **{**values, **kw})


def ok_body(text="2023년 여자 인구는 100명입니다."):
    return {
        "status": {"code": "20000", "message": "OK"},
        "result": {
            "message": {"role": "assistant", "content": text},
            "finishReason": "stop",
            "usage": {"promptTokens": 120, "completionTokens": 15, "totalTokens": 135},
        },
    }


def test_reasoning_model_request_shape(table, mock_settings, server_factory):
    srv = server_factory(body=ok_body())
    settings = _settings(mock_settings, srv.base_url)
    answer = Chatbot(table, settings=settings).ask("여자 인구는?")
    assert answer.status == "ok" and "100명" in answer.text
    req = srv.requests[0]
    assert [m["role"] for m in req["messages"]] == ["system", "user"]
    assert req["temperature"] == 0 and req["maxCompletionTokens"] == settings.max_tokens
    assert req["thinking"] == {"effort": "none"} and "maxTokens" not in req
    assert srv.request_headers[0]["Authorization"] == "Bearer nv-dummy"
    assert HCXClient(settings).url == srv.base_url + "/v3/chat-completions/HCX-007"


def test_light_model_uses_max_tokens_without_thinking(mock_settings, server_factory):
    srv = server_factory(body=ok_body())
    client = HCXClient(_settings(mock_settings, srv.base_url, model_name="HCX-DASH-002", max_tokens=512))
    assert client.generate("sys", "user").usage["totalTokens"] == 135
    assert srv.requests[0]["maxTokens"] == 512 and "thinking" not in srv.requests[0]


def test_retries_rate_limit_then_succeeds(mock_settings, server_factory):
    srv = server_factory(status=429, body={"status": {"code": "42901", "message": "Too many requests"}})
    delays = []

    def fake_sleep(delay):
        delays.append(delay)
        srv.status, srv.body = 200, ok_body("ok")

    result = HCXClient(_settings(mock_settings, srv.base_url), sleep=fake_sleep).generate("s", "u")
    assert result.status == "ok" and result.text == "ok"
    assert delays == [HCXClient.RETRY_DELAYS[0]] and len(srv.requests) == 2


def test_retry_waits_for_longer_retry_after(mock_settings, server_factory):
    srv = server_factory(status=429, body={"status": {"code": "42901"}}, headers={"Retry-After": "40"})
    delays = []
    HCXClient(_settings(mock_settings, srv.base_url), sleep=delays.append).generate("s", "u")
    assert delays == [40.0, 40.0, 40.0, 60.0]


def test_errors_become_statuses(mock_settings, server_factory):
    busy = server_factory(status=503, body={"status": {"code": "50000", "message": "busy"}})
    result = HCXClient(_settings(mock_settings, busy.base_url), sleep=no_sleep).generate("s", "u")
    assert result.status == "api_error" and "HTTP 503" in result.error
    assert len(busy.requests) == 1 + len(HCXClient.RETRY_DELAYS)

    bad = server_factory(status=401, body={"status": {"code": "40100", "message": "Unauthorized"}})
    assert "HTTP 401" in HCXClient(_settings(mock_settings, bad.base_url), sleep=no_sleep).generate("s", "u").error
    assert len(bad.requests) == 1  # auth errors are not retried

    long = server_factory(status=400, body={"status": {"code": "40003", "message": "Context length exceeded."}})
    assert HCXClient(_settings(mock_settings, long.base_url)).generate("s", "u").status == "input_too_long"

    empty = server_factory(body=ok_body(""))
    assert "empty response" in HCXClient(_settings(mock_settings, empty.base_url)).generate("s", "u").error


def test_timeout_and_connection_failure(mock_settings, server_factory):
    slow = server_factory(body=ok_body(), delay=1.0)
    assert HCXClient(_settings(mock_settings, slow.base_url, timeout_seconds=0.2)).generate("s", "u").status == "timeout"
    assert HCXClient(_settings(mock_settings, "http://127.0.0.1:9")).generate("s", "u").status == "api_error"


def test_missing_key_or_model_is_a_config_error(mock_settings, monkeypatch):
    monkeypatch.delenv("HCX_API_KEY")
    with pytest.raises(ValueError, match="HCX_API_KEY"):
        build_client(_settings(mock_settings, "http://x"))
    monkeypatch.setenv("HCX_API_KEY", "nv-dummy")
    with pytest.raises(ValueError, match="model"):
        HCXClient(_settings(mock_settings, "http://x", model_name="TBD"))


def test_load_settings_reads_model_block(tmp_path):
    cfg = tmp_path / "m.yaml"
    cfg.write_text("backend: hcx\nmodel: {name: HCX-007, thinking_effort: low, max_tokens: 5120}\n", encoding="utf-8")
    s = load_settings(cfg)
    assert (s.model_name, s.thinking_level, s.max_tokens) == ("HCX-007", "low", 5120)
    assert (s.base_url, s.api_key_env) == ("https://clovastudio.stream.ntruss.com", "HCX_API_KEY")
    assert load_settings(cfg, thinking_level="none", model_name="HCX-DASH-002").model_name == "HCX-DASH-002"


def test_repo_config_defaults_to_hcx():
    s = load_settings()
    assert s.backend == "hcx" and s.model_name.startswith("HCX-")
