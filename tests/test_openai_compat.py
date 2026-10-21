"""OpenAICompatClient against a local fake /chat/completions server (no network, no key)."""

from __future__ import annotations

import json
import threading
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from kosis_chat.chatbot import Chatbot
from kosis_chat.config import load_settings
from kosis_chat.data.loader import load_table
from kosis_chat.llm.model import OpenAICompatClient, build_client


@pytest.fixture
def table(fixtures):
    return load_table(fixtures / "pop_a.csv")


class FakeServer:
    def __init__(self, status=200, body=None, delay=0.0):
        self.status, self.body, self.delay = status, body, delay
        self.requests: list[dict] = []
        self.headers: list[dict] = []
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                length = int(self.headers["Content-Length"])
                outer.requests.append(json.loads(self.rfile.read(length)))
                outer.headers.append(dict(self.headers))
                if outer.delay:
                    threading.Event().wait(outer.delay)
                payload = json.dumps(outer.body).encode()
                self.send_response(outer.status)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, *args):
                pass

        self.httpd = HTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()
        self.base_url = f"http://127.0.0.1:{self.httpd.server_port}/v1"

    def close(self):
        self.httpd.shutdown()


@pytest.fixture
def server_factory():
    servers = []

    def make(**kwargs):
        servers.append(FakeServer(**kwargs))
        return servers[-1]

    yield make
    for s in servers:
        s.close()


def _settings(mock_settings, base_url, **kw):
    return replace(mock_settings, backend="openai_compat", base_url=base_url, model_name="fake-model", **kw)


OK_BODY = {"choices": [{"message": {"content": "2023년 여자 인구는 100명입니다. 출처: 테스트"}}], "usage": {"total_tokens": 9}}


def test_ok_reply_and_request_shape(table, mock_settings, server_factory, monkeypatch):
    monkeypatch.setenv("PRETEST_API_KEY", "dummy")
    srv = server_factory(body=OK_BODY)
    answer = Chatbot(table, settings=_settings(mock_settings, srv.base_url)).ask("여자 인구는?")
    assert answer.status == "ok" and "100명" in answer.text
    req = srv.requests[0]
    assert req["model"] == "fake-model" and req["temperature"] == 0
    assert [m["role"] for m in req["messages"]] == ["system", "user"]
    assert srv.headers[0]["Authorization"] == "Bearer dummy"


def test_no_key_sends_no_auth_header(table, mock_settings, server_factory, monkeypatch):
    monkeypatch.delenv("PRETEST_API_KEY", raising=False)
    srv = server_factory(body=OK_BODY)
    Chatbot(table, settings=_settings(mock_settings, srv.base_url)).ask("q")
    assert "Authorization" not in srv.headers[0]


def test_http_errors_become_statuses(table, mock_settings, server_factory):
    srv = server_factory(status=500, body={"error": "boom"})
    answer = Chatbot(table, settings=_settings(mock_settings, srv.base_url)).ask("q")
    assert answer.status == "api_error" and "HTTP 500" in answer.error
    srv2 = server_factory(status=400, body={"error": {"code": "context_length_exceeded"}})
    assert Chatbot(table, settings=_settings(mock_settings, srv2.base_url)).ask("q").status == "input_too_long"


def test_timeout_and_connection_failure(table, mock_settings, server_factory):
    srv = server_factory(body=OK_BODY, delay=1.0)
    answer = Chatbot(table, settings=_settings(mock_settings, srv.base_url, timeout_seconds=0.2)).ask("q")
    assert answer.status == "timeout"
    dead = Chatbot(table, settings=_settings(mock_settings, "http://127.0.0.1:9/v1")).ask("q")
    assert dead.status == "api_error"


def test_config_requires_base_url_and_model(mock_settings):
    with pytest.raises(ValueError, match="base_url"):
        build_client(replace(mock_settings, backend="openai_compat", base_url=None, model_name="m"))
    with pytest.raises(ValueError, match="model"):
        OpenAICompatClient(replace(mock_settings, backend="openai_compat", base_url="http://x/v1", model_name="TBD"))


def test_load_settings_uses_compat_block_without_touching_hcx(tmp_path):
    cfg = tmp_path / "m.yaml"
    cfg.write_text(
        "backend: hcx\nmodel: {name: HCX-X}\nopenai_compat: {base_url: 'http://h/v1', model: small, api_key_env: K}\n",
        encoding="utf-8",
    )
    assert load_settings(cfg).model_name == "HCX-X"
    s = load_settings(cfg, backend="openai_compat")
    assert (s.model_name, s.base_url, s.api_key_env) == ("small", "http://h/v1", "K")
    assert load_settings(cfg, backend="openai_compat", model_name="other").model_name == "other"


def test_cli_reports_missing_compat_config(fixtures, capsys):
    from kosis_chat import cli

    assert cli.main(["--table", str(fixtures / "pop_a.csv"), "--backend", "openai_compat", "-q", "x"]) == 2
    assert "base_url" in capsys.readouterr().err
