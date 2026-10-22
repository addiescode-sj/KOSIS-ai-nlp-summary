"""Shared pytest fixtures: made-up tables, offline settings and a local fake HTTP server."""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

from kosis_chat.config import load_settings

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def fixtures() -> Path:
    return FIXTURES


@pytest.fixture
def mock_settings():
    return load_settings(backend="mock")


class FakeServer:
    """Answers every POST with `status` / `body` / `headers` (changeable between requests) and records requests."""

    def __init__(self, status=200, body=None, headers=None, delay=0.0):
        self.status, self.body, self.headers, self.delay = status, body, headers or {}, delay
        self.requests: list[dict] = []
        self.request_headers: list[dict] = []
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                length = int(self.headers["Content-Length"])
                outer.requests.append(json.loads(self.rfile.read(length)))
                outer.request_headers.append(dict(self.headers))
                if outer.delay:
                    threading.Event().wait(outer.delay)
                payload = json.dumps(outer.body).encode()
                self.send_response(outer.status)
                self.send_header("Content-Type", "application/json")
                for name, value in outer.headers.items():
                    self.send_header(name, value)
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, *args):
                pass

        self.httpd = HTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()
        self.base_url = f"http://127.0.0.1:{self.httpd.server_port}"

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
