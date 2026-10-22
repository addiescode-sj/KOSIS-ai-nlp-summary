"""LLM clients behind one interface.

MockClient: offline tests. HCXClient: HyperCLOVA X (the evaluated model).
GeminiClient: Google Gemini through the google-genai SDK (current default for development runs).
OpenAICompatClient: pre-test only, any /chat/completions endpoint (OpenAI, Ollama, vLLM ...).
"""

from __future__ import annotations

import json
import socket
import time
import urllib.error
import urllib.request
import uuid

from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from kosis_chat.config import Settings, get_secret

# Call status values recorded with every answer and eval result.
STATUS_OK = "ok"
STATUS_TIMEOUT = "timeout"
STATUS_API_ERROR = "api_error"
STATUS_INPUT_TOO_LONG = "input_too_long"
CALL_STATUSES = (STATUS_OK, STATUS_TIMEOUT, STATUS_API_ERROR, STATUS_INPUT_TOO_LONG)


@dataclass(frozen=True)
class LLMResult:
    text: str
    status: str = STATUS_OK
    usage: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


class LLMClient(ABC):
    """Common interface; implementations must not raise for API failures, they return a status."""

    name: str = "base"

    @abstractmethod
    def generate(self, system: str, user: str) -> LLMResult:
        """Return the model's reply for one system + user prompt pair."""


class MockClient(LLMClient):
    """Deterministic client for tests: fixed text, a queue of replies, or a function of the prompt."""

    name = "mock"

    def __init__(
        self,
        response: str | Sequence[LLMResult | str] | Callable[[str, str], LLMResult | str] = "[mock] answer",
    ) -> None:
        self._response = response
        self._queue = list(response) if isinstance(response, (list, tuple)) else None
        self.calls: list[tuple[str, str]] = []

    def generate(self, system: str, user: str) -> LLMResult:
        self.calls.append((system, user))
        if callable(self._response):
            reply = self._response(system, user)
        elif self._queue is not None:
            if not self._queue:
                return LLMResult(text="", status=STATUS_API_ERROR, error="mock queue exhausted")
            reply = self._queue.pop(0)
        else:
            reply = self._response
        return reply if isinstance(reply, LLMResult) else LLMResult(text=str(reply))


class HCXClient(LLMClient):
    """HyperCLOVA X through the CLOVA Studio v3 chat-completions API (non-streaming).

    POST {base_url}/v3/chat-completions/{model} with "Authorization: Bearer <HCX_API_KEY>".
    Reasoning models (HCX-007) take maxCompletionTokens and thinking.effort; the others take maxTokens.
    Rate limits (429) and server errors (5xx) are retried with the delays in RETRY_DELAYS,
    or the server's Retry-After value when it is longer (capped at MAX_RETRY_WAIT).
    """

    name = "hcx"
    REASONING_MODELS = ("HCX-007",)
    RETRY_DELAYS = (5.0, 15.0, 30.0, 60.0)
    MAX_RETRY_WAIT = 120.0
    RETRY_HTTP_CODES = (429, 500, 502, 503, 504)

    def __init__(self, settings: Settings, sleep: Callable[[float], None] = time.sleep) -> None:
        self.settings = settings
        if not settings.model_name or settings.model_name == "TBD":
            raise ValueError("model.name (HCX model) is not set in configs/model.yaml (or pass --model)")
        if not settings.base_url:
            raise ValueError("model.base_url (CLOVA Studio URL) is not set in configs/model.yaml")
        self._key = get_secret(settings.api_key_env)
        if not self._key:
            raise ValueError(f"{settings.api_key_env} is not set in .env")
        self.url = f"{settings.base_url.rstrip('/')}/v3/chat-completions/{settings.model_name}"
        self._sleep = sleep
        self._retry_after: float | None = None

    @property
    def is_reasoning(self) -> bool:
        return self.settings.model_name.upper().startswith(self.REASONING_MODELS)

    def _body(self, system: str, user: str) -> dict[str, Any]:
        body: dict[str, Any] = {
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "temperature": self.settings.temperature,
        }
        if self.is_reasoning:
            # Thinking tokens count toward maxCompletionTokens.
            body["maxCompletionTokens"] = self.settings.max_tokens
            body["thinking"] = {"effort": self.settings.thinking_level or "none"}
        else:
            body["maxTokens"] = self.settings.max_tokens
        return body

    def _request(self, system: str, user: str) -> urllib.request.Request:
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": f"Bearer {self._key}",
            "X-NCP-CLOVASTUDIO-REQUEST-ID": uuid.uuid4().hex,
        }
        data = json.dumps(self._body(system, user)).encode("utf-8")
        return urllib.request.Request(self.url, data=data, headers=headers, method="POST")

    def generate(self, system: str, user: str) -> LLMResult:
        result = self._call(system, user)
        for delay in self.RETRY_DELAYS:
            if not (result.error or "").startswith(tuple(f"HTTP {c}" for c in self.RETRY_HTTP_CODES)):
                break
            self._sleep(min(max(delay, self._retry_after or 0.0), self.MAX_RETRY_WAIT))
            result = self._call(system, user)
        return result

    def _call(self, system: str, user: str) -> LLMResult:
        self._retry_after = None
        try:
            with urllib.request.urlopen(self._request(system, user), timeout=self.settings.timeout_seconds) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            try:
                self._retry_after = float(exc.headers.get("Retry-After", ""))
            except (TypeError, ValueError):
                self._retry_after = None
            detail = exc.read().decode("utf-8", "replace")[:300]
            low = detail.lower()
            too_long = exc.code == 400 and "token" in low and any(w in low for w in ("exceed", "limit", "too long", "초과"))
            return LLMResult(text="", status=STATUS_INPUT_TOO_LONG if too_long else STATUS_API_ERROR, error=f"HTTP {exc.code}: {detail}")
        except (TimeoutError, socket.timeout):
            return LLMResult(text="", status=STATUS_TIMEOUT, error="request timed out")
        except urllib.error.URLError as exc:
            if isinstance(exc.reason, (TimeoutError, socket.timeout)):
                return LLMResult(text="", status=STATUS_TIMEOUT, error="request timed out")
            return LLMResult(text="", status=STATUS_API_ERROR, error=f"connection failed: {exc.reason}")
        except json.JSONDecodeError as exc:
            return LLMResult(text="", status=STATUS_API_ERROR, error=f"invalid JSON response: {exc}")
        code = str((data.get("status") or {}).get("code", ""))
        result = data.get("result") or {}
        if code != "20000" or not isinstance(result.get("message"), dict):
            return LLMResult(text="", status=STATUS_API_ERROR, error=f"unexpected response: {str(data)[:300]}")
        usage = result.get("usage") or {}
        text = result["message"].get("content") or ""
        if not text:
            # With thinking on, a small maxCompletionTokens can be used up before any answer text.
            return LLMResult(text="", status=STATUS_API_ERROR, usage=usage, error=f"empty response (finishReason={result.get('finishReason')})")
        return LLMResult(text=text, usage=usage)


class OpenAICompatClient(LLMClient):
    """Pre-test client for OpenAI-compatible chat APIs. Results are never B0..B4 scores (model differs)."""

    name = "openai_compat"

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        if not settings.base_url:
            raise ValueError("openai_compat.base_url is not set in configs/model.yaml")
        if not settings.model_name or settings.model_name == "TBD":
            raise ValueError("openai_compat.model is not set in configs/model.yaml (or pass --model)")
        self.url = settings.base_url.rstrip("/") + "/chat/completions"

    def _request(self, system: str, user: str) -> urllib.request.Request:
        body = {
            "model": self.settings.model_name,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "temperature": self.settings.temperature,
            "max_tokens": self.settings.max_tokens,
        }
        headers = {"Content-Type": "application/json"}
        key = get_secret(self.settings.api_key_env)
        if key:  # local servers such as Ollama need no key
            headers["Authorization"] = f"Bearer {key}"
        return urllib.request.Request(self.url, data=json.dumps(body).encode("utf-8"), headers=headers, method="POST")

    def generate(self, system: str, user: str) -> LLMResult:
        try:
            with urllib.request.urlopen(self._request(system, user), timeout=self.settings.timeout_seconds) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:300]
            status = STATUS_INPUT_TOO_LONG if "context_length" in detail or "too long" in detail.lower() else STATUS_API_ERROR
            return LLMResult(text="", status=status, error=f"HTTP {exc.code}: {detail}")
        except (TimeoutError, socket.timeout):
            return LLMResult(text="", status=STATUS_TIMEOUT, error="request timed out")
        except urllib.error.URLError as exc:
            if isinstance(exc.reason, (TimeoutError, socket.timeout)):
                return LLMResult(text="", status=STATUS_TIMEOUT, error="request timed out")
            return LLMResult(text="", status=STATUS_API_ERROR, error=f"connection failed: {exc.reason}")
        except json.JSONDecodeError as exc:
            return LLMResult(text="", status=STATUS_API_ERROR, error=f"invalid JSON response: {exc}")
        try:
            text = data["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError):
            return LLMResult(text="", status=STATUS_API_ERROR, error=f"unexpected response: {str(data)[:300]}")
        return LLMResult(text=text, usage=data.get("usage") or {})


class GeminiClient(LLMClient):
    """Gemini via google-genai (generate_content). Temperature is not sent: Gemini 3.x replaces it with thinking_level."""

    name = "gemini"

    def __init__(self, settings: Settings, sdk_client: Any | None = None) -> None:
        self.settings = settings
        if not settings.model_name or settings.model_name == "TBD":
            raise ValueError("gemini.model is not set in configs/model.yaml (or pass --model)")
        if sdk_client is None:
            try:
                from google import genai
                from google.genai import types
            except ImportError as exc:
                raise ValueError("google-genai is not installed: pip install -r requirements.txt") from exc
            key = get_secret(settings.api_key_env)
            if not key:
                raise ValueError(f"{settings.api_key_env} is not set (.env or Colab secrets)")
            sdk_client = genai.Client(
                api_key=key, http_options=types.HttpOptions(timeout=int(settings.timeout_seconds * 1000))
            )
        self._client = sdk_client

    def _config(self, system: str) -> Any:
        from google.genai import types

        thinking = types.ThinkingConfig(thinking_level=self.settings.thinking_level) if self.settings.thinking_level else None
        return types.GenerateContentConfig(
            system_instruction=system, max_output_tokens=self.settings.max_tokens, thinking_config=thinking
        )

    def generate(self, system: str, user: str) -> LLMResult:
        try:
            from google.genai import errors
        except ImportError:  # only reachable with an injected fake client
            errors = None
        try:
            resp = self._client.models.generate_content(
                model=self.settings.model_name, contents=user, config=self._config(system)
            )
        except Exception as exc:  # SDK raises its own and httpx exception types
            if "timeout" in type(exc).__name__.lower() or isinstance(exc, TimeoutError):
                return LLMResult(text="", status=STATUS_TIMEOUT, error="request timed out")
            if errors is not None and isinstance(exc, errors.APIError):
                msg = str(exc)[:300]
                too_long = exc.code == 400 and ("token" in msg.lower() and ("exceed" in msg.lower() or "limit" in msg.lower()))
                return LLMResult(text="", status=STATUS_INPUT_TOO_LONG if too_long else STATUS_API_ERROR, error=f"HTTP {exc.code}: {msg}")
            return LLMResult(text="", status=STATUS_API_ERROR, error=f"{type(exc).__name__}: {str(exc)[:300]}")
        usage = resp.usage_metadata.model_dump(exclude_none=True) if getattr(resp, "usage_metadata", None) else {}
        text = resp.text or ""
        if not text:
            reason = None
            if getattr(resp, "candidates", None):
                reason = getattr(resp.candidates[0], "finish_reason", None)
            # Thinking tokens count toward max_output_tokens, so MAX_TOKENS with no text means the budget was spent on thinking.
            return LLMResult(text="", status=STATUS_API_ERROR, usage=usage, error=f"empty response (finish_reason={reason})")
        return LLMResult(text=text, usage=usage)


BACKENDS = ("mock", "hcx", "gemini", "openai_compat")


def build_client(settings: Settings) -> LLMClient:
    """Create the client named by settings.backend."""
    if settings.backend == "mock":
        return MockClient(settings.mock_response)
    if settings.backend == "hcx":
        return HCXClient(settings)
    if settings.backend == "gemini":
        return GeminiClient(settings)
    if settings.backend == "openai_compat":
        return OpenAICompatClient(settings)
    raise ValueError(f"unknown backend: {settings.backend!r} (expected one of {BACKENDS})")
