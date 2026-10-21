"""Load settings from configs/model.yaml and secrets from .env, with paths relative to the project root."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "model.yaml"


@dataclass(frozen=True)
class Settings:
    backend: str = "mock"
    prompt_version: str = "baseline-v1"
    model_name: str = "TBD"
    temperature: float = 0.0
    max_tokens: int = 1024
    max_input_tokens: int | None = None
    timeout_seconds: float = 60.0
    mock_response: str = "[mock] answer"
    eval_runs: int = 1
    base_url: str | None = None
    api_key_env: str = "PRETEST_API_KEY"
    thinking_level: str | None = None
    raw: dict[str, Any] = field(default_factory=dict, compare=False, repr=False)

    def model_params(self) -> dict[str, Any]:
        """Parameters recorded in eval manifests."""
        return {
            "backend": self.backend,
            "model_name": self.model_name,
            "temperature": None if self.backend == "gemini" else self.temperature,  # not sent to Gemini 3.x
            "max_tokens": self.max_tokens,
            "max_input_tokens": self.max_input_tokens,
            "timeout_seconds": self.timeout_seconds,
            "base_url": self.base_url,
            "thinking_level": self.thinking_level,
        }


def resolve_path(path: str | Path) -> Path:
    """Resolve a path relative to the project root unless it is absolute."""
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p


def load_settings(path: str | Path | None = None, **overrides: Any) -> Settings:
    """Read the YAML config; keyword overrides (e.g. backend="mock") win over the file."""
    config_path = resolve_path(path) if path else DEFAULT_CONFIG
    data: dict[str, Any] = {}
    if config_path.exists():
        data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    model = data.get("model") or {}
    values: dict[str, Any] = {
        "backend": data.get("backend", "mock"),
        "prompt_version": data.get("prompt_version", "baseline-v1"),
        "model_name": model.get("name", "TBD"),
        "temperature": float(model.get("temperature", 0)),
        "max_tokens": int(model.get("max_tokens", 1024)),
        "max_input_tokens": model.get("max_input_tokens"),
        "timeout_seconds": float(model.get("timeout_seconds", 60)),
        "mock_response": (data.get("mock") or {}).get("response", "[mock] answer"),
        "eval_runs": int((data.get("eval") or {}).get("runs", 1)),
    }
    overrides = {k: v for k, v in overrides.items() if v is not None}
    if overrides.get("backend", values["backend"]) == "openai_compat":
        # Pre-test backend keeps its own model block so the HCX settings stay untouched.
        compat = data.get("openai_compat") or {}
        values["model_name"] = compat.get("model") or "TBD"
        values["base_url"] = compat.get("base_url")
        values["api_key_env"] = compat.get("api_key_env", "PRETEST_API_KEY")
    if overrides.get("backend", values["backend"]) == "gemini":
        # Gemini 3.x drops temperature; thinking_level replaces it and is recorded in manifests.
        gem = data.get("gemini") or {}
        values["model_name"] = gem.get("model") or "TBD"
        values["api_key_env"] = gem.get("api_key_env", "GEMINI_API_KEY")
        values["thinking_level"] = gem.get("thinking_level")
        if gem.get("max_tokens"):
            values["max_tokens"] = int(gem["max_tokens"])
    values.update(overrides)
    return Settings(raw=data, **values)


def load_env(path: str | Path | None = None) -> None:
    """Load .env into os.environ (existing variables are kept). Never print values."""
    try:
        from dotenv import load_dotenv
    except ImportError:  # optional for offline tests
        return
    load_dotenv(resolve_path(path) if path else PROJECT_ROOT / ".env", override=False)


def get_secret(name: str) -> str | None:
    """Return a secret from the environment, or None when unset."""
    value = os.environ.get(name)
    return value or None
