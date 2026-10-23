"""Load settings from configs/model.yaml and secrets from .env, with paths relative to the project root."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "model.yaml"
DEFAULT_BASE_URL = "https://clovastudio.stream.ntruss.com"


@dataclass(frozen=True)
class Settings:
    backend: str = "mock"
    prompt_version: str = "baseline-v1"
    serialization: str = "long"  # long | wide (kosis_chat/data/serializer.py)
    model_name: str = "TBD"
    thinking_level: str | None = None  # HCX-007 thinking.effort
    temperature: float = 0.0
    max_tokens: int = 1024
    max_input_tokens: int | None = None
    timeout_seconds: float = 60.0
    base_url: str = DEFAULT_BASE_URL
    api_key_env: str = "HCX_API_KEY"
    mock_response: str = "[mock] answer"
    eval_runs: int = 1

    def model_params(self) -> dict[str, Any]:
        """Parameters recorded in eval manifests."""
        return {
            "backend": self.backend,
            "model_name": self.model_name,
            "thinking_level": self.thinking_level,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "max_input_tokens": self.max_input_tokens,
            "timeout_seconds": self.timeout_seconds,
            "base_url": self.base_url,
        }


def resolve_path(path: str | Path) -> Path:
    """Resolve a path relative to the project root unless it is absolute."""
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p


def load_settings(path: str | Path | None = None, **overrides: Any) -> Settings:
    """Read the YAML config; keyword overrides (e.g. backend="mock") win over the file, None means no override."""
    config_path = resolve_path(path) if path else DEFAULT_CONFIG
    data: dict[str, Any] = {}
    if config_path.exists():
        data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    model = data.get("model") or {}
    values: dict[str, Any] = {
        "backend": data.get("backend", "mock"),
        "prompt_version": data.get("prompt_version", "baseline-v1"),
        "serialization": data.get("serialization", "long"),
        "model_name": model.get("name", "TBD"),
        "thinking_level": model.get("thinking_effort"),
        "temperature": float(model.get("temperature", 0)),
        "max_tokens": int(model.get("max_tokens", 1024)),
        "max_input_tokens": model.get("max_input_tokens"),
        "timeout_seconds": float(model.get("timeout_seconds", 60)),
        "base_url": model.get("base_url", DEFAULT_BASE_URL),
        "api_key_env": model.get("api_key_env", "HCX_API_KEY"),
        "mock_response": (data.get("mock") or {}).get("response", "[mock] answer"),
        "eval_runs": int((data.get("eval") or {}).get("runs", 1)),
    }
    values.update({k: v for k, v in overrides.items() if v is not None})
    return Settings(**values)


def load_env(path: str | Path | None = None) -> None:
    """Load .env into os.environ (existing variables are kept). Never print values."""
    try:
        from dotenv import load_dotenv
    except ImportError:  # optional for offline tests
        return
    load_dotenv(resolve_path(path) if path else PROJECT_ROOT / ".env", override=False)


def get_secret(name: str) -> str | None:
    """Return a secret from the environment, or None when unset."""
    return os.environ.get(name) or None
