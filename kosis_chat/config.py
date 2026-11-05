"""Settings for the pipelines and Phase 2 (a shim since T1-28 S2c), with work paths relative to the work root.

configs/model.yaml is gone (D25): load_settings() builds the same Settings it read from that file out of the LLMConfig
field defaults (kosis_chat/schemas/resources.py) and the constants below; tests/test_llm_config.py keeps the old
file's values. The module reads no .env and no environment variable for the model: callers inject an LLMConfig.
Kept for Phase 2 until it moves to LLMConfig (plans/T1-28.md S2c, S6):
- Settings / load_settings(path, **overrides): a path still reads that YAML file the old way (deprecated).
- load_env / get_secret: deprecated (DeprecationWarning); no module code calls them.
- a Settings passed where a client is built (HCXClient(settings), build_client(settings)) reads the key from the
  shell environment variable Settings.api_key_env names, as before (env_llm_config, the one such read).

The work root is KOSIS_WORKSPACE when set, else the project root (src/kosis_ai/paths.py, plans/T1-27.md).
"""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import yaml

try:
    from kosis_ai.paths import workspace_root
except ImportError:  # same fallback as kosis_chat/contract.py: src/ is not on the path for `python -m ...`
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from kosis_ai.paths import workspace_root

from kosis_chat.schemas.resources import DEFAULT_BASE_URL, DEFAULT_MODEL, MOCK_RESPONSE, LLMConfig  # isort: skip

PROJECT_ROOT = Path(__file__).resolve().parents[1]  # code: the source snapshot

# What load_settings() returns besides the LLMConfig fields (the rest of the old configs/model.yaml).
DEFAULT_BACKEND = "hcx"
DEFAULT_PROMPT_VERSION = "baseline-v1"
DEFAULT_SERIALIZATION = "long"  # long (one value per row) | wide (periods as columns), kosis_chat/data/serializer.py
DEFAULT_KEY_ENV = "HCX_API_KEY"
DEFAULT_EVAL_RUNS = 1  # runs per question; keep fixed across B0..B4


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
    """Resolve a path relative to the work root unless it is absolute."""
    p = Path(path)
    return p if p.is_absolute() else workspace_root() / p


def _llm_default(name: str) -> Any:
    return LLMConfig.model_fields[name].default


def load_settings(path: str | Path | None = None, **overrides: Any) -> Settings:
    """The default Settings (the old configs/model.yaml values); keyword overrides (e.g. backend="mock") win, None
    means no override. A path reads that YAML file as before T1-28 S2c (deprecated: pass an LLMConfig instead)."""
    config_path = resolve_path(path) if path else None
    if config_path is not None and config_path.exists():
        values = _read_yaml(config_path)
    else:
        if config_path is not None:  # e.g. the deleted configs/model.yaml: its values are the defaults below
            warnings.warn(f"load_settings: {config_path} does not exist; using the built-in defaults (LLMConfig)",
                          DeprecationWarning, stacklevel=2)
        values = {
            "backend": DEFAULT_BACKEND,
            "prompt_version": DEFAULT_PROMPT_VERSION,
            "serialization": DEFAULT_SERIALIZATION,
            "model_name": DEFAULT_MODEL,
            "thinking_level": _llm_default("thinking_effort"),
            "temperature": float(_llm_default("temperature")),
            "max_tokens": int(_llm_default("max_tokens")),
            "max_input_tokens": _llm_default("max_input_tokens"),
            "timeout_seconds": float(_llm_default("timeout_seconds")),
            "base_url": _llm_default("base_url"),
            "api_key_env": DEFAULT_KEY_ENV,
            "mock_response": MOCK_RESPONSE,
            "eval_runs": DEFAULT_EVAL_RUNS,
        }
    values.update({k: v for k, v in overrides.items() if v is not None})
    return Settings(**values)


def _read_yaml(config_path: Path) -> dict[str, Any]:
    """A settings file in the old configs/model.yaml layout (Phase 2 shim), read exactly as before."""
    warnings.warn("load_settings(path) is deprecated: pass an LLMConfig (kosis_chat.schemas.resources)",
                  DeprecationWarning, stacklevel=3)
    data: dict[str, Any] = {}
    if config_path.exists():
        data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    model = data.get("model") or {}
    return {
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


def settings_from_llm(llm: LLMConfig, **overrides: Any) -> Settings:
    """Settings that carry an LLMConfig's model settings (for the code that still reads Settings); no key."""
    return replace(
        load_settings(**overrides), backend=llm.provider, model_name=llm.model, thinking_level=llm.thinking_effort,
        temperature=llm.temperature, max_tokens=llm.max_tokens, max_input_tokens=llm.max_input_tokens,
        timeout_seconds=llm.timeout_seconds, base_url=llm.base_url,
    )


def env_llm_config(settings: Settings) -> LLMConfig:
    """Phase 2 shim: the LLMConfig of a Settings, its key read from the shell environment variable it names (None
    when unset). Only the Settings paths call this (HCXClient(settings)); LLMConfig callers pass the key."""
    return LLMConfig.from_settings(settings, api_key=os.environ.get(settings.api_key_env) or None)


def load_env(path: str | Path | None = None) -> None:
    """Deprecated (T1-28 S2c): load .env into os.environ (existing variables are kept). Never print values.

    Phase 2 still calls it; Phase 1 does not (HCX_API_KEY is a shell environment variable)."""
    warnings.warn("kosis_chat.config.load_env is deprecated: set HCX_API_KEY in the shell and pass an LLMConfig",
                  DeprecationWarning, stacklevel=2)
    try:
        from dotenv import load_dotenv
    except ImportError:  # optional for offline tests
        return
    load_dotenv(resolve_path(path) if path else workspace_root() / ".env", override=False)


def get_secret(name: str) -> str | None:
    """Deprecated (T1-28 S2c): a secret from the environment, or None when unset. Phase 2 shim."""
    warnings.warn("kosis_chat.config.get_secret is deprecated: pass the key in LLMConfig.api_key",
                  DeprecationWarning, stacklevel=2)
    return os.environ.get(name) or None
