"""What the stages share for one table (T1-28 S2): the cleaned TableInput, its index and its lookup caches.

StatResources is built once per table (load_resources) and held by StatPipeline. The lookups the compute stage
derives from a table (cells by key, series, the analysis lookup, the index) live in table.caches (LookupCaches), one
per TableInput object, in place of the module-global dicts keyed by id(table) they replaced; resources.caches is the
same object. Concurrency: no lock (see LookupCaches): a table may be prepared twice by two threads, with equal results.
The prepared-pipeline cache by file stays kosis_chat.api._cached (an lru_cache, Phase 2 calls cache_clear()).

LLMConfig (T1-28 S2c, D25): the model a caller injects: key, model, provider and the generation settings that
were the `model` block of the settings file removed in S2c (its values are the field defaults). The module reads no
.env, environment variable or config file for it. The key is not checked when the config is made: a provider other
than "mock" without api_key raises LLMKeyMissing when an LLM step first needs its client (the pipelines build it on
first use, kosis_chat/llm/providers.py build_client), so questions the rules refuse or ask back need no key.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, SecretStr

from kosis_chat.prepare.clean import clean
from kosis_chat.prepare.index import TableIndex, index_of
from kosis_chat.prepare.schemas import LookupCaches, TableInput

if TYPE_CHECKING:  # the Settings shim (Phase 2); only from_settings reads one
    from kosis_chat.config import Settings

# What the settings file (configs/, removed in T1-28 S2c) held; tests/test_llm_config.py keeps its values.
DEFAULT_MODEL = "HCX-DASH-002"
DEFAULT_BASE_URL = "https://clovastudio.stream.ntruss.com"
MOCK_PROVIDER = "mock"  # tests and CI only (N3)
MOCK_RESPONSE = "[mock] 표에 있는 값만으로 답변합니다. 출처: KOSIS"  # the MockClient reply of a mock run


class LLMKeyMissing(RuntimeError, ValueError):
    """A provider that needs a key has none. RuntimeError for summarize_stat callers (a failed model call), and
    ValueError as HCXClient raised for a missing key before T1-28 S2c."""


class LLMConfig(BaseModel):
    """The model a caller injects (D25). Frozen and hashable; the key is a SecretStr (never in repr, logs or dumps)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    api_key: SecretStr | None = None  # None is allowed here; LLMKeyMissing when an LLM step needs the client
    model: str  # e.g. DEFAULT_MODEL
    provider: str = "hcx"  # kosis_chat.llm.providers.PROVIDERS key: "hcx", "mock" (tests and CI only)
    temperature: float = 0.0
    max_tokens: int = 1024  # HCX-007 counts thinking tokens here; about 5120 when thinking_effort is not none
    max_input_tokens: int | None = None  # input token limit checked before calling; None = not checked
    timeout_seconds: float = 60.0
    base_url: str = DEFAULT_BASE_URL  # CLOVA Studio v3 chat-completions
    thinking_effort: str | None = "none"  # none | low | medium | high, sent to HCX-007 only (Settings.thinking_level)

    def require_key(self) -> str | None:
        """The key as text; None for the mock provider. LLMKeyMissing when another provider has none (or an empty one)."""
        if self.api_key is None or not self.api_key.get_secret_value().strip():  # an empty key is no key
            if self.provider == MOCK_PROVIDER:
                return None
            raise LLMKeyMissing(f"LLMConfig.api_key is required for provider={self.provider!r}")
        return self.api_key.get_secret_value()

    @classmethod
    def from_settings(cls, settings: Settings, api_key: str | SecretStr | None = None) -> LLMConfig:
        """The same model settings from a kosis_chat.config.Settings (Phase 2 shim); the key is passed, never read."""
        return cls(
            api_key=api_key, model=settings.model_name, provider=settings.backend, temperature=settings.temperature,
            max_tokens=settings.max_tokens, max_input_tokens=settings.max_input_tokens,
            timeout_seconds=settings.timeout_seconds, base_url=settings.base_url, thinking_effort=settings.thinking_level,
        )


@dataclass(frozen=True)
class StatResources:
    table: TableInput  # cleaned
    index: TableIndex  # index_of(table): the same object every stage gets
    org_id: str | None = None

    @property
    def caches(self) -> LookupCaches:
        return self.table.caches


def load_resources(table: TableInput, org_id: str | None = None) -> StatResources:
    """Clean the table and index it (the index is kept in the cleaned table's caches)."""
    cleaned = clean(table)
    return StatResources(table=cleaned, index=index_of(cleaned), org_id=org_id)


__all__ = ["DEFAULT_BASE_URL", "DEFAULT_MODEL", "LLMConfig", "LLMKeyMissing", "LookupCaches", "MOCK_PROVIDER",
           "MOCK_RESPONSE", "StatResources", "load_resources"]
