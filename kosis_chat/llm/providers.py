"""Provider table (T1-28 S2c, E5): LLMConfig.provider -> client class, so a new provider is one register call.

build_client(config) is where a missing key surfaces: a provider other than "mock" without LLMConfig.api_key raises
LLMKeyMissing. The pipelines call it when an LLM step first needs the client, never when they are made, so the
questions the rules refuse or ask back are answered without a key. A Settings (Phase 2 shim) still builds the old way
(kosis_chat.llm.model.build_client).
"""

from __future__ import annotations

from kosis_chat.config import Settings
from kosis_chat.llm import model
from kosis_chat.llm.model import HCXClient, LLMClient, MockClient
from kosis_chat.schemas.resources import LLMConfig, LLMKeyMissing

PROVIDERS: dict[str, type[LLMClient]] = {"hcx": HCXClient, "mock": MockClient}


def register_provider(name: str, cls: type[LLMClient]) -> type[LLMClient]:
    """Add (or replace) a provider; cls.from_config(config) makes its client (LLMClient's default: cls(config))."""
    if not (isinstance(cls, type) and issubclass(cls, LLMClient)):
        raise TypeError(f"provider {name!r}: {cls!r} is not an LLMClient subclass")
    PROVIDERS[name] = cls
    return cls


def build_client(config: LLMConfig | Settings) -> LLMClient:
    """The client of config.provider. LLMKeyMissing when the provider needs a key and has none."""
    if isinstance(config, Settings):
        return model.build_client(config)
    cls = PROVIDERS.get(config.provider)
    if cls is None:
        raise ValueError(f"unknown provider: {config.provider!r} (registered: {sorted(PROVIDERS)})")
    config.require_key()
    return cls.from_config(config)


__all__ = ["PROVIDERS", "LLMKeyMissing", "build_client", "register_provider"]
