"""Re-export shim (T1-28 S6): the B0 module moved to kosis_chat/legacy/raw.py.

Every name the old module had is re-exported by name (private ones too: `import *` would drop them), the same
objects, so callers of the old path (tests, eval, Phase 2) need no change."""

from kosis_chat.legacy.raw import (  # noqa: F401
    Answer,
    Any,
    build_client,
    DECISION_METADATA,
    estimate_tokens,
    hashlib,
    LLMClient,
    load_settings,
    raw_prompt,
    RawRunner,
    scope,
    serializer,
    Settings,
    STATUS_API_ERROR,
    STATUS_INPUT_TOO_LONG,
    STATUS_TIMEOUT,
    TableInput,
    time,
    TraceWriteError,
)
