"""Re-export shim (T1-28 S3b): the event-lookup-v1 prompt moved to kosis_chat/analyze/event_lookup.py.

Every name the old module had is re-exported by name (private ones too: `import *` would drop them), the same
objects, so callers of the old path (tests, eval, Phase 2) need no change."""

from kosis_chat.analyze.event_lookup import (  # noqa: F401
    BaseModel,
    ConfigDict,
    datetime,
    EventStart,
    field_validator,
    FIRST_YEAR,
    re,
    _START,
    SYSTEM_PROMPT,
    timezone,
    user_prompt,
    VERSION,
)
