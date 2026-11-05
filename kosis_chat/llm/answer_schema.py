"""Re-export shim (T1-28 S6): the B0 module moved to kosis_chat/legacy/answer_schema.py.

Every name the old module had is re-exported by name (private ones too: `import *` would drop them), the same
objects, so callers of the old path (tests, eval, Phase 2) need no change."""

from kosis_chat.legacy.answer_schema import (  # noqa: F401
    answer_model,
    BaseModel,
    CompactAnswer,
    ConfigDict,
    Field,
    LLMAnswer,
    model_validator,
    Outcome,
    PLAIN_TEXT,
    PROJECT_ROOT,
    RefusalReason,
    sys,
)
