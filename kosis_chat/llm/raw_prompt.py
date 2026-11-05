"""Re-export shim (T1-28 S6): the B0 module moved to kosis_chat/legacy/raw_prompt.py.

Every name the old module had is re-exported by name (private ones too: `import *` would drop them), the same
objects, so callers of the old path (tests, eval, Phase 2) need no change."""

from kosis_chat.legacy.raw_prompt import (  # noqa: F401
    SYSTEM_PROMPT,
    user_prompt,
    USER_TEMPLATE,
    VERSION,
)
