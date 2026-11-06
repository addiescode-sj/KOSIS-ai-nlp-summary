"""Re-export shim (T1-28 S6): the B0 module moved to kosis_chat/legacy/prompts.py.

Every name the old module had is re-exported by name (private ones too: `import *` would drop them), the same
objects, so callers of the old path (tests, eval, Phase 2) need no change."""

from kosis_chat.legacy.prompts import (  # noqa: F401
    build_user_prompt,
    FORM_TEMPLATE,
    get_system_prompt,
    NUMBER_RULE,
    resolve_answer_prompt,
    STRUCTURED_PROMPT_VERSIONS,
    SYSTEM_PROMPTS,
    USER_TEMPLATE,
)
