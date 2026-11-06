"""Re-export shim (T1-28 S3b): the compose-v2 prompt, its user prompt and to_parts moved to kosis_chat/compose/compose.py.

Every name the old module had is re-exported by name (private ones too: `import *` would drop them), the same
objects, so callers of the old path (tests, eval, Phase 2) need no change."""

from kosis_chat.compose.compose import (  # noqa: F401
    ComposeOutput,
    Computed,
    derived_kind,
    direction,
    EXAMPLE,
    Fact,
    FACT_ID,
    fact_label,
    fact_line,
    FactPart,
    format_fact,
    is_change,
    period_text,
    _PLACEHOLDER,
    re,
    RETRY,
    retry_prompt,
    SYSTEM_PROMPT,
    TextPart,
    to_parts,
    user_prompt,
    VERSION,
)
