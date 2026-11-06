"""Re-export shim (T1-28 S3a): the refusal and ask-back sentences moved to kosis_chat/judge/texts.py.

Every name the old module had is re-exported by name (private ones too: `import *` would drop them), the same
objects, so callers of the old path (tests, eval, Phase 2) need no change."""

from kosis_chat.judge.texts import (  # noqa: F401
    CLARIFY_COMPARE,
    CLARIFY_ITEM,
    CLARIFY_LABEL,
    COMPARE_OPTIONS,
    _DEFAULT_KEYS,
    default_note,
    _default_text,
    defaults,
    _Fields,
    _FORM,
    josa,
    _LATEST,
    _PAIRS,
    _PLACEHOLDER_JOSA,
    _PREVIOUS,
    _READ_LAST,
    _RECENT,
    refusal_message,
    REFUSAL_MESSAGES,
    STOCK_OVER_TIME,
    with_josa,
)
