"""Re-export shim (T1-28 S3a): verify moved to kosis_chat/compose/verify.py.

Every name the old module had is re-exported by name (private ones too: `import *` would drop them), the same
objects, so callers of the old path (tests, eval, Phase 2) need no change."""

from kosis_chat.compose.verify import (  # noqa: F401
    allowed_numbers,
    Computed,
    direction,
    _DOWN,
    FACT_ID,
    FactPart,
    is_change,
    label_digits,
    NUMBER,
    _over_time,
    re,
    TextPart,
    _UP,
    verify,
)
