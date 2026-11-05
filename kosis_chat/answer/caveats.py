"""Re-export shim (T1-28 S3a): the caveats moved to kosis_chat/compose/caveats.py.

Every name the old module had is re-exported by name (private ones too: `import *` would drop them), the same
objects, so callers of the old path (tests, eval, Phase 2) need no change."""

from kosis_chat.compose.caveats import (  # noqa: F401
    Alternatives,
    cautions,
    Caveat,
    caveats_of,
    Computed,
    _EVENT_MODEL,
    fact_label,
    _from_caution,
    _from_note,
    period_text,
    _PREFIX,
    SYMBOL_MEANING,
    TableInput,
    _when,
)
