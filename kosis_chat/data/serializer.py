"""Re-export shim (T1-28 S6): the B0 module moved to kosis_chat/legacy/serializer.py.

Every name the old module had is re-exported by name (private ones too: `import *` would drop them), the same
objects, so callers of the old path (tests, eval, Phase 2) need no change."""

from kosis_chat.legacy.serializer import (  # noqa: F401
    display_value,
    _escape,
    MAX_PERIOD_COLUMNS,
    meta_header,
    serialize,
    STYLES,
    TableInput,
    to_markdown,
    to_wide_markdown,
    _TRAILING_ZEROS,
)
