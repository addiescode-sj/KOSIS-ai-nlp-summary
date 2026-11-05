"""Re-export shim (T1-28 S3a): the reference / pick rules moved to kosis_chat/analyze/roles.py.

Every name the old module had is re-exported by name (private ones too: `import *` would drop them), the same
objects, so callers of the old path (tests, eval, Phase 2) need no change."""

from kosis_chat.analyze.roles import (  # noqa: F401
    is_province,
    label_forms,
    _MIN_WORDS,
    pick_of,
    _PROVINCE,
    re,
    _REFERENCE_AFTER,
    reference_index,
    REGION_SUFFIX,
    squash,
)
