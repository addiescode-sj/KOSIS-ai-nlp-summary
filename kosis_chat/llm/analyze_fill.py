"""Re-export shim (T1-28 S3b): the analyze-fill-v2 prompt and checks moved to kosis_chat/analyze/fill.py.

Every name the old module had is re-exported by name (private ones too: `import *` would drop them), the same
objects, so callers of the old path (tests, eval, Phase 2) need no change."""

from kosis_chat.analyze.fill import (  # noqa: F401
    AnalyzeFill,
    apply,
    asked_axes,
    Axis,
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    label_forms,
    Literal,
    menu,
    _menu_codes,
    MENU_LIMIT,
    _move,
    PERIOD_AXIS,
    Plan,
    QuestionType,
    resolve_code,
    RETRY,
    retry_prompt,
    squash,
    SYSTEM_PROMPT,
    TableIndex,
    user_prompt,
    VERSION,
)
