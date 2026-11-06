"""Re-export shim (T1-28 S3a): render_parts / default_sentence / final_text moved to kosis_chat/render/text.py.

Every name the old module had is re-exported by name (private ones too: `import *` would drop them), the same
objects, so callers of the old path (tests, eval, Phase 2) need no change."""

from kosis_chat.render.text import (  # noqa: F401
    _AVG,
    _CHANGE,
    default_sentence,
    defaults,
    derived_kind,
    direction,
    Fact,
    fact_label,
    FactPart,
    final_text,
    format_fact,
    format_number,
    _HOUSEHOLD,
    is_change,
    item_label,
    josa,
    _KIND_WORDS,
    _number,
    period_text,
    render_parts,
    Source,
    span_text,
    StatResult,
    _SUM,
    SYMBOL_MEANING,
    TextPart,
    UNIT_PAREN,
    _UNIT_SUFFIX,
)
