"""Re-export shim (T1-28 S3a): the plan block builders moved to kosis_chat/analyze/blocks.py.

Every name the old module had is re-exported by name (private ones too: `import *` would drop them), the same
objects, so callers of the old path (tests, eval, Phase 2) need no change."""

from kosis_chat.analyze.blocks import (  # noqa: F401
    Agg,
    AGG_WORDS,
    AggregateStep,
    Built,
    by_period,
    CombineStep,
    compare,
    _compared,
    extreme,
    gap,
    _group,
    groups_compared,
    growth,
    Literal,
    over_periods,
    Pattern,
    period_of,
    _plan,
    PlanV3,
    QuestionType,
    RankStep,
    SelectStep,
    share,
    share_change,
    span_text,
    TermSteps,
    trend,
    value,
    vs_average,
    written,
)
