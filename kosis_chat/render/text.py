"""Numbers in the answer are written by code (T1-11): a FactPart becomes '123,456가구', '12.3%', '1.5배' or a symbol.

render_parts joins the parts; final_text adds the lines code always writes (default conditions, notes, source).
Those code lines are not parts: parts are what the model chose and what the grader checks for unsupported numbers.

T1-28 S2b: the number, period and label helpers moved to kosis_chat/shared/{numbers,periods,labels}.py and josa to
kosis_chat/shared/texts.py; every old name is re-exported here (the same objects).
T1-28 S3a: moved here from answer/render.py (a re-export shim now).
"""

from __future__ import annotations

from kosis_chat.shared import defaults
from kosis_chat.shared.labels import _HOUSEHOLD, fact_label, item_label  # noqa: F401
from kosis_chat.shared.numbers import (  # noqa: F401
    _AVG,
    _CHANGE,
    _KIND_WORDS,
    _SUM,
    SYMBOL_MEANING,
    _number,
    derived_kind,
    direction,
    format_fact,
    format_number,
    is_change,
)
from kosis_chat.shared.patterns import UNIT_PAREN
from kosis_chat.shared.periods import period_text, span_text  # noqa: F401
from kosis_chat.shared.texts import josa  # noqa: F401  (was messages.josa: the same function)

from kosis_ai.schemas import Fact, FactPart, Source, StatResult, TextPart  # noqa: F401  # isort: skip

_UNIT_SUFFIX = UNIT_PAREN  # old name of the trailing '(...)' pattern


def render_parts(parts: list, result: StatResult) -> str:
    out = []
    for part in parts:
        if isinstance(part, FactPart):
            out.append(format_fact(result.get(part.fact_id)))
        elif isinstance(part, TextPart):
            out.append(part.text)
    return "".join(out).strip()


def default_sentence(applied: dict[str, str]) -> str:
    """The conditions the rules chose, in one sentence written by code."""
    out = []
    for key, value in applied.items():
        parsed = defaults.parse_period(value) if key == defaults.PERIOD_KEY else None
        if parsed is not None and parsed[0] == "previous":
            out.append(f"질문에 시점이 없어 가장 최근 시점인 {parsed[1][0]}년을 그 직전 시점 {parsed[1][1]}년과 비교합니다.")
        elif parsed is not None and parsed[0] == "latest":
            out.append(f"질문에 시점이 없어 가장 최근 시점인 {parsed[1][0]}년 기준으로 답합니다.")
        elif parsed is not None and parsed[0] == "recent":
            out.append(f"질문에 시점이 없어 최근 {parsed[1][0]}개 시점({parsed[1][1]}년)으로 답합니다.")
        else:
            out.append(f"질문에 없는 조건은 {defaults.DEFAULT_KEYS.get(key, key)} {value} 기준으로 답합니다.")
    return " ".join(out)


def final_text(body: str, source: Source | None, applied: dict[str, str] | None = None, notes: list[str] | None = None) -> str:
    """The default sentence, the body, one '참고:' line per note (S6: the caveats' texts), then the source."""
    lines = [s for s in (default_sentence(applied or {}), body) if s]
    lines += [f"참고: {n}" for n in dict.fromkeys(notes or []) if n]
    text = "\n".join(lines)
    return f"{text}\n\n출처: {source.citation()}" if source else text
