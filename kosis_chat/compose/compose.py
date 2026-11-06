"""compose-v2: the model writes the answer sentence around computed facts, citing them as {f1}, {f2} (T1-09).

The model sees each fact's value (to choose words such as 늘었다) but never types a number: code puts the numbers in
(kosis_chat/render/text.py). Its reply {"answer": "...{f1}..."} is split into TextPart / FactPart.
v2 (S6): the facts shown beside the answer are tagged [보조] (another scenario, a newer period), and the caveats code
shows with the answer are listed under [주의] so the sentence does not contradict or repeat them.
T1-28 S3b: moved from llm/compose.py (which re-exports it; pipeline.VERSION is its VERSION); the prompts are
compose/prompts/compose-v2*.md.
"""

from __future__ import annotations

import re

from kosis_chat.shared.labels import fact_label  # isort: skip  # first, as answer.render was: shared puts src/ on the path
from kosis_chat.shared.numbers import derived_kind, direction, format_fact, is_change  # isort: skip
from kosis_chat.shared.patterns import FACT_ID  # isort: skip
from kosis_chat.shared.periods import period_text  # isort: skip
from kosis_chat.compose import prompts  # isort: skip
from kosis_chat.compute import Computed  # isort: skip

from kosis_ai.schemas import Fact, FactPart, TextPart  # isort: skip
from kosis_chat.compose.schemas import ComposeOutput  # isort: skip  # re-export: moved in T1-28 S1b

VERSION = prompts.version(prompts.COMPOSE)
SYSTEM_PROMPT = prompts.load(prompts.COMPOSE)
EXAMPLE = prompts.load(prompts.COMPOSE_EXAMPLE)
RETRY = prompts.load(prompts.COMPOSE_RETRY)
# '{f3}' in the model's sentence: the shared FACT_ID inside braces (T1-28 S2b; the pattern text is unchanged).
_PLACEHOLDER = re.compile(rf"\{{\s*({FACT_ID.pattern})\s*\}}")


def fact_line(fact: Fact, main: bool, context: bool = False) -> str:
    tag = "[반드시 인용]" if main else "[보조]" if context else "[참고]"
    if fact.derivation:
        what = f"{derived_kind(fact)} ({fact.derivation})"
        way = f" ({direction(fact)})" if is_change(fact) else ""
    else:
        what = f"{period_text(fact.period)} {fact_label(fact, skip_totals=False)}"
        way = ""
    note = f" — {fact.note}" if fact.note else ""
    return f"{{{fact.id}}} {tag} {what} = {format_fact(fact)}{way}{note}"


def user_prompt(question: str, computed: Computed, cautions: list[str] | None = None,
                context_ids: list[str] | None = None) -> str:
    shown = set(context_ids or [])
    lines = [fact_line(f, f.id in computed.main, f.id in shown) for f in computed.result.facts]
    extra = "\n\n[주의]\n" + "\n".join(f"- {c}" for c in cautions) if cautions else ""
    return f"[질문]\n{question}\n\n[사실 목록]\n" + "\n".join(lines) + extra + f"\n\n[출력 예]\n{EXAMPLE}"


def retry_prompt(user: str, problems: list[str]) -> str:
    return user + "\n\n" + RETRY.format(problems="\n".join(f"- {p}" for p in problems))


def to_parts(answer: str, computed: Computed) -> list:
    """'...{f1}가구...' -> [TextPart, FactPart, TextPart]; a unit the model repeats after a placeholder is dropped."""
    by_id = {f.id: f for f in computed.result.facts}
    parts: list = []
    pieces = _PLACEHOLDER.split(answer)
    for n, piece in enumerate(pieces):
        if n % 2:
            parts.append(FactPart(fact_id=piece))
            continue
        if parts and isinstance(parts[-1], FactPart) and (fact := by_id.get(parts[-1].fact_id)) and fact.unit:
            if piece.startswith(fact.unit):
                piece = piece[len(fact.unit):]
        if piece:
            parts.append(TextPart(text=piece))
    return parts
