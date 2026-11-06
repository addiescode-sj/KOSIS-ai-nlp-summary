"""Wrap the free-text Answer in the shared StatAnswer contract (src/kosis_ai/schemas.py, T0-03), stage 1 of adoption.

The model's sentence is kept as one TextPart and numbers are not Facts yet (T1-07 / T1-11 move them into StatResult),
so every answer built here records numbers_origin="model": its numbers are the model's and unverified.
A text answer is "answered"; refusals and ask-backs come from the model's JSON or the rule verdict (T1-08, --judge rules).
An Answer the C pipeline built (Answer.stat_answer) is returned as it is.

T1-28 S6: moved here from kosis_chat/contract.py (B0 text handling: strip_model_source, render_text, to_stat_answer);
the source helpers are kosis_chat/render/source.py (S3a). The old path re-exports every name.
"""

from __future__ import annotations

import re
import sys
from typing import TYPE_CHECKING
from urllib.parse import urlsplit

from kosis_chat.config import PROJECT_ROOT

try:
    from kosis_ai.schemas import Refusal, Source, StatAnswer, StatQuery, TextPart
except ImportError:  # `python -m eval.run_eval` has no src/ on the path until T0-01 makes kosis_ai an installed package
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
    from kosis_ai.schemas import Refusal, Source, StatAnswer, StatQuery, TextPart

from kosis_chat.render.source import _kosis_host, to_source

if TYPE_CHECKING:  # only the annotation of to_stat_answer; no import of the chatbot at run time (T1-28 S2b)
    from kosis_chat.legacy.chatbot import Answer


_SOURCE_LINE = re.compile(r"^[ \t>*#+-]*출처[ \t*_]*[:：]")
_SOURCE_LINK = re.compile(
    r"(?:링크[ \t]*[:：][ \t]*)?"
    r"(?:\(?https?://[^\s)]+\)?|\(?\[[^\]\n]+\]\(https?://[^\s)]+\)\)?)"
)
_SEPARATOR = re.compile(r"[ \t]*(?:-{3,}|\*{3,}|_{3,})[ \t]*")


def _is_kosis_link_line(line: str) -> bool:
    link = re.sub(r"^[ \t]*(?:>[ \t]*)*(?:[-*+][ \t]+)?", "", line).strip()
    if _SOURCE_LINK.fullmatch(link) is None:
        return False
    urls = re.findall(r"https?://[^\s)]+", link)
    try:
        return bool(urls) and all(_kosis_host(urlsplit(url).hostname) for url in urls)
    except ValueError:
        return False


def strip_model_source(text: str) -> str:
    """Remove standalone source blocks while preserving body claims and unrelated links."""
    kept: list[str] = []
    in_source = False
    removed = False
    source_gap = False
    for line in text.splitlines():
        if _SOURCE_LINE.match(line):
            # A separator belongs to a source block only when it directly precedes its label.
            previous = len(kept) - 1
            while previous >= 0 and not kept[previous].strip():
                previous -= 1
            if previous >= 0 and _SEPARATOR.fullmatch(kept[previous]):
                del kept[previous:]
            in_source = removed = source_gap = True
            continue
        if in_source:
            if _is_kosis_link_line(line):
                continue
            in_source = False
            if _SEPARATOR.fullmatch(line):
                continue
        if source_gap and not line.strip() and kept and not kept[-1].strip():
            continue
        kept.append(line)
        if line.strip():
            source_gap = False
    if not removed:
        return text.rstrip()
    return "\n".join(kept).rstrip()


def render_text(body: str, source: Source | None) -> str:
    """Append one metadata-built source line to a cleaned body."""
    body = strip_model_source(body)
    if source is None:
        return body
    citation = f"출처: {source.citation()}"
    return f"{body}\n\n{citation}" if body else citation


def to_stat_answer(question: str, answer: Answer, *, org_id: str | None = None) -> StatAnswer | None:
    """None when the call failed: a timeout or an API error is a run failure, not an answer."""
    if not answer.ok:
        return None
    if answer.stat_answer is not None:  # built by the C pipeline from computed facts
        return answer.stat_answer
    parsed = answer.parsed
    if answer.answer_format == "json" and parsed is None:
        raise ValueError("JSON answer has no validated payload")
    source = None if parsed and parsed.refusal_reason == "out_of_scope" else to_source(answer.table_meta, org_id=org_id)
    # Defaults the rules chose (T1-08) are the record; the model's own list is kept only when the rules chose none.
    # compact-v1 stays frozen without them; compact-v2 keeps the rules' choice (the model never states defaults there).
    judged_defaults = {} if answer.prompt_version == "compact-v1" else (answer.judge.get("default_applied") or {})
    if parsed is not None:
        if parsed.outcome == "answered":
            body = strip_model_source(parsed.answer)
            if not body.strip():
                raise ValueError("answered body is blank after removing source")
            return StatAnswer(
                outcome="answered", query=StatQuery(question=question), parts=[TextPart(text=body)],
                text=render_text(body, source), source=source, default_applied=judged_defaults or parsed.default_applied,
                numbers_origin="model",
            )
        if parsed.outcome == "refused":
            body = strip_model_source(parsed.refusal_message)
            if not body.strip():
                raise ValueError("refusal body is blank after removing source")
            return StatAnswer(
                outcome="refused", query=StatQuery(question=question), parts=[TextPart(text=body)],
                text=render_text(body, source), source=source,
                refusal=Refusal(reason=parsed.refusal_reason, message=body, available=parsed.available),
                numbers_origin="model",
            )
        body = strip_model_source(parsed.clarify_question)
        if not body.strip():
            raise ValueError("clarification question is blank after removing source")
        body += "\n" + "\n".join(f"- {option}" for option in parsed.clarify_options)
        return StatAnswer(
            outcome="clarify", query=StatQuery(question=question), parts=[TextPart(text=body)],
            text=render_text(body, source), source=source, clarify_options=parsed.clarify_options,
            numbers_origin="model",
        )
    body = strip_model_source(answer.text)
    if not body.strip():
        raise ValueError("answered body is blank after removing source")
    return StatAnswer(
        outcome="answered",
        query=StatQuery(question=question),
        parts=[TextPart(text=body)],
        text=render_text(body, source),
        source=source,
        default_applied=judged_defaults,
        numbers_origin="model",
    )
