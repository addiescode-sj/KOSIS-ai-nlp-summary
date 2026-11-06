"""render nodes (T1-28 S3a): the finished StatAnswer, its text written by code around the answer's parts.

Pure steps: (RenderIn | AnswerFinalIn, StatResources) -> RenderOut. StatPipeline.ask calls them (no span, as before).
"""

from __future__ import annotations

from kosis_chat.judge.question import CAUSAL_WORDS
from kosis_chat.judge.texts import REFUSAL_MESSAGES
from kosis_chat.render.schemas import AnswerFinalIn, RenderIn, RenderOut
from kosis_chat.render.source import to_source
from kosis_chat.render.table import table_part
from kosis_chat.render.text import final_text, render_parts
from kosis_chat.schemas.resources import StatResources

from kosis_ai.schemas import ClarifyChoice, Refusal, StatAnswer, StatQuery  # isort: skip


def causal_note(question: str) -> str | None:
    """'추이와 그렇게 된 원인을 알려줘': the numbers are answered, the cause part is declined in a note (T1-14)."""
    return REFUSAL_MESSAGES["causal"] if CAUSAL_WORDS.search(question) else None


def step_render(inp: RenderIn, res: StatResources) -> RenderOut:
    """The answered StatAnswer: the parts rendered, the default sentence, '참고:' lines, the source and table.

    In: RenderIn(question, query, computed, parts, verified, source, default_applied, caveats, shown, context_ids)
    Out: RenderOut(stat: StatAnswer, outcome "answered")
    Side effects: none
    Node: render (no span)
    """
    body = render_parts(inp.parts, inp.computed.result)
    table = table_part(inp.computed)
    stat = StatAnswer(
        outcome="answered", query=inp.query, result=inp.computed.result, parts=inp.parts,
        text=final_text(body, inp.source, inp.default_applied, [*inp.shown, causal_note(inp.question)]),
        default_applied=inp.default_applied, source=inp.source,
        numbers_origin="code", numbers_verified=inp.verified, defaults_verified=True,  # rule_plan computed with them
        caveats=inp.caveats, context_fact_ids=inp.context_ids, table=table,
    )
    return RenderOut(stat=stat)


def step_answer_final(inp: AnswerFinalIn, res: StatResources) -> RenderOut:
    """An answer without facts: the judge's refusal or ask-back, the calculation's refusal, or the plan's ask-back.

    In: AnswerFinalIn(kind "verdict" | "refused" | "clarify", question, verdict | refusal + query + source | clarify)
    Out: RenderOut(stat: StatAnswer, outcome "refused" or "clarify", numbers_origin "none")
    Side effects: none (reads resources.table.meta and resources.org_id for the source)
    Node: answer_final (no span)
    """
    if inp.kind == "refused":
        r = inp.refusal
        return RenderOut(stat=StatAnswer(outcome="refused", query=inp.query, refusal=r, text=final_text(r.message, inp.source),
                                         source=inp.source, numbers_origin="none"))
    if inp.kind == "clarify":
        asked = inp.clarify
        options = [label for _, label in asked.options]
        body = "\n".join([asked.message, *(f"- {o}" for o in options)])
        source = to_source(res.table.meta, org_id=res.org_id)
        choices = [ClarifyChoice(code=code, label=label, axis=asked.axis) for code, label in asked.options]
        return RenderOut(stat=StatAnswer(outcome="clarify", query=StatQuery(question=inp.question), clarify_options=options,
                                         clarify_choices=choices, text=final_text(body, source), source=source,
                                         numbers_origin="none"))
    verdict = inp.verdict
    query = StatQuery(question=inp.question)
    if verdict.outcome == "refused":
        r = verdict.refusal
        source = None if r.reason == "out_of_scope" else to_source(res.table.meta, org_id=res.org_id)
        return RenderOut(stat=StatAnswer(outcome="refused", query=query,
                                         refusal=Refusal(reason=r.reason, message=r.message, available=r.available),
                                         text=final_text(r.message, source), source=source, numbers_origin="none"))
    options = list(verdict.clarify_options)
    body = "\n".join([verdict.clarify_question, *(f"- {o}" for o in options)])
    source = to_source(res.table.meta, org_id=res.org_id)
    return RenderOut(stat=StatAnswer(outcome="clarify", query=query, clarify_options=options, text=final_text(body, source),
                                     source=source, numbers_origin="none"))
