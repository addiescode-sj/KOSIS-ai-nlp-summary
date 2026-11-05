"""compose nodes (T1-28 S3a, S3b): the caveats, the model's sentence, verification of answer parts, and the code
template.

Steps: (XxxIn, StatResources) -> XxxOut. StatPipeline.ask runs compose_llm and verify in a loop of at most
COMPOSE_MAX_ATTEMPTS attempts, then the template, and records the "compose" / "verify" / "template" spans.
step_compose_llm calls the model through a ModelAccess (keyword llm=, kosis_chat/llm/calls.py); the others are pure.
"""

from __future__ import annotations

from kosis_chat.compose import compose
from kosis_chat.compose.caveats import caveats_of
from kosis_chat.compose.schemas import (
    COMPOSE_MAX_ATTEMPTS,
    CaveatsIn,
    CaveatsOut,
    ComposeLLMIn,
    ComposeLLMOut,
    TemplateIn,
    TemplateOut,
    VerifyIn,
    VerifyOut,
)
from kosis_chat.compose.template import template_parts
from kosis_chat.compose.verify import verify
from kosis_chat.llm.calls import ModelAccess
from kosis_chat.llm.model import STATUS_OK
from kosis_chat.schemas.resources import StatResources


def step_caveats(inp: CaveatsIn, res: StatResources) -> CaveatsOut:
    """The facts shown beside the answer, the caveats, and the caveat texts shown ('참고:' lines, compose's input).

    In: CaveatsIn(computed, built, alternatives, default_applied)
    Out: CaveatsOut(context_ids, caveats, shown)
    Side effects: none (reads resources.table)
    Node: caveats (no span)
    """
    context = set(inp.built.context)
    context_ids = [f.id for f in inp.computed.result.facts if f.source_ref in context and f.id not in inp.computed.main]
    caveats = caveats_of(inp.computed, res.table, inp.alternatives, context_ids)
    # The default sentence already names the conditions the rules chose; a plan's own defaults are shown here.
    shown = [c.text for c in caveats if c.kind != "default_applied" or not inp.default_applied]
    return CaveatsOut(context_ids=context_ids, caveats=caveats, shown=shown)


def step_compose_llm(inp: ComposeLLMIn, res: StatResources, *, llm: ModelAccess) -> ComposeLLMOut:
    """One model attempt at the answer sentence (compose-v2), citing the facts as {f1}; a retry says back the problems.

    In: ComposeLLMIn(question, computed, cautions, context_ids, attempt, problems, prompt, called)
    Out: ComposeLLMOut(parts, prompt, call, meta, route "verify" | "template")
    Side effects: one model call through llm
    Node: compose_llm (span "compose", attribute attempt)
    """
    user = inp.prompt if inp.prompt is not None else compose.user_prompt(inp.question, inp.computed, inp.cautions,
                                                                          inp.context_ids)
    if inp.problems:
        user = compose.retry_prompt(user, inp.problems)
    result, call = llm.ask("compose", compose.SYSTEM_PROMPT, user, compose.ComposeOutput, called=inp.called,
                           attempt=inp.attempt)
    if result.status != STATUS_OK:
        return ComposeLLMOut(prompt=user, call=call, meta={"compose_status": result.status}, route="template")
    return ComposeLLMOut(parts=compose.to_parts(result.value.answer, inp.computed), prompt=user, call=call, route="verify")


def step_verify(inp: VerifyIn, res: StatResources) -> VerifyOut:
    """Check answer parts against the facts (verify(): cited ids, numbers in the text, direction words).

    In: VerifyIn(parts, computed, attempt)
    Out: VerifyOut(problems, route "ok" | "retry" (another model attempt is left) | "template")
    Side effects: none
    Node: verify (span "verify")
    """
    problems = verify(inp.parts, inp.computed)
    if not problems:
        route = "ok"
    else:
        route = "retry" if inp.attempt + 1 < COMPOSE_MAX_ATTEMPTS else "template"
    return VerifyOut(problems=problems, route=route)


def step_template(inp: TemplateIn, res: StatResources) -> TemplateOut:
    """The answer written by code (template_parts), verified like a model's (recorded, not assumed).

    In: TemplateIn(type, computed)
    Out: TemplateOut(parts, problems, verified)
    Side effects: none
    Node: template (span "template")
    """
    parts = template_parts(inp.type, inp.computed)
    problems = verify(parts, inp.computed)
    return TemplateOut(parts=parts, problems=problems, verified=not problems)
