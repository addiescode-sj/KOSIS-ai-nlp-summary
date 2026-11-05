"""compose nodes (T1-28 S3a): the caveats, verification of answer parts, and the code template. The model's sentence
(compose_llm) stays in kosis_chat/pipeline.py until S3b.

Pure steps: (XxxIn, StatResources) -> XxxOut. StatPipeline.ask calls them and records the "verify" / "template" spans.
"""

from __future__ import annotations

from kosis_chat.compose.caveats import caveats_of
from kosis_chat.compose.schemas import (
    COMPOSE_MAX_ATTEMPTS,
    CaveatsIn,
    CaveatsOut,
    TemplateIn,
    TemplateOut,
    VerifyIn,
    VerifyOut,
)
from kosis_chat.compose.template import template_parts
from kosis_chat.compose.verify import verify
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
