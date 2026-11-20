"""Which computed facts the answer uses: the step after compute (loop engineering, 2026-11-20).

compute returns every fact it made. The answer uses some of them: the number that was asked, and the values a reader
needs beside it (the cells it was computed from, the points a chart of the answer would draw). Naming those facts is
this step. kosis_chat.calc.calculate calls step_answer_facts when compute succeeded and returns its list as
CalcResult.answer; the improvement loop's evaluation grades that list and nothing else:

  - every fact the question asks for has to be in the list, and
  - a fact in the list that is neither asked for nor something the asked number was computed from costs points
    (every candidate of a ranking beside its winner, a second formula "in case" the first is not the one asked).

For now the list is what the plan marks to cite (Computed.main, the facts of plan.show), unchanged.

What must stay (kosis_chat/calc.py reads them): the names AnswerFactsIn, AnswerFactsOut and step_answer_facts, and
their fields. Everything else in this file can change. The step chooses among the facts compute made: it returns ids
of inp.computed.result.facts and makes no number. Computed.main itself stays as it is (templates and charts read it
by position); what this step returns is a list of its own.
"""

from __future__ import annotations

from pydantic import Field, SkipValidation

from kosis_chat.compute.schemas import Computed
from kosis_chat.schemas.resources import MultiResources, StatResources
from kosis_chat.shared.boundary import BoundaryModel


class AnswerFactsIn(BoundaryModel):
    """step_answer_facts's input: the question and what compute returned for it."""

    question: str
    computed: SkipValidation[Computed]


class AnswerFactsOut(BoundaryModel):
    """step_answer_facts's output: ids of the computed facts the answer uses, in the order to show them."""

    fact_ids: SkipValidation[list[str]] = Field(default_factory=list)


def step_answer_facts(inp: AnswerFactsIn, res: StatResources | MultiResources) -> AnswerFactsOut:
    """The facts of inp.computed the answer to inp.question uses.

    In: AnswerFactsIn(question, computed)
    Out: AnswerFactsOut(fact_ids: ids of computed.result.facts)
    Side effects: none (res is the question's table, or its two or three tables: MultiResources)
    """
    return AnswerFactsOut(fact_ids=list(dict.fromkeys(inp.computed.main)))
