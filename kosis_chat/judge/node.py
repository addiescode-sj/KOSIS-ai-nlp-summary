"""judge node (T1-28 S3a): the rules decide first whether the question is answered, refused or asked back.

A pure step: (JudgeIn, StatResources) -> JudgeOut. StatPipeline.ask calls it and records the "judge" span.
"""

from __future__ import annotations

from kosis_chat.judge.rules import judge, judge_multi
from kosis_chat.judge.schemas import JudgeIn, JudgeOut
from kosis_chat.schemas.resources import MultiResources, StatResources


def step_judge(inp: JudgeIn, res: StatResources) -> JudgeOut:
    """Run the judge rules (judge/rules.py, first match decides) on the question against the whole table.

    In: JudgeIn(question)
    Out: JudgeOut(verdict, route "final" = refused / ask-back by the rules, "plan" = go on to the analysis)
    Side effects: none (reads resources.table and resources.index; fills the table's lookup caches)
    Node: judge (span "judge")
    """
    verdict = judge(inp.question, res.table, res.index)
    return JudgeOut(verdict=verdict, route="final" if verdict.final else "plan")


def step_judge_multi(inp: JudgeIn, res: MultiResources) -> JudgeOut:
    """judge for two or three tables (T1-30): the question-level rules only (judge/rules.py judge_multi).

    In: JudgeIn(question)
    Out: JudgeOut(verdict, route "final" | "plan")
    Side effects: none (reads every table's index)
    Node: judge of the multi-table graph (span "judge")
    """
    verdict = judge_multi(inp.question, [r.index for r in res.tables])
    return JudgeOut(verdict=verdict, route="final" if verdict.final else "plan")

