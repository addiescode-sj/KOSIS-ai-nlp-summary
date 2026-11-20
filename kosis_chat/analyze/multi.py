"""analyze-plan-multi-v1: the model writes the calculation over two or three tables as v3 blocks (T1-30, D3).

The rules' planner reads one table, so a question over several tables is planned by the model only. The prompt lists
every table's categories under its label ([표 A], [표 B], ...); every select names its table by that label, and code
turns the label into the tbl_id, checks the codes against that table's menu (as analyze/write.py does for one table),
requires a select on every table, and computes the plan with compute(tables=). What goes wrong comes back as problems
in words the model can act on, once; a plan that still fails is refused (no compact-v2 fallback for several tables).
"""

from __future__ import annotations

from pydantic import ValidationError

from kosis_chat.analyze import blocks as build
from kosis_chat.analyze import prompts, write
from kosis_chat.analyze.rules import rule_plan
from kosis_chat.analyze.schemas import Placed
from kosis_chat.compute import CalcRefused, compute
from kosis_chat.compute.plan_engine import PlanUnsupported
from kosis_chat.compute.resolve import EventLookup, NeedsClarify
from kosis_chat.prepare.store import store_of
from kosis_chat.schemas.resources import MultiResources

from kosis_ai.schemas import CombineStep, MapStep, PlanV3, RankStep, SelectStep  # isort: skip

VERSION = prompts.version(prompts.PLAN_MULTI)
SYSTEM_PROMPT = prompts.load(prompts.PLAN_MULTI)
EXAMPLES = prompts.load(prompts.PLAN_MULTI_EXAMPLES)
RETRY = prompts.load(prompts.PLAN_RETRY)


def user_prompt(question: str, res: MultiResources) -> str:
    """The question, then every table under its label: name, units, categories (codes) and periods."""
    blocks = []
    for label, r in zip(res.labels, res.tables):
        rule = rule_plan(question, r.index)  # what the rules matched shows up in a long menu
        units = store_of(r.table).units()
        blocks.append(
            f"[표 {label}] {r.table.meta.tbl_nm} (table: \"{label}\")"
            + (f"\n[단위] {', '.join(units[:6])}" if units else "")
            + f"\n[분류 목록] (항목은 item, 나머지는 dims)\n{write.menu(r.index, rule)}\n[시점]\n{write._periods_line(r.index)}"
        )
    return f"[질문]\n{question}\n\n" + "\n\n".join(blocks) + f"\n\n{EXAMPLES}"


def retry_prompt(user: str, problems: list[str]) -> str:
    return user + "\n\n" + RETRY.format(problems="\n".join(f"- {p}" for p in problems))


def _table_id(name: str | None, res: MultiResources) -> str | None:
    """A select's table as the model wrote it ("A", "b", "표 A" or the tbl_id) -> the tbl_id; None when unknown."""
    if not name:
        return None
    text = name.strip().removeprefix("표").strip()
    labels = res.labels
    if text.upper() in labels:
        return labels[text.upper()]
    return text if text in res.by_id else None


def _select(step: SelectStep, res: MultiResources, problems: list[str]) -> SelectStep:
    tbl_id = _table_id(step.table, res)
    if tbl_id is None:
        choices = "·".join(res.labels)
        problems.append(f"select {step.id}: table에 표 기호({choices}) 하나를 씁니다." if not step.table
                        else f"select {step.id}: '{step.table}'은(는) 주어진 표가 아닙니다. {choices} 중에서 씁니다.")
        return step
    fixed = write._select(step.model_copy(update={"table": None}), res.by_id[tbl_id].index, problems)
    return fixed.model_copy(update={"table": tbl_id})


def normalized(plan: PlanV3, res: MultiResources) -> tuple[PlanV3, list[str]]:
    """(the plan with tbl_ids for labels, codes for labels and a note on every combine, the problems found)."""
    problems: list[str] = []
    steps = []
    for step in plan.steps:
        if isinstance(step, SelectStep):
            step = _select(step, res, problems)
        elif isinstance(step, CombineStep) and not step.note:
            step = step.model_copy(update={"note": write.FN_WORDS[step.fn]})
        elif isinstance(step, MapStep):
            body = [_select(b, res, problems) if isinstance(b, SelectStep) else b for b in step.body]
            step = step.model_copy(update={"body": body})
        elif isinstance(step, RankStep) and step.candidates:
            problems.append(f"rank {step.id}: candidates 대신 map 블록의 결과(of)로 순위를 매깁니다.")
        steps.append(step)
    if plan.present == "table":
        problems.append("present는 value, trend, compare, share, extreme 중 하나입니다.")
    read = {s.table for s in steps if isinstance(s, SelectStep)}
    read |= {b.table for s in steps if isinstance(s, MapStep) for b in s.body if isinstance(b, SelectStep)}
    unread = [label for label, tbl_id in res.labels.items() if tbl_id not in read]
    if unread and not problems:
        problems.append(f"표 {'·'.join(unread)}에서 읽는 select가 없습니다. 주어진 표마다 select가 하나 이상 있어야 합니다.")
    try:
        fixed = PlanV3.model_validate(plan.model_dump() | {"steps": [s.model_dump() for s in steps]})
    except ValidationError as exc:
        return plan, problems + [str(exc).splitlines()[0]]
    return fixed, problems


def place(question: str, plan: PlanV3, res: MultiResources,
          lookup: EventLookup | None = None) -> tuple[Placed, CalcRefused | None]:
    """Run the model's plan over every table: (what was placed or the problems in words the model can act on, the
    calculation's refusal when that was the problem)."""
    plan, problems = normalized(plan, res)
    if problems:
        return Placed(plan, problems=problems), None
    built = build.written(plan)
    try:
        computed = compute(res.table, built, lookup, tables=res.table_inputs())
    except NeedsClarify as exc:
        return Placed(plan, problems=[exc.message], clarify=exc), None
    except CalcRefused as exc:  # said back once; if the second plan is refused too, the refusal is the answer
        return Placed(plan, problems=[exc.message]), exc
    except PlanUnsupported as exc:
        return Placed(plan, problems=[str(exc)]), None
    missing = write._missing_periods(question, computed)
    if missing:
        return Placed(plan, problems=[f"질문이 말한 시점 {', '.join(missing)}을(를) 읽는 select가 없습니다."]), None
    return Placed(plan, None, built, computed), None


__all__ = ["EXAMPLES", "RETRY", "SYSTEM_PROMPT", "VERSION", "normalized", "place", "retry_prompt", "user_prompt"]
