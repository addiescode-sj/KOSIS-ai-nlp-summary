"""analyze-plan-v1: the model writes the calculation as v3 blocks when neither the rules nor the fill could place the
question (T0-04 S5b, D3).

The model names cells by meaning only (item and category codes from the menu, periods as PeriodSpec), never by
source_ref, and never writes a number: code resolves the cells, runs the plan through the meaning checks
(semantics.py) and computes the facts. place() turns the model's plan into a computed answer or into the problems to
say back once; a plan that still fails goes to the compact-v2 fallback, as before S5b. A label written where a code
belongs is turned back into its code (as analyze_fill does), and a combine step without a note gets the word for
what it computes, so the sentence can name it.
T1-28 S3b: moved from llm/analyze_plan.py (which re-exports it); the prompts are analyze/prompts/analyze-plan-v1*.md.
"""

from __future__ import annotations

from pydantic import ValidationError

from kosis_chat.analyze import blocks as build  # was kosis_chat.compute.build (the shim of analyze/blocks.py)
from kosis_chat.analyze import prompts
from kosis_chat.analyze.blocks import Built
from kosis_chat.analyze.fill import MENU_LIMIT, resolve_code
from kosis_chat.analyze.rules import PERIOD_AXIS, Plan
from kosis_chat.compute import CalcRefused, Computed, compute
from kosis_chat.compute.engine import _read
from kosis_chat.compute.plan_engine import PlanUnsupported
from kosis_chat.compute.resolve import EventLookup, NeedsClarify
from kosis_chat.judge.question import ITEM_AXIS, TableIndex, asked_periods
from kosis_chat.prepare.schemas import TableInput

from kosis_ai.schemas import CombineStep, MapStep, PlanV3, RankStep, SelectStep, TargetSpec  # isort: skip
from kosis_chat.analyze.schemas import Placed  # isort: skip  # re-export: moved in T1-28 S1b

VERSION = prompts.version(prompts.PLAN)
PERIODS_SHOWN = 8  # a long period list shows its first and last periods
FN_WORDS = {"diff": "차이", "pp_diff": "차이(%p)", "ratio": "배수", "pct_change": "증감률", "share": "비중",
            "per": "비율", "cagr": "연평균 증감률", "index": "지수(기준=100)"}
SYSTEM_PROMPT = prompts.load(prompts.PLAN)
EXAMPLES = prompts.load(prompts.PLAN_EXAMPLES)
RETRY = prompts.load(prompts.PLAN_RETRY)


def _periods_line(index: TableIndex) -> str:
    periods = index.periods
    shown = periods if len(periods) <= PERIODS_SHOWN else periods[:3] + ["…"] + periods[-3:]
    years = any(len(p) == 4 for p in periods)
    months = any(len(p) == 6 for p in periods)
    kind = "월(YYYYMM)" + (", 연 값(YYYY)도 있음" if years else "") if months else "연(YYYY)"
    return f"{', '.join(shown)} (모두 {len(periods)}개, 단위 {kind})"


def menu(index: TableIndex, rule: Plan) -> str:
    """Every category with its codes; a long one lists its top level and the codes the rules matched."""
    lines = []
    for name, axis in index.axes.items():
        named = rule.codes.get(name, [])
        codes = list(axis.labels)
        if len(codes) > MENU_LIMIT:
            top = [c for c in codes if axis.parents.get(c) is None][:MENU_LIMIT]
            codes = top + [c for c in named if c not in top and c in axis.labels]
        lines.append(f"- {name}: " + ", ".join(f"{c}={axis.labels[c].strip()}" for c in codes))
    return "\n".join(lines)


def user_prompt(question: str, index: TableIndex, rule: Plan) -> str:
    found = {n: [index.axes[n].labels[c].strip() for c in cs if c in index.axes[n].labels]
             for n, cs in rule.codes.items() if n in index.axes}
    events = ", ".join(f"{phrase} = {start}" for phrase, start in rule.events)
    return (
        f"[질문]\n{question}\n\n[표]\n{index.table.meta.tbl_nm}\n\n[분류 목록] (항목은 item, 나머지는 dims)\n"
        f"{menu(index, rule)}\n\n[시점]\n{_periods_line(index)}\n\n"
        f"[규칙으로 찾은 것] (참고, 틀릴 수 있음)\n유형={rule.type}, 분류={found}, 시점={rule.periods}"
        + (f", 사건={events}" if events else "")
        + f"\n\n{EXAMPLES}"
    )


def retry_prompt(user: str, problems: list[str]) -> str:
    return user + "\n\n" + RETRY.format(problems="\n".join(f"- {p}" for p in problems))


# ---------------------------------------------------------------- checking the model's plan


def _codes(index: TableIndex, name: str, spec: TargetSpec | None, problems: list[str]) -> TargetSpec | None:
    """spec with labels turned back into codes; a code not in the menu is a problem."""
    if spec is None or not spec.codes or spec.codes == ["*"]:
        if spec is not None and spec.children_of is not None and spec.children_of not in index.axes[name].labels:
            problems.append(f"'{spec.children_of}'은(는) {name} 목록에 없는 코드입니다.")
        return spec
    axis = index.axes[name]
    codes = []
    for c in spec.codes:
        code = resolve_code(axis, c)
        if code is None:
            problems.append(f"'{c}'은(는) {name} 목록에 없는 값입니다. 목록에서 '=' 앞의 코드를 씁니다.")
        elif code not in codes:
            codes.append(code)
    return spec.model_copy(update={"codes": codes}) if codes else spec


def _select(step: SelectStep, index: TableIndex, problems: list[str]) -> SelectStep:
    if step.cells or step.table:
        problems.append(f"select {step.id}: cells·table 대신 item·dims·periods로 칸을 적습니다.")
        return step
    dims = {}
    for name, spec in step.dims.items():
        if name == ITEM_AXIS:
            problems.append(f"select {step.id}: 항목은 dims가 아니라 item에 적습니다.")
        elif name not in index.axes:
            choices = ", ".join(n for n in index.axes if n != ITEM_AXIS)
            problems.append(f"select {step.id}: '{name}'은(는) 분류 이름이 아닙니다. dims의 키는 {choices} 중에서 씁니다.")
        else:
            dims[name] = _codes(index, name, spec, problems)
    item = _codes(index, ITEM_AXIS, step.item, problems) if ITEM_AXIS in index.axes else step.item
    return step.model_copy(update={"item": item, "dims": dims})


def normalized(plan: PlanV3, index: TableIndex) -> tuple[PlanV3, list[str]]:
    """(the plan with codes for labels and a note on every combine, the problems found before running it)."""
    problems: list[str] = []
    steps = []
    for step in plan.steps:
        if isinstance(step, SelectStep):
            step = _select(step, index, problems)
        elif isinstance(step, CombineStep) and not step.note:
            step = step.model_copy(update={"note": FN_WORDS[step.fn]})
        elif isinstance(step, MapStep):
            body = [_select(b, index, problems) if isinstance(b, SelectStep) else b for b in step.body]
            step = step.model_copy(update={"body": body})
        elif isinstance(step, RankStep) and step.candidates:
            problems.append(f"rank {step.id}: candidates 대신 map 블록의 결과(of)로 순위를 매깁니다.")
        steps.append(step)
    if plan.present == "table":
        problems.append("present는 value, trend, compare, share, extreme 중 하나입니다.")
    try:
        fixed = PlanV3.model_validate(plan.model_dump() | {"steps": [s.model_dump() for s in steps]})
    except ValidationError as exc:  # a fix above cannot make a valid plan invalid, but say so if it does
        return plan, problems + [str(exc).splitlines()[0]]
    return fixed, problems


def _missing_periods(question: str, computed: Computed) -> list[str]:
    """Years or months the question names that no cell of the plan reads."""
    asked = asked_periods(question)
    named = list(asked.singles) + [p for r in asked.ranges for p in r]
    read = {f.period for f in computed.result.facts if f.source_ref and f.period}
    return [p for p in dict.fromkeys(named) if not any(q.startswith(p) for q in read)]


def query_plan(question: str, plan: PlanV3, computed: Computed, table: TableInput, index: TableIndex) -> Plan:
    """The cells the plan read, as the rules' Plan (codes per category in reading order), for StatQuery."""
    codes: dict[str, list[str]] = {}
    periods: list[str] = []
    for fact in computed.result.facts:
        if not fact.source_ref:
            continue
        cell = _read(table, fact.source_ref).cell
        for name in index.axes:
            code = index.axis_of(cell, name)
            if code and code not in codes.setdefault(name, []):
                codes[name].append(code)
        if cell.period_raw not in periods:
            periods.append(cell.period_raw)
    rank = next((s for s in plan.steps if isinstance(s, RankStep)), None)
    mapped = next((s for s in plan.steps if isinstance(s, MapStep)), None)
    kind = None if plan.present == "table" else plan.present
    return Plan(question=question, type=kind, codes={n: c for n, c in codes.items() if c}, periods=sorted(periods),
                over=(mapped.axis if mapped else PERIOD_AXIS) if rank else None, pick=rank.pick if rank else None)


def place(question: str, plan: PlanV3, table: TableInput, index: TableIndex,
          lookup: EventLookup | None = None) -> Placed:
    """Run the model's plan; what went wrong comes back as problems in words the model can act on."""
    plan, problems = normalized(plan, index)
    if problems:
        return Placed(plan, problems=problems)
    built = build.written(plan)
    try:
        computed = compute(table, built, lookup)
    except NeedsClarify as exc:
        return Placed(plan, problems=[exc.message], clarify=exc)
    except (CalcRefused, PlanUnsupported) as exc:
        return Placed(plan, problems=[getattr(exc, "message", None) or str(exc)])
    missing = _missing_periods(question, computed)
    if missing:
        return Placed(plan, problems=[f"질문이 말한 시점 {', '.join(missing)}을(를) 읽는 select가 없습니다."])
    return Placed(plan, query_plan(question, plan, computed, table, index), built, computed)
