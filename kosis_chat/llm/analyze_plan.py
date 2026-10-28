"""analyze-plan-v1: the model writes the calculation as v3 blocks when neither the rules nor the fill could place the
question (T0-04 S5b, D3).

The model names cells by meaning only (item and category codes from the menu, periods as PeriodSpec), never by
source_ref, and never writes a number: code resolves the cells, runs the plan through the meaning checks
(semantics.py) and computes the facts. place() turns the model's plan into a computed answer or into the problems to
say back once; a plan that still fails goes to the compact-v2 fallback, as before S5b. A label written where a code
belongs is turned back into its code (as analyze_fill does), and a combine step without a note gets the word for
what it computes, so the sentence can name it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from pydantic import ValidationError

from kosis_chat.compute import CalcRefused, Computed, build, compute
from kosis_chat.compute.analyze import PERIOD_AXIS, Plan
from kosis_chat.compute.build import Built
from kosis_chat.compute.engine import _read
from kosis_chat.compute.plan_engine import PlanUnsupported
from kosis_chat.compute.resolve import EventLookup, NeedsClarify
from kosis_chat.data.schema import TableInput
from kosis_chat.judge.question import ITEM_AXIS, TableIndex, asked_periods
from kosis_chat.llm.analyze_fill import MENU_LIMIT, resolve_code

from kosis_ai.schemas import CombineStep, MapStep, PlanV3, RankStep, SelectStep, TargetSpec  # isort: skip

VERSION = "analyze-plan-v1"
PERIODS_SHOWN = 8  # a long period list shows its first and last periods
FN_WORDS = {"diff": "차이", "pp_diff": "차이(%p)", "ratio": "배수", "pct_change": "증감률", "share": "비중",
            "per": "비율", "cagr": "연평균 증감률", "index": "지수(기준=100)"}
SYSTEM_PROMPT = """당신은 통계 질문을 계산 순서(블록 계획)로 옮기는 도우미입니다. 숫자는 쓰지 않습니다. 계산은 코드가 합니다.

블록(steps), 위에서 아래 순서, 각 블록은 앞 블록의 id만 씁니다.
- select: 칸 고르기. {"id": "a", "op": "select", "item": {"codes": ["<항목 코드>"]}, "dims": {"<분류 이름>": {"codes": ["<코드>"]}}, "periods": <시점>}
  · 대상: {"codes": [...]} 또는 분류 합계 {"total": true} 또는 어떤 코드의 바로 아래 전부 {"children_of": "<코드>"}
  · 시점: {"kind": "list", "periods": ["2015"]} · {"kind": "range", "start": "2015", "end": "2020"} · {"kind": "recent", "n": 5}
    · {"kind": "anchor", "event": "<사건 이름>", "offset": -1} (사건 기준 앞뒤, 단위는 표의 시점 단위) · {"kind": "span", "year": "2024", "part": "H1"} (part: year, H1, H2, Q1~Q4)
- aggregate: 여러 칸을 하나로. {"id": "m", "op": "aggregate", "of": "a", "fn": "avg"} (fn: sum, avg, min, max, first, last)
- combine: 두 값으로 계산. {"id": "d", "op": "combine", "a": "b", "b": "a", "fn": "diff"}
  · fn: diff(a-b), pp_diff(%끼리 차이, %p), ratio(a/b 배), pct_change(b 대비 a 증감률 %), share(a 칸들의 합 / b 칸 하나 × 100), index(b=100일 때 a)
  · a, b는 칸 하나짜리 select나 aggregate·combine 결과여야 합니다(share의 a만 여러 칸 가능).
  · orient: "larger_first"(큰 값에서 작은 값을 뺌), 기본은 a-b 그대로
- map + rank: 분류의 여러 코드마다 같은 계산을 하고 순위. {"id": "m", "op": "map", "axis": "<분류 이름>", "codes": [...], "body": [select에서 그 분류를 {"codes": ["*"]}로 쓴 블록들], "result": "<body 블록 id>"}, {"id": "k", "op": "rank", "of": "m", "pick": "max"}

규칙
1. 코드는 [분류 목록]에서 '=' 앞의 값만 씁니다. 분류 이름도 목록 그대로 씁니다. 항목은 item에, 나머지 분류는 dims에 넣습니다.
2. 질문이 말한 분류는 모두 dims에 적습니다. 적지 않은 분류는 코드가 합계로 채웁니다.
3. 시점은 [시점]의 형식(연 YYYY, 월 YYYYMM)으로 씁니다. 질문이 말한 연도·월을 빠짐없이 씁니다.
4. show에는 답에 보여 줄 블록 id를 순서대로 씁니다(비교하는 값과 계산 결과 모두).
5. present: value(값 하나), trend(시간에 따른 변화), compare(두 대상 비교), share(비중), extreme(가장 높은/낮은 것) 중 하나.
6. 표에 없는 원인·전망은 계산하지 않습니다. 계산할 수 없으면 가장 가까운 계산을 하지 말고 그대로 적은 계획만 냅니다.

출력: 설명 없이 JSON 객체 하나 {"steps": [...], "show": [...], "present": "..."}"""
EXAMPLES = """[예 1] 질문: 2010년보다 2020년에 A도와 B도의 사망자 수 차이가 커졌어?
{"steps": [{"id": "x1", "op": "select", "item": {"codes": ["T1"]}, "dims": {"지역": {"codes": ["A"]}}, "periods": {"kind": "list", "periods": ["2010"]}},
 {"id": "y1", "op": "select", "item": {"codes": ["T1"]}, "dims": {"지역": {"codes": ["B"]}}, "periods": {"kind": "list", "periods": ["2010"]}},
 {"id": "x2", "op": "select", "item": {"codes": ["T1"]}, "dims": {"지역": {"codes": ["A"]}}, "periods": {"kind": "list", "periods": ["2020"]}},
 {"id": "y2", "op": "select", "item": {"codes": ["T1"]}, "dims": {"지역": {"codes": ["B"]}}, "periods": {"kind": "list", "periods": ["2020"]}},
 {"id": "g1", "op": "combine", "a": "x1", "b": "y1", "fn": "diff", "orient": "larger_first", "note": "2010년 격차"},
 {"id": "g2", "op": "combine", "a": "x2", "b": "y2", "fn": "diff", "orient": "larger_first", "note": "2020년 격차"},
 {"id": "d", "op": "combine", "a": "g2", "b": "g1", "fn": "diff", "note": "격차 변화"}],
 "show": ["x1", "y1", "x2", "y2", "g1", "g2", "d"], "present": "compare"}
[예 2] 질문: 2023년 전체 출생아 중 3월 출생아의 비중은? (월 표, 연 값 있음)
{"steps": [{"id": "p", "op": "select", "item": {"codes": ["T1"]}, "periods": {"kind": "list", "periods": ["202303"]}},
 {"id": "t", "op": "select", "item": {"codes": ["T1"]}, "periods": {"kind": "list", "periods": ["2023"]}},
 {"id": "s", "op": "combine", "a": "p", "b": "t", "fn": "share"}],
 "show": ["p", "t", "s"], "present": "share"}"""
RETRY = "[이전 계획의 문제]\n{problems}\n위 문제를 고쳐 같은 형식의 JSON 객체 하나만 다시 출력하세요."


@dataclass
class Placed:
    """What place() made of the model's plan: the computed answer, or the problems to say back (and, when the
    plan named something several codes could be, the ask-back)."""

    plan: PlanV3 | None = None
    query_plan: Plan | None = None  # the plan as the rules' Plan, for to_query (StatQuery slots)
    built: Built | None = None
    computed: Computed | None = None
    problems: list[str] = field(default_factory=list)
    clarify: NeedsClarify | None = None


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
