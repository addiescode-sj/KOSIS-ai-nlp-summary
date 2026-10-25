"""analyze-fill-v1: when the rules cannot pick the cells, the model chooses codes from a menu of the table (T1-06).

Only what the rules left open is taken from the model: a category the rules pinned to one code keeps it. Codes are
checked against the table, so the model can name nothing that is not there.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from kosis_chat.compute.analyze import PERIOD_AXIS, Plan
from kosis_chat.judge.question import TableIndex

from kosis_ai.schemas import QuestionType  # isort: skip

VERSION = "analyze-fill-v1"
MENU_LIMIT = 60  # labels per category; a longer category lists its top level and the codes the rules matched
SYSTEM_PROMPT = """당신은 통계 질문을 통계표의 칸으로 옮기는 도우미입니다.
[분류 목록]에서 질문이 가리키는 코드를 고릅니다. 목록에 없는 코드는 쓰지 않습니다.

규칙
1. codes에는 질문이 직접 가리키는 분류만 넣습니다. 질문에 없는 분류는 넣지 않습니다(코드가 합계로 채웁니다).
2. 비교 질문이면 비교하는 두 대상의 코드를 같은 분류에 두 개 넣습니다.
3. type: value(값 하나), trend(시간에 따른 변화), compare(두 대상 비교), share(비중), extreme(가장 높은/낮은 것).
4. extreme이면 over에 순위를 매길 분류 이름(시점이면 "시점")을, pick에 max 또는 min을 넣습니다.

출력: 설명 없이 JSON 객체 하나
{"type": "...", "codes": {"<분류 이름>": ["<코드>"]}, "over": null, "pick": null}"""


class AnalyzeFill(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: QuestionType | None = None
    codes: dict[str, list[str]] = Field(default_factory=dict)
    over: str | None = None
    pick: Literal["max", "min"] | None = None


def menu(index: TableIndex, plan: Plan) -> str:
    lines = []
    for name, axis in index.axes.items():
        codes = list(axis.labels)
        if len(codes) > MENU_LIMIT:
            named = set(plan.codes.get(name, []))
            codes = [c for c in codes if axis.parents.get(c) is None or c in named][:MENU_LIMIT]
        listed = ", ".join(f"{c}={axis.labels[c].strip()}" for c in codes)
        lines.append(f"- {name}: {listed}")
    return "\n".join(lines)


def user_prompt(question: str, index: TableIndex, plan: Plan) -> str:
    found = {n: [index.axes[n].labels[c].strip() for c in cs] for n, cs in plan.codes.items()}
    periods = ", ".join(index.periods[:3] + (["…"] if len(index.periods) > 6 else []) + index.periods[-3:])
    return (
        f"[질문]\n{question}\n\n[분류 목록]\n{menu(index, plan)}\n- 시점: {periods}\n\n"
        f"[규칙으로 찾은 것]\n유형={plan.type}, 분류={found}, 시점={plan.periods}\n"
        f"[규칙이 정하지 못한 것]\n{', '.join(plan.unresolved) or '셀 선택'}"
    )


def apply(plan: Plan, fill: AnalyzeFill, index: TableIndex) -> tuple[Plan, list[str]]:
    """The plan with the model's choices where the rules had none; returns (plan, ignored choices)."""
    ignored: list[str] = []
    codes = dict(plan.codes)
    for name, chosen in fill.codes.items():
        axis = index.axes.get(name)
        if axis is None:
            ignored.append(f"분류 없음 {name}")
            continue
        valid = [c for c in dict.fromkeys(chosen) if c in axis.labels]
        ignored += [f"코드 없음 {name}={c}" for c in chosen if c not in axis.labels]
        if valid and len(codes.get(name, [])) != 1:
            codes[name] = valid
    over = plan.over
    if fill.over and over is None and (fill.over == PERIOD_AXIS or fill.over in index.axes):
        over = fill.over
    kind = plan.type or fill.type
    return replace(plan, codes=codes, over=over, pick=plan.pick or fill.pick, type=kind, unresolved=[]), ignored
