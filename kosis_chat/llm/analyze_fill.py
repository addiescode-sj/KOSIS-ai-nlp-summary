"""analyze-fill-v2: when the rules cannot pick the cells, the model chooses codes from a menu of the table (T1-06).

Only what the rules left open is taken from the model: a category the rules pinned to one code keeps it. The menu
lists just the categories the rules could not settle (all of them when the gap is not a category), the model's
answer is checked against the table, a label written where a code belongs is turned back into its code, and what
remains wrong is said back to the model once (pipeline) so it can correct itself. Codes it names that are not in the
table never get through.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from kosis_chat.compute.analyze import PERIOD_AXIS, Plan
from kosis_chat.judge.question import Axis, TableIndex, label_forms, squash

from kosis_ai.schemas import QuestionType  # isort: skip

VERSION = "analyze-fill-v2"
MENU_LIMIT = 60  # labels per category; a longer category lists its top level and the codes the rules matched
SYSTEM_PROMPT = """당신은 통계 질문을 통계표의 칸으로 옮기는 도우미입니다.
규칙이 정하지 못한 분류만 채웁니다. [채울 분류 목록]에 있는 분류의 코드를 고릅니다.

규칙
1. codes의 키는 [채울 분류 목록]의 분류 이름만 씁니다. 값은 목록에서 '=' 앞의 코드입니다. 이름(라벨)을 쓰지 않습니다.
2. 목록에 없는 코드·분류는 쓰지 않습니다. 시점(연도·월)은 codes에 넣지 않습니다(코드가 따로 다룹니다).
3. 질문이 가리키는 것만 넣습니다. 비교 질문이면 비교하는 두 대상의 코드를 같은 분류에 두 개 넣습니다.
4. type: value(값 하나), trend(시간에 따른 변화), compare(두 대상 비교), share(비중), extreme(가장 높은/낮은 것).
5. extreme이면 over에 순위를 매길 분류 이름(시점이면 "시점")을, pick에 max 또는 min을 넣습니다.

출력: 설명 없이 JSON 객체 하나
{"type": "...", "codes": {"<분류 이름>": ["<코드>"]}, "over": null, "pick": null}"""
RETRY = "[이전 답변의 문제]\n{problems}\n위 문제를 고쳐 같은 형식의 JSON 객체 하나만 다시 출력하세요."


class AnalyzeFill(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: QuestionType | None = None
    codes: dict[str, list[str]] = Field(default_factory=dict)
    over: str | None = None
    pick: Literal["max", "min"] | None = None

    @field_validator("codes", mode="before")
    @classmethod
    def _flatten(cls, value):
        """A code written bare ("T1"), as a number, or in a nested list (["전국", ["11", "21"]]) is still a choice."""
        if not isinstance(value, dict):
            return value
        out = {}
        for name, chosen in value.items():
            items = chosen if isinstance(chosen, list) else [chosen]
            flat = []
            for c in items:
                flat += [x for x in c if isinstance(x, (str, int, float))] if isinstance(c, list) else [c]
            out[name] = [str(x) for x in flat if isinstance(x, (str, int, float)) and not isinstance(x, bool)]
        return out


def asked_axes(plan: Plan, index: TableIndex) -> list[str]:
    """The categories named in Plan.unresolved ("항목", "행정구역별: 2개 중 하나"); [] when the gap is something else."""
    names = [u.split(":")[0].strip() for u in plan.unresolved]
    return [n for n in dict.fromkeys(names) if n in index.axes]


def _menu_codes(axis: Axis, named: list[str]) -> list[str]:
    if len(named) > 1:  # "2개 중 하나": the rules narrowed it to these
        return named
    codes = list(axis.labels)
    if len(codes) > MENU_LIMIT:
        codes = [c for c in codes if axis.parents.get(c) is None or c in named][:MENU_LIMIT]
    return codes


def menu(index: TableIndex, plan: Plan, axes: list[str] | None = None) -> str:
    lines = []
    for name in axes or list(index.axes):
        axis = index.axes[name]
        listed = ", ".join(f"{c}={axis.labels[c].strip()}" for c in _menu_codes(axis, plan.codes.get(name, [])))
        lines.append(f"- {name}: {listed}")
    return "\n".join(lines)


def user_prompt(question: str, index: TableIndex, plan: Plan) -> str:
    asked = asked_axes(plan, index)
    found = {n: [index.axes[n].labels[c].strip() for c in cs] for n, cs in plan.codes.items() if n not in asked}
    periods = ", ".join(index.periods if len(index.periods) <= 6 else index.periods[:3] + ["…"] + index.periods[-3:])
    return (
        f"[질문]\n{question}\n\n[채울 분류 목록]\n{menu(index, plan, asked)}\n- 시점(참고, codes에 넣지 않음): {periods}\n\n"
        f"[규칙으로 찾은 것]\n유형={plan.type}, 분류={found}, 시점={plan.periods}\n"
        f"[규칙이 정하지 못한 것]\n{', '.join(plan.unresolved) or '셀 선택'}"
    )


def retry_prompt(user: str, problems: list[str]) -> str:
    return user + "\n\n" + RETRY.format(problems="\n".join(f"- {p}" for p in problems))


def resolve_code(axis: Axis, chosen: str) -> str | None:
    """The code the model meant: the code itself, the code of the label it wrote instead ('부산광역시' -> '21'), or
    either side of a menu line it copied whole ('21=부산광역시')."""
    pieces = [chosen.strip()] + ([p.strip() for p in chosen.split("=", 1)] if "=" in chosen else [])
    for piece in pieces:
        if piece in axis.labels:
            return piece
    for piece in pieces:
        form = squash(piece)
        if not form:
            continue
        for code, label in axis.labels.items():
            if form == squash(label) or form in label_forms(label, regional_total=axis.regional)[0]:
                return code
    return None


def apply(plan: Plan, fill: AnalyzeFill, index: TableIndex) -> tuple[Plan, list[str], list[str]]:
    """The plan with the model's choices where the rules had none.

    Returns (plan, ignored, problems): ignored records what was dropped (for the eval record), problems says the
    same to the model for one more try. The pattern and period groups the rules set are kept as they are."""
    ignored: list[str] = []
    problems: list[str] = []
    asked = asked_axes(plan, index)
    codes = dict(plan.codes)
    for name, chosen in fill.codes.items():
        axis = index.axes.get(name)
        if axis is None:
            # "T10=출생건수 (명)": ["2025"] — the category's code written as the key; or codes filed under a made-up name
            moved = _move([name, *chosen], asked or list(index.axes), index) if name != PERIOD_AXIS else {}
            for target, found in moved.items():
                if len(codes.get(target, [])) != 1:
                    codes[target] = found
            if not moved:
                ignored.append(f"분류 없음 {name}")
                choices = ", ".join(asked) if asked else "목록의 분류 이름"
                problems.append("시점은 codes에 넣지 않습니다." if name == PERIOD_AXIS else f"'{name}'은(는) 분류 이름이 아닙니다. codes의 키는 {choices} 중에서 씁니다.")
            continue
        valid: list[str] = []
        for c in dict.fromkeys(chosen):
            code = resolve_code(axis, c)
            if code is None:
                ignored.append(f"코드 없음 {name}={c}")
                problems.append(f"'{c}'은(는) {name} 목록에 없는 값입니다. 목록에서 '=' 앞의 코드를 씁니다.")
            elif code not in valid:
                valid.append(code)
        if valid and len(codes.get(name, [])) != 1:
            codes[name] = valid
    for name in asked:
        if not codes.get(name) or (len(plan.codes.get(name, [])) > 1 and codes[name] == plan.codes[name]):
            problems.append(f"{name}: 질문이 가리키는 코드를 고르지 않았습니다.")
    over = plan.over
    if fill.over and over is None and (fill.over == PERIOD_AXIS or fill.over in index.axes):
        over = fill.over
    kind = plan.type or fill.type
    return replace(plan, codes=codes, over=over, pick=plan.pick or fill.pick, type=kind, unresolved=[]), ignored, problems


def _move(chosen: list[str], axes: list[str], index: TableIndex) -> dict[str, list[str]]:
    """Codes or labels the model filed under a wrong category name, placed in the category that has them."""
    out: dict[str, list[str]] = {}
    for c in chosen:
        for name in axes:
            if (code := resolve_code(index.axes[name], c)) is not None:
                out.setdefault(name, [])
                if code not in out[name]:
                    out[name].append(code)
                break
    return out
