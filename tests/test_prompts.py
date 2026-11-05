"""The prompts moved to .md files in T1-28 S3b (D15) load as the exact text the Python constants held.

The OLD_* strings are the constants as they were in llm/analyze_fill.py, llm/analyze_plan.py, llm/event_lookup.py and
llm/compose.py before the move, copied here character for character (repr). This test belongs to the move only
(plans/T1-28.md S3): delete it, and these copies, once the .md files are the prompts' only home.
"""

from __future__ import annotations

import importlib
import subprocess
import sys
from pathlib import Path

import pytest

from kosis_chat.analyze import event_lookup, fill, prompts as analyze_prompts, write
from kosis_chat.compose import compose, prompts as compose_prompts

ROOT = Path(__file__).resolve().parents[1]

OLD_FILL_SYSTEM = '당신은 통계 질문을 통계표의 칸으로 옮기는 도우미입니다.\n규칙이 정하지 못한 분류만 채웁니다. [채울 분류 목록]에 있는 분류의 코드를 고릅니다.\n\n규칙\n1. codes의 키는 [채울 분류 목록]의 분류 이름만 씁니다. 값은 목록에서 \'=\' 앞의 코드입니다. 이름(라벨)을 쓰지 않습니다.\n2. 목록에 없는 코드·분류는 쓰지 않습니다. 시점(연도·월)은 codes에 넣지 않습니다(코드가 따로 다룹니다).\n3. 질문이 가리키는 것만 넣습니다. 비교 질문이면 비교하는 두 대상의 코드를 같은 분류에 두 개 넣습니다.\n4. type: value(값 하나), trend(시간에 따른 변화), compare(두 대상 비교), share(비중), extreme(가장 높은/낮은 것).\n5. extreme이면 over에 순위를 매길 분류 이름(시점이면 "시점")을, pick에 max 또는 min을 넣습니다.\n\n출력: 설명 없이 JSON 객체 하나\n{"type": "...", "codes": {"<분류 이름>": ["<코드>"]}, "over": null, "pick": null}'
OLD_FILL_RETRY = '[이전 답변의 문제]\n{problems}\n위 문제를 고쳐 같은 형식의 JSON 객체 하나만 다시 출력하세요.'
OLD_PLAN_SYSTEM = '당신은 통계 질문을 계산 순서(블록 계획)로 옮기는 도우미입니다. 숫자는 쓰지 않습니다. 계산은 코드가 합니다.\n\n블록(steps), 위에서 아래 순서, 각 블록은 앞 블록의 id만 씁니다.\n- select: 칸 고르기. {"id": "a", "op": "select", "item": {"codes": ["<항목 코드>"]}, "dims": {"<분류 이름>": {"codes": ["<코드>"]}}, "periods": <시점>}\n  · 대상: {"codes": [...]} 또는 분류 합계 {"total": true} 또는 어떤 코드의 바로 아래 전부 {"children_of": "<코드>"}\n  · 시점: {"kind": "list", "periods": ["2015"]} · {"kind": "range", "start": "2015", "end": "2020"} · {"kind": "recent", "n": 5}\n    · {"kind": "anchor", "event": "<사건 이름>", "offset": -1} (사건 기준 앞뒤, 단위는 표의 시점 단위) · {"kind": "span", "year": "2024", "part": "H1"} (part: year, H1, H2, Q1~Q4)\n- aggregate: 여러 칸을 하나로. {"id": "m", "op": "aggregate", "of": "a", "fn": "avg"} (fn: sum, avg, min, max, first, last)\n- combine: 두 값으로 계산. {"id": "d", "op": "combine", "a": "b", "b": "a", "fn": "diff"}\n  · fn: diff(a-b), pp_diff(%끼리 차이, %p), ratio(a/b 배), pct_change(b 대비 a 증감률 %), share(a 칸들의 합 / b 칸 하나 × 100), index(b=100일 때 a)\n  · a, b는 칸 하나짜리 select나 aggregate·combine 결과여야 합니다(share의 a만 여러 칸 가능).\n  · orient: "larger_first"(큰 값에서 작은 값을 뺌), 기본은 a-b 그대로\n- map + rank: 분류의 여러 코드마다 같은 계산을 하고 순위. {"id": "m", "op": "map", "axis": "<분류 이름>", "codes": [...], "body": [select에서 그 분류를 {"codes": ["*"]}로 쓴 블록들], "result": "<body 블록 id>"}, {"id": "k", "op": "rank", "of": "m", "pick": "max"}\n\n규칙\n1. 코드는 [분류 목록]에서 \'=\' 앞의 값만 씁니다. 분류 이름도 목록 그대로 씁니다. 항목은 item에, 나머지 분류는 dims에 넣습니다.\n2. 질문이 말한 분류는 모두 dims에 적습니다. 적지 않은 분류는 코드가 합계로 채웁니다.\n3. 시점은 [시점]의 형식(연 YYYY, 월 YYYYMM)으로 씁니다. 질문이 말한 연도·월을 빠짐없이 씁니다.\n4. show에는 답에 보여 줄 블록 id를 순서대로 씁니다(비교하는 값과 계산 결과 모두).\n5. present: value(값 하나), trend(시간에 따른 변화), compare(두 대상 비교), share(비중), extreme(가장 높은/낮은 것) 중 하나.\n6. 표에 없는 원인·전망은 계산하지 않습니다. 계산할 수 없으면 가장 가까운 계산을 하지 말고 그대로 적은 계획만 냅니다.\n\n출력: 설명 없이 JSON 객체 하나 {"steps": [...], "show": [...], "present": "..."}'
OLD_PLAN_EXAMPLES = '[예 1] 질문: 2010년보다 2020년에 A도와 B도의 사망자 수 차이가 커졌어?\n{"steps": [{"id": "x1", "op": "select", "item": {"codes": ["T1"]}, "dims": {"지역": {"codes": ["A"]}}, "periods": {"kind": "list", "periods": ["2010"]}},\n {"id": "y1", "op": "select", "item": {"codes": ["T1"]}, "dims": {"지역": {"codes": ["B"]}}, "periods": {"kind": "list", "periods": ["2010"]}},\n {"id": "x2", "op": "select", "item": {"codes": ["T1"]}, "dims": {"지역": {"codes": ["A"]}}, "periods": {"kind": "list", "periods": ["2020"]}},\n {"id": "y2", "op": "select", "item": {"codes": ["T1"]}, "dims": {"지역": {"codes": ["B"]}}, "periods": {"kind": "list", "periods": ["2020"]}},\n {"id": "g1", "op": "combine", "a": "x1", "b": "y1", "fn": "diff", "orient": "larger_first", "note": "2010년 격차"},\n {"id": "g2", "op": "combine", "a": "x2", "b": "y2", "fn": "diff", "orient": "larger_first", "note": "2020년 격차"},\n {"id": "d", "op": "combine", "a": "g2", "b": "g1", "fn": "diff", "note": "격차 변화"}],\n "show": ["x1", "y1", "x2", "y2", "g1", "g2", "d"], "present": "compare"}\n[예 2] 질문: 2023년 전체 출생아 중 3월 출생아의 비중은? (월 표, 연 값 있음)\n{"steps": [{"id": "p", "op": "select", "item": {"codes": ["T1"]}, "periods": {"kind": "list", "periods": ["202303"]}},\n {"id": "t", "op": "select", "item": {"codes": ["T1"]}, "periods": {"kind": "list", "periods": ["2023"]}},\n {"id": "s", "op": "combine", "a": "p", "b": "t", "fn": "share"}],\n "show": ["p", "t", "s"], "present": "share"}'
OLD_PLAN_RETRY = '[이전 계획의 문제]\n{problems}\n위 문제를 고쳐 같은 형식의 JSON 객체 하나만 다시 출력하세요.'
OLD_EVENT_SYSTEM = '당신은 사건이 언제 시작됐는지 답하는 도우미입니다.\n사건이 대한민국에서 처음 시작된(발생한) 연도와 월을 답합니다. 확실하지 않으면 null로 답합니다. 추측하지 않습니다.\n\n출력: 설명 없이 JSON 객체 하나 {"start": "YYYY-MM"} (월을 모르면 "YYYY", 모르면 null)'
OLD_COMPOSE_SYSTEM = '당신은 KOSIS 통계표의 계산 결과를 한국어 답변 문장으로 쓰는 도우미입니다.\n수치는 코드가 이미 계산했습니다. 당신은 [사실 목록]의 사실을 {f번호}로 넣어 문장만 씁니다.\n\n규칙\n1. 숫자는 직접 쓰지 말고 반드시 {f번호}로 넣습니다. 연도(예: 2023년)와 이름 속 숫자(예: 1인 가구)만 그대로 씁니다.\n2. [반드시 인용] 표시가 있는 사실은 모두 넣습니다. [참고] 사실은 도움이 될 때만 넣습니다. [보조] 사실(다른 가정의 값 등)은 필요하면 한 문장으로 덧붙입니다.\n3. {f번호} 자리에는 단위까지 들어갑니다. 뒤에 단위를 다시 쓰지 않습니다.\n4. 차이와 증감률은 부호 없이 표시됩니다. 사실 목록의 (증가)/(감소)에 맞게 \'늘었습니다/줄었습니다\', \'많습니다/적습니다\'로 방향을 씁니다.\n5. 표에 없는 원인, 전망, 평가는 쓰지 않습니다. 출처, 기본 조건 안내, [주의] 내용은 코드가 붙이므로 쓰지 않습니다.\n6. 존댓말로 1~3문장. 질문에 바로 답하는 문장으로 시작합니다.\n\n출력: 설명 없이 JSON 객체 하나 {"answer": "<{f번호}를 넣은 답변 문장>"}'
OLD_COMPOSE_EXAMPLE = '{"answer": "2023년 서울특별시의 1인 가구는 {f1}입니다."}'
OLD_COMPOSE_RETRY = '[이전 답변의 문제]\n{problems}\n위 문제를 고쳐 같은 형식의 JSON 객체 하나만 다시 출력하세요.'

# (module, constant, the old text, the prompts module, the .md file)
CASES = [
    (fill, "SYSTEM_PROMPT", OLD_FILL_SYSTEM, analyze_prompts, "analyze-fill-v2.md"),
    (fill, "RETRY", OLD_FILL_RETRY, analyze_prompts, "analyze-fill-v2-retry.md"),
    (write, "SYSTEM_PROMPT", OLD_PLAN_SYSTEM, analyze_prompts, "analyze-plan-v1.md"),
    (write, "EXAMPLES", OLD_PLAN_EXAMPLES, analyze_prompts, "analyze-plan-v1-examples.md"),
    (write, "RETRY", OLD_PLAN_RETRY, analyze_prompts, "analyze-plan-v1-retry.md"),
    (event_lookup, "SYSTEM_PROMPT", OLD_EVENT_SYSTEM, analyze_prompts, "event-lookup-v1.md"),
    (compose, "SYSTEM_PROMPT", OLD_COMPOSE_SYSTEM, compose_prompts, "compose-v2.md"),
    (compose, "EXAMPLE", OLD_COMPOSE_EXAMPLE, compose_prompts, "compose-v2-example.md"),
    (compose, "RETRY", OLD_COMPOSE_RETRY, compose_prompts, "compose-v2-retry.md"),
]


@pytest.mark.parametrize(("module", "name", "old", "loader", "file"), CASES, ids=[c[4] for c in CASES])
def test_md_text_is_the_old_constant(module, name, old, loader, file):
    assert getattr(module, name) == old  # character for character, no trailing newline
    assert loader.load(file) == old
    text = (ROOT / "kosis_chat" / loader.__name__.split(".")[1] / "prompts" / file).read_text(encoding="utf-8")
    assert text == old + "\n"  # the file: the text and one final newline (a CRLF checkout reads the same)


def test_versions_come_from_the_file_names():
    assert (fill.VERSION, write.VERSION, event_lookup.VERSION, compose.VERSION) == (
        "analyze-fill-v2", "analyze-plan-v1", "event-lookup-v1", "compose-v2")
    from kosis_chat import pipeline
    from kosis_chat.llm import compose as old_compose

    assert pipeline.VERSION is old_compose.VERSION == "compose-v2"  # Phase 2 scripts/stat_baseline.py:53


def test_old_paths_give_the_same_prompts():
    for old, new in (("kosis_chat.llm.analyze_fill", fill), ("kosis_chat.llm.analyze_plan", write),
                     ("kosis_chat.llm.event_lookup", event_lookup), ("kosis_chat.llm.compose", compose)):
        mod = importlib.import_module(old)
        assert mod.SYSTEM_PROMPT is new.SYSTEM_PROMPT and mod.VERSION is new.VERSION


def test_load_is_cached():
    assert analyze_prompts.load("analyze-fill-v2.md") is analyze_prompts.load("analyze-fill-v2.md")


def test_md_files_are_package_data():
    import tomllib

    data = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["tool"]["setuptools"]["package-data"]
    assert data["kosis_chat.analyze"] == ["prompts/*.md"] and data["kosis_chat.compose"] == ["prompts/*.md"]
    listed = {p.name for p in (ROOT / "kosis_chat" / "analyze" / "prompts").glob("*.md")}
    listed |= {p.name for p in (ROOT / "kosis_chat" / "compose" / "prompts").glob("*.md")}
    assert listed == {c[4] for c in CASES}


def test_prompts_load_from_a_clean_interpreter(tmp_path):
    """Only the folder on sys.path (the Phase 2 PYTHONPATH way): importlib.resources finds the .md files."""
    code = ("import sys; sys.path.insert(0, %r); from kosis_chat.compose import compose; "
            "from kosis_chat.analyze import write; print(compose.VERSION, len(write.SYSTEM_PROMPT))" % str(ROOT))
    done = subprocess.run([sys.executable, "-I", "-c", code], cwd=tmp_path, capture_output=True, text=True, timeout=120)
    assert done.returncode == 0, done.stderr
    assert done.stdout.split() == ["compose-v2", str(len(OLD_PLAN_SYSTEM))]
