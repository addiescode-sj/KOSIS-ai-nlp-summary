"""M2: does rule analysis of the question pick the golden cells? (no LLM)

For every answerable / default_allowed golden item: judge -> kosis_chat.compute.analyze (rules) -> compute, graded by
eval.grade like a model answer. An item the rules leave unresolved is listed as UNRESOLVED (the LLM step's work).
Reports go to eval/results/_checks/analyze/ (data values: not committed).

    python -m eval.analyze_check --golden eval/golden/stat_golden.jsonl
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

from eval.golden import GoldenItem, load_items, read_cells, read_source
from eval.grade import grade
from kosis_chat.compute import CalcRefused, compute
from kosis_chat.compute.analyze import Unresolved, build_plan, rule_plan, to_query
from kosis_chat.config import PROJECT_ROOT
from kosis_chat.data.loader import load_table
from kosis_chat.data.preprocessing import clean
from kosis_chat.judge.answerability import judge
from kosis_chat.judge.question import build_index

from kosis_ai.schemas import FactPart, StatAnswer  # isort: skip

OUT_DIR = PROJECT_ROOT / "eval" / "results" / "_checks" / "analyze"
COMPUTED_KINDS = ("answerable", "default_allowed")
_TABLES: dict[Path, tuple] = {}


def _table(item: GoldenItem) -> tuple:
    if item.path not in _TABLES:
        table = clean(load_table(item.path))
        _TABLES[item.path] = (table, build_index(table))
    return _TABLES[item.path]


def check(item: GoldenItem) -> dict:
    table, index = _table(item)
    row = {"id": item.id, "kind": item.kind, "type": item.type, "question": item.question}
    verdict = judge(item.question, table, index)
    plan = rule_plan(item.question, index, verdict.default_applied)
    row |= {"judge": verdict.outcome, "plan_type": plan.type,
            "codes": {k: [index.axes[k].labels[c] for c in v] for k, v in plan.codes.items()}, "periods": plan.periods}
    if verdict.outcome != "answered":
        return row | {"verdict": "FAIL", "error_types": [f"judge_{verdict.outcome}"], "details": [verdict.rule or ""]}
    try:
        built = build_plan(plan, index, table)
        computed = compute(table, built)
    except Unresolved as exc:
        return row | {"verdict": "UNRESOLVED", "error_types": [], "details": exc.what}
    except CalcRefused as exc:
        return row | {"verdict": "FAIL", "error_types": ["calc_refused"], "details": [f"{exc.reason}: {exc.message}"]}
    answer = StatAnswer(
        outcome="answered",
        query=to_query(plan, index, built),
        result=computed.result,
        parts=[FactPart(fact_id=i) for i in computed.main],
        default_applied=verdict.default_applied,
        source=read_source(item.path),
        numbers_origin="code",
    )
    row |= {"pattern": built.pattern, "plan": built.plan.model_dump(exclude_defaults=True)}
    try:
        graded = grade(item, answer, read_cells(item.path))
    except ValueError as exc:  # the grader cannot place an error type for this kind (eval/scorer.RULES)
        return row | {"verdict": "ERROR", "error_types": ["grader_error"], "details": [str(exc)]}
    types = list(graded.error_types) + (["type_mismatch"] if plan.type != item.type else [])
    return row | {"verdict": graded.verdict, "error_types": types, "details": graded.details}


def render(rows: list[dict], meta: dict) -> str:
    names = ("PASS", "PARTIAL", "FAIL", "UNRESOLVED", "ERROR")
    counts = {v: sum(r["verdict"] == v for r in rows) for v in names}
    lines = [
        f"# M2 규칙 질의 분석 골든 대조 ({meta['started_at']})",
        "",
        f"질문만 보고 규칙으로 셀을 골라 계산했을 때 골든 답과 맞는지 (LLM 없음). 문항 {len(rows)}개: "
        + " / ".join(f"{v} {counts[v]}" for v in names),
        "",
        "| 문항 | 유형(골든→규칙) | 판정 | 오류 | 상세 |",
        "| --- | --- | --- | --- | --- |",
    ]
    for r in rows:
        detail = "; ".join(map(str, r["details"]))[:200].replace("|", "/")
        lines.append(f"| {r['id']} | {r['type']}→{r['plan_type']} | {r['verdict']} | {', '.join(r['error_types'])} | {detail} |")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m eval.analyze_check", description=__doc__.splitlines()[0])
    parser.add_argument("--golden", default="eval/golden/stat_golden.jsonl")
    parser.add_argument("--out", default=str(OUT_DIR))
    args = parser.parse_args(argv)
    version, items, golden_sha = load_items(Path(args.golden))
    rows = [check(i) for i in items if i.kind in COMPUTED_KINDS]
    meta = {"started_at": dt.datetime.now().astimezone().isoformat(timespec="seconds"), "golden": args.golden,
            "golden_version": version, "golden_sha256": golden_sha}
    out = Path(args.out) / f"{dt.datetime.now():%Y%m%d-%H%M%S}"
    out.mkdir(parents=True, exist_ok=False)
    report = render(rows, meta)
    (out / "report.md").write_text(report, encoding="utf-8")
    (out / "rows.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    print(report)
    print(f"saved: {out}")
    return 0 if all(r["verdict"] == "PASS" for r in rows) else 1


if __name__ == "__main__":
    sys.exit(main())
