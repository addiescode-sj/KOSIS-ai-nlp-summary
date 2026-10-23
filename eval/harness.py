"""Automatic evaluation harness: grade StatAnswers against the golden set and save a dated report.

    python -m eval.harness --suite stat --answers answers.jsonl --stage T1-12
    python -m eval.harness --suite stat --oracle --stage oracle   # golden answers graded by themselves (self-check)

answers.jsonl: one {"question_id": ..., "answer": <StatAnswer JSON>} per line. A missing answer is a run_failure.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from collections import Counter
from pathlib import Path

from eval.golden import GoldenItem, load_items, read_cells, read_source
from eval.grade import FACT_ERRORS, Grade, grade
from eval.scorer import aggregate, read_jsonl, write_jsonl
from kosis_ai.schemas import FactPart, Refusal, StatAnswer, StatResult

REPORTS_DIR = Path(__file__).resolve().parent / "reports"


def oracle_answer(item: GoldenItem) -> StatAnswer:
    """The answer the golden item describes; grading it must give PASS (checks golden, data and grader agree)."""
    source = read_source(item.path)
    if item.kind == "not_answerable":
        refusal = Refusal(reason=item.allowed_reasons[0], message="oracle")
        return StatAnswer(outcome="refused", refusal=refusal, source=source)
    if item.kind == "clarify":
        return StatAnswer(outcome="clarify", clarify_options=item.clarify_options, source=source)
    return StatAnswer(
        outcome="answered",
        result=StatResult(facts=item.required_facts + item.aux_facts),
        parts=[FactPart(fact_id=f.id) for f in item.required_facts],
        default_applied=item.default_applied,
        source=source,
    )


def _failed(item: GoldenItem, why: str) -> Grade:
    return Grade(question_id=item.id, kind=item.kind, type=item.type, outcome="none", verdict="FAIL",
                 error_types=["run_failure"], details=[why])


def grade_all(items: list[GoldenItem], answers: dict[str, StatAnswer]) -> list[Grade]:
    grades = []
    for item in items:
        answer = answers.get(item.id)
        if answer is None:
            grades.append(_failed(item, "no answer"))
        else:
            grades.append(grade(item, answer, read_cells(item.path)))
    return grades


def _rate(hits: int, total: int) -> dict:
    return {"hits": hits, "total": total, "rate": hits / total if total else None}


def summarize(grades: list[Grade]) -> dict:
    """T1-04 metrics on top of scorer.aggregate (same verdict table as human review)."""
    rows = [{"kind": g.kind, "review_status": "confirmed", "final_verdict": g.verdict, "error_types": g.error_types,
             "call_status": "failed" if "run_failure" in g.error_types else "ok"} for g in grades]
    numeric = [g for g in grades if g.kind in ("answerable", "default_allowed")]
    refusal = [g for g in grades if g.kind == "not_answerable"]
    by_type: dict[str, Counter] = {}
    for g in grades:
        by_type.setdefault(g.type or "none", Counter())[g.verdict] += 1
    return aggregate(rows) | {
        # A numeric question is correct when it was answered with every required fact right.
        "numeric_accuracy": _rate(
            sum(1 for g in numeric if g.outcome == "answered" and not set(g.error_types) & set(FACT_ERRORS)), len(numeric)
        ),
        # A refusal counts only with an allowed reason.
        "refusal_rate": _rate(sum(1 for g in refusal if g.outcome == "refused" and "wrong_reason" not in g.error_types), len(refusal)),
        "false_refusals": sum(1 for g in grades if "false_refusal" in g.error_types),
        "number_errors": sum(g.number_errors for g in grades),
        "by_type": {t: dict(c) for t, c in sorted(by_type.items())},
        "needs_review": sum(1 for g in grades if g.needs_review),
    }


def _pct(r: dict) -> str:
    return "-" if r["rate"] is None else f"{r['hits']}/{r['total']} ({r['rate']:.0%})"


def render_markdown(summary: dict, meta: dict) -> str:
    lines = [
        f"# 평가 리포트 — {meta['stage']}",
        "",
        f"- 실행: {meta['started_at']} · 골든 {meta['golden_version']} ({meta['questions']}문항, sha256 {meta['golden_sha256'][:12]})",
        f"- 답변: {meta['answers']}",
        "",
        "| 지표 | 값 |",
        "| --- | --- |",
        f"| 수치 정확도 (answerable + default_allowed) | {_pct(summary['numeric_accuracy'])} |",
        f"| 거절률 (not_answerable, 허용 사유) | {_pct(summary['refusal_rate'])} |",
        f"| 잘못된 거절 | {summary['false_refusals']} |",
        f"| 숫자 오류 건수 | {summary['number_errors']} |",
        f"| 실행 실패 | {summary['run_failures']} |",
        f"| 사람 확인 필요 | {summary['needs_review']} |",
        "",
        "| 분해 | PASS | PARTIAL | FAIL |",
        "| --- | --- | --- | --- |",
    ]
    for kind, k in summary["by_kind"].items():
        if k["total"]:
            lines.append(f"| kind={kind} | {k['pass']} | {k['partial']} | {k['fail']} |")
    for t, c in summary["by_type"].items():
        lines.append(f"| type={t} | {c.get('PASS', 0)} | {c.get('PARTIAL', 0)} | {c.get('FAIL', 0)} |")
    if summary["error_types"]:
        lines += ["", "오류 유형: " + ", ".join(f"{k} {v}" for k, v in summary["error_types"].items())]
    return "\n".join(lines) + "\n"


def save_report(grades: list[Grade], meta: dict, out_root: Path = REPORTS_DIR) -> Path:
    summary = summarize(grades)
    run_dir = out_root / f"{dt.datetime.now():%Y%m%d-%H%M%S}-{meta['stage']}"
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "report.json").write_text(json.dumps(meta | {"summary": summary}, ensure_ascii=False, indent=2), encoding="utf-8")
    (run_dir / "report.md").write_text(render_markdown(summary, meta), encoding="utf-8")
    write_jsonl(run_dir / "grades.jsonl", [g.model_dump() for g in grades])  # holds data values: gitignored
    return run_dir


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m eval.harness", description="Grade StatAnswers automatically.")
    parser.add_argument("--suite", choices=["stat"], required=True)
    parser.add_argument("--golden", default="eval/golden/stat_golden.jsonl")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--answers", help="jsonl of {question_id, answer}")
    group.add_argument("--oracle", action="store_true", help="grade the golden answers themselves")
    parser.add_argument("--stage", required=True)
    parser.add_argument("--out", default=str(REPORTS_DIR))
    args = parser.parse_args(argv)

    version, items, golden_sha = load_items(Path(args.golden))
    if args.oracle:
        answers = {i.id: oracle_answer(i) for i in items}
    else:
        answers = {r["question_id"]: StatAnswer.model_validate(r["answer"]) for r in read_jsonl(Path(args.answers))}
    grades = grade_all(items, answers)
    meta = {
        "stage": args.stage,
        "started_at": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "golden_version": version,
        "golden_sha256": golden_sha,
        "questions": len(items),
        "answers": "oracle" if args.oracle else args.answers,
    }
    run_dir = save_report(grades, meta, Path(args.out))
    print((run_dir / "report.md").read_text(encoding="utf-8"))
    print(f"saved: {run_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
