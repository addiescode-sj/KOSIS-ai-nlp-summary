"""Grade a run_eval folder automatically: each saved StatAnswer against the golden copy kept in the run (no LLM).

    python -m eval.grade_run --run eval/results/<run_id>

The report goes inside the run folder (<run>/grades/<stamp>-<stage>/, data values: not committed). Besides the
eval.harness metrics it lists how each question was answered (rules, model fill, template, chatbot fallback) and how
many 'unsupported text numbers' are only digits inside labels ('1인 가구'), which eval/grade.py counts as errors.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import sys
from collections import Counter
from pathlib import Path

from eval.golden import load_items, read_cells
from eval.grade import grade
from eval.harness import _failed, save_report
from eval.scorer import read_jsonl

from kosis_ai.schemas import StatAnswer  # isort: skip

_TEXT_NUMBER = re.compile(r"unsupported text number '([^']+)'")
_NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?")


def label_numbers(answer: StatAnswer) -> set[str]:
    return {n for f in answer.result.facts for label in (f.dims or {}).values() for n in _NUMBER.findall(label)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m eval.grade_run", description=__doc__.splitlines()[0])
    parser.add_argument("--run", required=True, help="run folder written by eval.run_eval")
    parser.add_argument("--run-index", type=int, default=0, help="which repetition to grade when runs > 1 (default 0)")
    args = parser.parse_args(argv)
    run = Path(args.run)
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    version, items, golden_sha = load_items(run / "golden.jsonl")
    rows = {r["question_id"]: r for r in read_jsonl(run / "results.jsonl") if r["run_index"] == args.run_index}
    grades, grader_errors, routes, label_only, label_only_fails = [], [], Counter(), 0, []
    for item in items:
        row = rows.get(item.id)
        meta = (row or {}).get("decision_metadata") or {}
        routes[f"{meta.get('route', '-')}/{meta.get('compose', '-')}"] += 1
        if row is None or row.get("stat_answer") is None:  # a failed call: run_failure, as eval.harness grades it
            grades.append(_failed(item, (row or {}).get("call_error") or "no answer"))
            continue
        answer = StatAnswer.model_validate(row["stat_answer"])
        try:
            graded = grade(item, answer, read_cells(item.path))
        except ValueError as exc:  # eval/scorer.RULES has no place for an error type on this kind
            grader_errors.append(f"{item.id}: {exc}")
            continue
        labels = label_numbers(answer)
        unsupported = [d for d in graded.details if d.startswith("unsupported")]
        by_label = [d for d in unsupported if (m := _TEXT_NUMBER.match(d)) and m.group(1) in labels]
        label_only += len(by_label)
        if graded.verdict != "PASS" and graded.error_types == ["unsupported_claim"] and len(by_label) == len(unsupported):
            label_only_fails.append(item.id)  # would pass if label digits were allowed in text
        grades.append(graded)
    meta = {
        "stage": f"{manifest.get('stage', 'run')}-graded",
        "started_at": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "golden_version": version, "golden_sha256": golden_sha, "questions": len(items),
        "answers": str(run), "pipeline": manifest.get("pipeline", "chatbot"), "prompt_version": manifest.get("prompt_version"),
        "routes": dict(routes), "label_digit_numbers": label_only, "label_digit_only_fails": label_only_fails, "grader_errors": grader_errors,
    }
    out = save_report(grades, meta, run / "grades")
    extra = [
        "", f"- 파이프라인: {meta['pipeline']} · 프롬프트: {meta['prompt_version']}",
        "- 답한 경로(route/compose): " + ", ".join(f"{k} {v}" for k, v in sorted(routes.items())),
        f"- 숫자 오류 중 라벨 속 숫자('1인' 등)만인 것: {label_only}건",
        f"- 라벨 숫자만 때문에 FAIL인 문항: {len(label_only_fails)}개 {', '.join(label_only_fails)}",
    ] + ([f"- 채점기 오류(집계 제외): {e}" for e in grader_errors] or [])
    with (out / "report.md").open("a", encoding="utf-8") as f:
        f.write("\n".join(extra) + "\n")
    print((out / "report.md").read_text(encoding="utf-8"))
    print(f"saved: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
