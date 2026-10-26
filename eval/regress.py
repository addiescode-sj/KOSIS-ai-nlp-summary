"""Golden-set regression gate: compare a graded run against the pinned baseline question by question.

    python -m eval.regress --run eval/results/<run_id>              # compare with eval/baselines/stat.json
    python -m eval.regress --run eval/results/<run_id> --accept     # pin this run as the new baseline
    python -m eval.regress --run eval/results/<new> --base eval/results/<old>   # compare two local runs

Place in the flow: run_eval -> grade_run -> view_run -> regress. Verdict rank is PASS > PARTIAL > FAIL; a drop is a
Regression, a rise an Improvement. Exit code 1 when any question regressed. The baseline file keeps only verdicts,
error types, routes and run conditions (no cell values or answer text), so it is committed and shared by the team.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

from eval.scorer import read_jsonl

ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "eval" / "baselines" / "stat.json"
GRADER_FILES = (ROOT / "eval" / "grade.py", ROOT / "eval" / "scorer.py")

RANK = {"FAIL": 0, "PARTIAL": 1, "PASS": 2}
# Different values here mean a different exam or a different grader: comparing would count grader changes as regressions.
STRICT = ("golden_sha256", "scoring_version", "grader_sha256")
# Different values here are allowed (that is what an experiment changes) but are shown at the top of the report.
WATCH = ("pipeline", "model_name", "temperature", "prompt_version", "scope_mode", "serialization", "data_sha256")
REFUSAL_ERRORS = {"false_refusal", "missed_refusal"}
# Pipeline step (see the chatbot structure diagram) each error type points to.
STAGE = {
    "run_failure": "실행",
    "false_refusal": "1 판정", "missed_refusal": "1 판정", "wrong_reason": "1 판정", "insufficient_reason": "1 판정",
    "wrong_dims": "2 분석", "wrong_period": "2 분석", "assumed_condition": "2 분석",
    "missing_clarification": "2 분석", "disallowed_default": "2 분석", "unstated_default": "2 분석",
    "wrong_value": "3 계산", "wrong_unit": "3 계산",
    "unsupported_claim": "4-5 문장·검증", "missing_aux": "4-5 문장·검증", "no_source": "4-5 문장·검증",
}


def grader_sha256() -> str:
    h = hashlib.sha256()
    for path in GRADER_FILES:
        h.update(path.read_bytes())
    return h.hexdigest()


def latest_grades(run: Path) -> Path:
    dirs = sorted(p for p in (run / "grades").glob("*") if (p / "grades.jsonl").exists())
    if not dirs:
        raise SystemExit(f"no grades in {run}/grades — run `python -m eval.grade_run --run {run}` first")
    return dirs[-1]


def snapshot(run: Path) -> dict:
    """Verdicts and conditions of the latest grading of a run_eval folder (no data values)."""
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    grades_dir = latest_grades(run)
    report = json.loads((grades_dir / "report.json").read_text(encoding="utf-8"))
    model = manifest.get("model") or {}
    routes = {}
    for r in read_jsonl(run / "results.jsonl"):
        if r.get("run_index", 0) == 0:
            meta = r.get("decision_metadata") or {}
            routes[r["question_id"]] = f"{meta.get('route', '-')}/{meta.get('compose', '-')}"
    return {
        "run_id": manifest.get("run_id", run.name),
        "grades": grades_dir.name,
        "conditions": {
            "golden_sha256": manifest.get("golden_sha256"),
            "scoring_version": manifest.get("scoring_version"),
            # ponytail: older reports lack the hash; fall back to today's grader, which is right unless it changed since
            "grader_sha256": report.get("grader_sha256") or grader_sha256(),
            "pipeline": manifest.get("pipeline"),
            "model_name": model.get("model_name"),
            "temperature": model.get("temperature"),
            "prompt_version": manifest.get("prompt_version"),
            "scope_mode": manifest.get("scope_mode"),
            "serialization": manifest.get("serialization"),
            "data_sha256": manifest.get("data_sha256"),
        },
        "questions": {
            g["question_id"]: {"kind": g["kind"], "type": g.get("type"), "verdict": g["verdict"],
                               "error_types": g["error_types"], "route": routes.get(g["question_id"], "-/-")}
            for g in read_jsonl(grades_dir / "grades.jsonl")
        },
    }


def load_base(path: Path) -> dict:
    if path.is_dir():
        return snapshot(path)
    if not path.exists():
        raise SystemExit(f"no baseline at {path} — pin one with `python -m eval.regress --run <run> --accept`")
    return json.loads(path.read_text(encoding="utf-8"))


def compare(base: dict, new: dict) -> dict:
    mismatch = {k: (base["conditions"].get(k), new["conditions"].get(k)) for k in STRICT
                if base["conditions"].get(k) != new["conditions"].get(k)}
    if mismatch:
        return {"refused": mismatch}
    changed = {k: (base["conditions"].get(k), new["conditions"].get(k)) for k in WATCH
               if base["conditions"].get(k) != new["conditions"].get(k)}
    counts, regressions, improvements, by_group = Counter(), [], [], {}
    for qid, b in base["questions"].items():
        n = new["questions"].get(qid)
        if n is None:  # same golden sha256, so only a partial run can miss a question; grade_run marks it run_failure
            continue
        drop = RANK[b["verdict"]] - RANK[n["verdict"]]
        row = {"id": qid, "kind": n["kind"], "type": n["type"], "from": b["verdict"], "to": n["verdict"],
               "new_errors": sorted(set(n["error_types"]) - set(b["error_types"])), "route": [b["route"], n["route"]]}
        if drop > 0:
            row["stages"] = sorted({STAGE.get(e, "?") for e in row["new_errors"]})
            regressions.append(row)
            counts["regression"] += 1
        elif drop < 0:
            improvements.append(row)
            counts["improvement"] += 1
        else:
            counts["kept_pass" if n["verdict"] == "PASS" else "kept_not_pass"] += 1
        for group in (f"kind={n['kind']}", f"type={n['type'] or 'none'}"):
            c = by_group.setdefault(group, Counter())
            c["base"] += b["verdict"] == "PASS"
            c["new"] += n["verdict"] == "PASS"
            c["total"] += 1
    return {
        "base_run": base["run_id"], "new_run": new["run_id"], "changed_conditions": changed,
        "counts": {k: counts[k] for k in ("kept_pass", "improvement", "regression", "kept_not_pass")},
        "regressions": regressions, "improvements": improvements,
        "refusal_regressions": [r["id"] for r in regressions if set(r["new_errors"]) & REFUSAL_ERRORS],
        "by_group": {g: dict(c) for g, c in sorted(by_group.items())},
    }


def render(result: dict) -> str:
    if "refused" in result:
        lines = ["# 회귀 비교 거부 — 같은 시험지·같은 채점기가 아님", ""]
        lines += [f"- {k}: 기준 `{b}` ≠ 새 실행 `{n}`" for k, (b, n) in result["refused"].items()]
        lines += ["", "기준 실행을 지금 채점기로 `grade_run` 한 뒤 `--accept`로 기준선을 다시 고정하세요."]
        return "\n".join(lines) + "\n"
    c = result["counts"]
    lines = [f"# 회귀 비교 — {result['base_run']} → {result['new_run']}", ""]
    if result["changed_conditions"]:
        lines += ["바뀐 평가 조건 (의도한 변경인지 확인):", ""]
        lines += [f"- {k}: `{b}` → `{n}`" for k, (b, n) in result["changed_conditions"].items()]
        lines.append("")
    lines += [
        "| 전환 | 문항 수 |", "| --- | --- |",
        f"| 기존 PASS → 그대로 PASS | {c['kept_pass']} |",
        f"| 판정 상승 (Improvement) | {c['improvement']} |",
        f"| 판정 하락 (Regression) | {c['regression']} |",
        f"| 기존 PASS 아님 → 그대로 | {c['kept_not_pass']} |",
        "",
    ]
    if result["refusal_regressions"]:
        lines += [f"**거절 판단 회귀**: {', '.join(result['refusal_regressions'])}", ""]
    for title, rows in (("Regression", result["regressions"]), ("Improvement", result["improvements"])):
        if not rows:
            continue
        lines += [f"## {title}", "", "| 문항 | 판정 | 새 오류 | 원인 단계 | 경로 |", "| --- | --- | --- | --- | --- |"]
        for r in rows:
            route = r["route"][1] if r["route"][0] == r["route"][1] else f"{r['route'][0]} → {r['route'][1]}"
            lines.append(f"| {r['id']} | {r['from']} → {r['to']} | {', '.join(r['new_errors']) or '-'} "
                         f"| {', '.join(r.get('stages', [])) or '-'} | {route} |")
        lines.append("")
    lines += ["## PASS 수 (기준 → 새 실행)", "", "| 분해 | 기준 | 새 실행 | 문항 |", "| --- | --- | --- | --- |"]
    lines += [f"| {g} | {v['base']} | {v['new']} | {v['total']} |" for g, v in result["by_group"].items()]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m eval.regress", description=__doc__.splitlines()[0])
    parser.add_argument("--run", required=True, help="graded run_eval folder to check")
    parser.add_argument("--base", default=str(BASELINE), help="baseline json or another run folder (default: pinned baseline)")
    parser.add_argument("--accept", action="store_true", help="pin --run as the baseline instead of comparing")
    args = parser.parse_args(argv)
    run, base_path = Path(args.run), Path(args.base)
    new = snapshot(run)
    if args.accept:
        old = json.loads(base_path.read_text(encoding="utf-8"))["run_id"] if base_path.is_file() else None
        base_path.parent.mkdir(parents=True, exist_ok=True)
        base_path.write_text(json.dumps(new, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"baseline {base_path}: {old or '(none)'} -> {new['run_id']} ({len(new['questions'])} questions)")
        return 0
    result = compare(load_base(base_path), new)
    out = latest_grades(run)
    (out / "regress.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "regress.md").write_text(text := render(result), encoding="utf-8")
    print(text)
    print(f"saved: {out / 'regress.md'}")
    return 2 if "refused" in result else int(result["counts"]["regression"] > 0)


if __name__ == "__main__":
    sys.exit(main())
