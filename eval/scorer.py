"""Scoring rules (verdict order FAIL -> PASS -> PARTIAL) and aggregation of confirmed verdicts only."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

from kosis_ai.schemas import QUESTION_KINDS as KINDS

SCORING_VERSION = "baseline-scoring-v1"

KIND_ANSWERABLE, KIND_NOT_ANSWERABLE, KIND_CLARIFY, KIND_DEFAULT_ALLOWED = KINDS
VERDICTS = ("PASS", "PARTIAL", "FAIL")

_FACT_ERRORS = {"wrong_value", "wrong_unit", "wrong_period", "wrong_dims"}

# Per kind: error types that force FAIL, and error types that make an otherwise passing answer PARTIAL.
RULES: dict[str, dict[str, set[str]]] = {
    KIND_ANSWERABLE: {
        "fail": _FACT_ERRORS | {"unsupported_claim", "false_refusal", "run_failure"},
        "partial": {"missing_aux", "no_source"},
    },
    KIND_NOT_ANSWERABLE: {
        "fail": {"missed_refusal", "unsupported_claim", "wrong_reason", "run_failure"},
        "partial": {"insufficient_reason"},
    },
    KIND_CLARIFY: {
        "fail": {"assumed_condition", "unsupported_claim", "run_failure"},
        "partial": {"missing_clarification"},
    },
    KIND_DEFAULT_ALLOWED: {
        "fail": _FACT_ERRORS | {"disallowed_default", "unsupported_claim", "run_failure"},
        "partial": {"unstated_default", "no_source", "missing_aux"},
    },
}
ERROR_TYPES = sorted(set().union(*(r["fail"] | r["partial"] for r in RULES.values())))


def decide_verdict(kind: str, error_types: list[str] | set[str]) -> str:
    """FAIL conditions first, then PASS (no errors), else PARTIAL — so verdicts never overlap."""
    if kind not in RULES:
        raise ValueError(f"unknown question kind {kind!r}; expected one of {KINDS}")
    errors = set(error_types)
    unknown = errors - (RULES[kind]["fail"] | RULES[kind]["partial"])
    if unknown:
        raise ValueError(f"error types {sorted(unknown)} do not apply to kind {kind!r}")
    if errors & RULES[kind]["fail"]:
        return "FAIL"
    if not errors:
        return "PASS"
    return "PARTIAL"


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _pass_rate(rows: list[dict]) -> dict:
    confirmed = [r for r in rows if r.get("review_status") == "confirmed"]
    passed = sum(1 for r in confirmed if r.get("final_verdict") == "PASS")
    return {
        "total": len(rows),
        "confirmed": len(confirmed),
        "pass": passed,
        "partial": sum(1 for r in confirmed if r.get("final_verdict") == "PARTIAL"),
        "fail": sum(1 for r in confirmed if r.get("final_verdict") == "FAIL"),
        # Denominator is ALL rows of the kind, so pending rows are never counted as correct or dropped.
        "pass_rate": passed / len(rows) if rows else None,
    }


def aggregate(results: list[dict]) -> dict:
    """Summary over every result row; provisional when any row is still pending review."""
    pending = sum(1 for r in results if r.get("review_status") != "confirmed")
    by_kind = {kind: _pass_rate([r for r in results if r.get("kind") == kind]) for kind in KINDS}
    errors = Counter(e for r in results if r.get("review_status") == "confirmed" for e in r.get("error_types", []))
    run_indexes = sorted({r.get("run_index", 0) for r in results})
    per_run = [_pass_rate([r for r in results if r.get("run_index", 0) == i])["pass_rate"] for i in run_indexes]
    per_run = [p for p in per_run if p is not None]
    return {
        "scoring_version": SCORING_VERSION,
        "total": len(results),
        "confirmed": len(results) - pending,
        "pending": pending,
        "provisional": pending > 0,
        "by_kind": by_kind,
        "unsupported_claims": errors.get("unsupported_claim", 0),
        "run_failures": sum(1 for r in results if r.get("call_status") != "ok"),
        "error_types": dict(sorted(errors.items())),
        "quality": quality_summary(results),
        "runs": len(run_indexes),
        "overall_pass_rate_per_run": {
            "mean": statistics.fmean(per_run) if per_run else None,
            "min": min(per_run) if per_run else None,
            "max": max(per_run) if per_run else None,
        },
    }


def quality_summary(results: list[dict]) -> dict:
    """Human natural-language quality (T1-21): mean over confirmed rows that have a score, with n/N stated."""
    confirmed = [r for r in results if r.get("review_status") == "confirmed"]
    scored = [r for r in confirmed if r.get("quality_score") is not None]
    return {
        "scored": len(scored), "confirmed": len(confirmed),
        "mean": round(statistics.fmean(r["quality_score"] for r in scored), 2) if scored else None,
        "zero": [r["question_id"] for r in scored if r["quality_score"] == 0],
    }


def quality_agreement(results: list[dict], judged: list[dict]) -> dict:
    """Human vs judge quality, joined on (question_id, run_index): agreement, 3x3 confusion, per group, linear-weighted kappa.

    The judge score is the reported one (capped at 1 by format checks). Refusals and clarifications are a separate
    group because they use a separate rubric.
    """
    judge = {(j["question_id"], j.get("run_index", 0)): j["score"] for j in judged if j.get("score") is not None}
    pairs = [(r["quality_score"], judge[k], r.get("kind"))
             for r in results if r.get("quality_score") is not None
             and (k := (r["question_id"], r.get("run_index", 0))) in judge]

    def stats(ps: list[tuple]) -> dict:
        n = len(ps)
        if not n:
            return {"n": 0}
        confusion = [[sum(1 for h, j, _ in ps if h == a and j == b) for b in range(3)] for a in range(3)]
        observed = sum(1 - abs(h - j) / 2 for h, j, _ in ps) / n
        rows, cols = [sum(confusion[a]) / n for a in range(3)], [sum(confusion[a][b] for a in range(3)) / n for b in range(3)]
        expected = sum(rows[a] * cols[b] * (1 - abs(a - b) / 2) for a in range(3) for b in range(3))
        kappa = None if expected == 1 else round((observed - expected) / (1 - expected), 2)
        return {"n": n, "agreement": round(sum(1 for h, j, _ in ps if h == j) / n, 2), "weighted_kappa": kappa,
                "confusion_human_by_judge": confusion}

    refusal = {KIND_NOT_ANSWERABLE, KIND_CLARIFY}
    return {"all": stats(pairs), "answered": stats([p for p in pairs if p[2] not in refusal]),
            "refused_or_clarify": stats([p for p in pairs if p[2] in refusal])}


HISTORY_HEADER = (
    "| 단계 | 바꾼 것 | 문항(전체/확정/미검토) | 잠정 | 답변 가능 PASS | 답변 불가 PASS | 되묻기 PASS "
    "| 기본값 PASS | 근거 없는 주장 | 실행 실패 | runs | 모델 | manifest |\n"
    "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |\n"
)


def _fmt(kind_summary: dict) -> str:
    if not kind_summary["total"]:
        return "-"
    return f"{kind_summary['pass']}/{kind_summary['total']} ({kind_summary['pass_rate']:.0%})"


def history_row(summary: dict, stage: str, change: str, manifest_ref: str, model: str = "-") -> str:
    k = summary["by_kind"]
    return (
        f"| {stage} | {change} | {summary['total']}/{summary['confirmed']}/{summary['pending']} "
        f"| {'예' if summary['provisional'] else '아니오'} | {_fmt(k[KIND_ANSWERABLE])} | {_fmt(k[KIND_NOT_ANSWERABLE])} "
        f"| {_fmt(k[KIND_CLARIFY])} | {_fmt(k[KIND_DEFAULT_ALLOWED])} | {summary['unsupported_claims']} "
        f"| {summary['run_failures']} | {summary['runs']} | {model} | {manifest_ref} |\n"
    )


def append_history(path: Path, row: str) -> None:
    """Add one row to eval-history.md (aggregate numbers only, safe to commit)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text("# Eval history\n\n같은 골든셋·데이터·모델·채점 기준으로 단계별 결과를 누적한다.\n\n" + HISTORY_HEADER, encoding="utf-8")
    with path.open("a", encoding="utf-8", newline="\n") as f:
        f.write(row)


def main(argv: list[str] | None = None) -> int:
    from kosis_chat.config import resolve_path

    parser = argparse.ArgumentParser(prog="python -m eval.scorer", description="Aggregate confirmed verdicts (offline).")
    parser.add_argument("--run", required=True, help="eval/results/<run_id> directory")
    parser.add_argument("--append-history", action="store_true", help="append a row to docs/reports/eval-history.md")
    parser.add_argument("--change", default="", help="what this stage changed (history column)")
    args = parser.parse_args(argv)

    run_dir = resolve_path(args.run)
    results = read_jsonl(run_dir / "results.jsonl")
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    summary = aggregate(results)
    if (run_dir / "quality.jsonl").exists():  # T1-26 judge output: compare with the blind human scores
        summary["quality_agreement"] = quality_agreement(results, read_jsonl(run_dir / "quality.jsonl"))
    (run_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if summary["provisional"]:
        print(f"[잠정] 미검토 {summary['pending']}개가 남아 있습니다.", file=sys.stderr)
    if args.append_history and manifest.get("pretest", manifest["model"]["backend"] == "mock"):
        print(f"[사전 테스트] backend={manifest['model']['backend']} 결과는 eval-history.md에 기록하지 않습니다.", file=sys.stderr)
    elif args.append_history:
        model = f"{manifest['model']['backend']}:{manifest['model']['model_name']}"
        row = history_row(summary, manifest["stage"], args.change or manifest["stage"], f"{manifest['run_id']}", model)
        append_history(resolve_path("docs/reports/eval-history.md"), row)
    return 0


if __name__ == "__main__":
    sys.exit(main())
