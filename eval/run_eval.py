"""Run golden questions under fixed conditions and save raw answers, automatic checks and a manifest."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import re
import subprocess
import sys
import time
from pathlib import Path

from eval.scorer import KINDS, SCORING_VERSION, write_jsonl
from kosis_chat.chatbot import Chatbot
from kosis_chat.config import PROJECT_ROOT, Settings, load_env, load_settings, resolve_path
from kosis_chat.data.loader import load_table, sha256_of
from kosis_chat.llm.model import BACKENDS, LLMClient, build_client

REQUIRED_ITEM_KEYS = ("id", "kind", "question", "table", "table_sha256")
_NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?")


class GoldenError(ValueError):
    pass


def load_golden(path: Path) -> tuple[str, list[dict], str]:
    """Read a golden set (a .jsonl file or a version directory of .jsonl files) -> (version, items, sha256)."""
    files = sorted(path.glob("*.jsonl")) if path.is_dir() else [path]
    if not files:
        raise GoldenError(f"no .jsonl files in {path}")
    items: list[dict] = []
    digest = hashlib.sha256()
    for f in files:
        digest.update(f.read_bytes())
        for line_no, line in enumerate(f.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            item = json.loads(line)
            missing = [k for k in REQUIRED_ITEM_KEYS if k not in item]
            if missing:
                raise GoldenError(f"{f.name}:{line_no} missing keys {missing}")
            if item["kind"] not in KINDS:
                raise GoldenError(f"{f.name}:{line_no} kind {item['kind']!r} not in {KINDS}")
            items.append(item)
    ids = [i["id"] for i in items]
    if len(ids) != len(set(ids)):
        raise GoldenError("duplicate question ids in golden set")
    version = path.name if path.is_dir() else path.parent.name
    return version, items, digest.hexdigest()


def _number_forms(value: float) -> set[str]:
    forms = {f"{value:,.0f}", f"{value:.0f}"} if float(value).is_integer() else set()
    forms |= {f"{value:,.1f}", f"{value:.1f}", f"{value:,.2f}", f"{value:.2f}"}
    return forms


def automatic_checks(item: dict, answer_text: str, table_meta) -> dict:
    """Helper signals for the reviewer only; they never decide the verdict."""
    numbers = _NUMBER.findall(answer_text)
    expected_hits = []
    for fact in item.get("required_facts", []):
        value = fact.get("value")
        if isinstance(value, (int, float)):
            expected_hits.append({"value": value, "found_in_text": any(f in answer_text for f in _number_forms(value))})
    return {
        "source_mentioned": table_meta.survey_nm in answer_text or table_meta.tbl_nm in answer_text,
        "number_candidates": numbers,
        "expected_value_hits": expected_hits,
        "answer_chars": len(answer_text),
    }


def _git_commit() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=10)
        return out.stdout.strip() or "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def run(
    golden_path: Path,
    stage: str,
    settings: Settings,
    out_root: Path,
    client: LLMClient | None = None,
    runs: int | None = None,
    interval: float = 0.0,
    form_path: Path | None = None,
) -> Path:
    """Execute every golden question `runs` times, waiting `interval` seconds between calls; returns the run directory."""
    version, items, golden_sha = load_golden(golden_path)
    runs = runs or settings.eval_runs
    client = client or build_client(settings)
    form = form_path.read_text(encoding="utf-8-sig") if form_path else None

    bots: dict[str, Chatbot] = {}
    data_hashes: dict[str, str] = {}
    for item in items:
        table_path = resolve_path(item["table"])
        if item["table"] not in bots:
            actual = sha256_of(table_path) if table_path.exists() else "missing"
            if actual != item["table_sha256"]:
                raise GoldenError(f"{item['table']}: sha256 {actual} does not match golden {item['table_sha256']}")
            bots[item["table"]] = Chatbot(load_table(table_path), client=client, settings=settings, form=form)
            data_hashes[item["table"]] = actual

    started = dt.datetime.now().astimezone()
    run_id = f"{started:%Y%m%d-%H%M%S}-{stage}"
    run_dir = out_root / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    results = []
    for run_index in range(runs):
        for item in items:
            bot = bots[item["table"]]
            if interval and results:
                time.sleep(interval)  # stay under per-minute rate limits (HCX 429)
            answer = bot.ask(item["question"])
            failed = answer.status != "ok"
            results.append(
                {
                    "question_id": item["id"],
                    "kind": item["kind"],
                    "run_index": run_index,
                    "call_status": answer.status,
                    "call_error": answer.error,
                    "raw_answer": answer.text,
                    "automatic_checks": automatic_checks(item, answer.text, bot.table.meta),
                    # A failed call is a deterministic FAIL; everything else waits for a human.
                    "review_status": "confirmed" if failed else "pending",
                    "final_verdict": "FAIL" if failed else None,
                    "error_types": ["run_failure"] if failed else [],
                    "review_note": "",
                    "reviewer": "auto:run_failure" if failed else "",
                }
            )

    manifest = {
        "run_id": run_id,
        "stage": stage,
        "started_at": started.isoformat(timespec="seconds"),
        "git_commit": _git_commit(),
        "golden_version": version,
        "golden_sha256": golden_sha,
        "data_sha256": data_hashes,
        "scoring_version": SCORING_VERSION,
        "prompt_version": settings.prompt_version,
        # form.txt changes often, so its path, hash and a copy (form.txt in the run dir) are kept with the run.
        "form": {"path": str(form_path), "sha256": sha256_of(form_path)} if form_path else None,
        "model": settings.model_params(),
        "runs": runs,
        "questions": len(items),
        # Mock runs are pre-tests and stay out of eval-history.
        "pretest": settings.backend == "mock",
    }
    write_jsonl(run_dir / "results.jsonl", results)
    if form is not None:
        (run_dir / "form.txt").write_text(form, encoding="utf-8")
    (run_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (run_dir / "golden.jsonl").write_text(
        "".join(json.dumps(i, ensure_ascii=False) + "\n" for i in items), encoding="utf-8"
    )
    return run_dir


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m eval.run_eval", description="Run the golden set (calls the model).")
    parser.add_argument("--stage", required=True, help="stage name, e.g. B0")
    parser.add_argument("--golden", default="eval/golden/v1", help="golden version directory or .jsonl file")
    parser.add_argument("--config", default=None)
    parser.add_argument("--backend", choices=BACKENDS, default=None)
    parser.add_argument("--model", default=None, help="override the HCX model name, e.g. HCX-007")
    parser.add_argument(
        "--thinking", choices=["none", "low", "medium", "high"], default=None, help="override thinking effort (HCX-007 only)"
    )
    parser.add_argument("--max-tokens", type=int, default=None, help="override max output tokens (raise it when thinking is on)")
    parser.add_argument("--runs", type=int, default=None, help="override eval.runs (changing it requires re-running all stages)")
    parser.add_argument("--interval", type=float, default=0.0, help="seconds to wait between model calls (avoids HTTP 429)")
    parser.add_argument("--form", default=None, help="text file with answer-format instructions appended to each question")
    parser.add_argument("--out", default="eval/results")
    args = parser.parse_args(argv)

    load_env()
    settings = load_settings(
        args.config, backend=args.backend, model_name=args.model, thinking_level=args.thinking, max_tokens=args.max_tokens
    )
    run_dir = run(
        resolve_path(args.golden),
        args.stage,
        settings,
        resolve_path(args.out),
        runs=args.runs,
        interval=args.interval,
        form_path=resolve_path(args.form) if args.form else None,
    )
    print(f"saved: {run_dir}")
    print(f"next: python -m eval.review export --run {run_dir.relative_to(PROJECT_ROOT) if run_dir.is_relative_to(PROJECT_ROOT) else run_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
