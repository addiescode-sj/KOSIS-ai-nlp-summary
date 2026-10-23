"""Run golden questions under fixed conditions and save raw answers, automatic checks and a manifest."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import logging
import re
import subprocess
import sys
import time
from pathlib import Path

from eval.scorer import KINDS, SCORING_VERSION, write_jsonl
from kosis_chat.chatbot import Chatbot
from kosis_chat.config import PROJECT_ROOT, Settings, load_env, load_settings, resolve_path
from kosis_chat.data.loader import load_table, sha256_of
from kosis_chat.data.scope import MODES as SCOPE_MODES
from kosis_chat.llm.model import BACKENDS, LLMClient, build_client

REQUIRED_ITEM_KEYS = ("id", "kind", "question", "table", "table_sha256")
log = logging.getLogger("kosis.eval")
LOG_FORMAT = "%(asctime)s %(levelname)-7s %(message)s"
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
    scope_mode: str = "oracle",
) -> Path:
    """Execute every golden question `runs` times, waiting `interval` seconds between calls; returns the run directory.
    scope_mode picks the table part each question gets (oracle: the item's `scope`, auto: from the question, full)."""
    version, items, golden_sha = load_golden(golden_path)
    if scope_mode not in SCOPE_MODES:
        raise GoldenError(f"unknown scope mode {scope_mode!r}; expected one of {SCOPE_MODES}")
    if scope_mode == "oracle":
        no_scope = [i["id"] for i in items if not isinstance(i.get("scope"), dict)]
        if no_scope:
            raise GoldenError(f"--scope oracle needs a `scope` in every item; missing: {', '.join(no_scope)}")
    runs = runs or settings.eval_runs
    client = client or build_client(settings)
    form = form_path.read_text(encoding="utf-8-sig") if form_path else None

    log.info(
        "start: %d questions x %d runs, %d tables, model=%s:%s prompt=%s serialization=%s scope=%s interval=%.1fs",
        len(items), runs, len({i["table"] for i in items}), settings.backend, settings.model_name,
        settings.prompt_version, settings.serialization, scope_mode, interval,
    )
    run_started = time.perf_counter()
    bots: dict[str, Chatbot] = {}
    data_hashes: dict[str, str] = {}
    load_timing: dict[str, dict] = {}
    table_names = list(dict.fromkeys(i["table"] for i in items))
    for n, name in enumerate(table_names, start=1):
        table_path = resolve_path(name)
        t0 = time.perf_counter()
        actual = sha256_of(table_path) if table_path.exists() else "missing"
        if actual != next(i["table_sha256"] for i in items if i["table"] == name):
            expected = next(i["table_sha256"] for i in items if i["table"] == name)
            raise GoldenError(f"{name}: sha256 {actual} does not match golden {expected}")
        t1 = time.perf_counter()
        table = load_table(table_path)
        t2 = time.perf_counter()
        bots[name] = Chatbot(table, client=client, settings=settings, form=form, scope_mode=scope_mode)  # clean
        t3 = time.perf_counter()
        data_hashes[name] = actual
        load_timing[name] = {
            "cells": len(table.cells),
            "hash_s": round(t1 - t0, 3),
            "load_s": round(t2 - t1, 3),
            "prepare_s": round(t3 - t2, 3),
        }
        log.info(
            "[load %d/%d] %s: %s cells | hash %.2fs, load %.2fs, prepare %.2fs",
            n, len(table_names), Path(name).name, f"{len(table.cells):,}", t1 - t0, t2 - t1, t3 - t2,
        )
    load_total = time.perf_counter() - run_started
    log.info("data load done: %d tables in %.2fs", len(table_names), load_total)

    started = dt.datetime.now().astimezone()
    run_id = f"{started:%Y%m%d-%H%M%S}-{stage}"
    run_dir = out_root / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    results = []
    waited = 0.0
    total_calls = runs * len(items)
    for run_index in range(runs):
        for item in items:
            bot = bots[item["table"]]
            if interval and results:
                time.sleep(interval)  # stay under per-minute rate limits (HCX 429)
                waited += interval
            answer = bot.ask(item["question"], item.get("scope"))
            failed = answer.status != "ok"
            log.log(
                logging.WARNING if failed else logging.INFO,
                "[call %d/%d] %s %s %.2fs (attempts %d, scope %s cells / %s chars, tokens %s/%s)%s",
                len(results) + 1, total_calls, item["id"], answer.status, answer.latency_seconds, answer.attempts,
                f"{answer.scope.get('cells', 0):,}", f"{answer.scope.get('table_chars', 0):,}",
                answer.usage.get("promptTokens", "-"), answer.usage.get("completionTokens", "-"),
                f" {answer.error[:120]}" if failed and answer.error else "",
            )
            results.append(
                {
                    "question_id": item["id"],
                    "kind": item["kind"],
                    "run_index": run_index,
                    "call_status": answer.status,
                    "call_error": answer.error,
                    "raw_answer": answer.text,
                    "latency_seconds": round(answer.latency_seconds, 3),
                    "attempts": answer.attempts,
                    "usage": answer.usage,
                    "scope": answer.scope,
                    "automatic_checks": automatic_checks(item, answer.text, bot.table.meta),
                    # A failed call is a deterministic FAIL; everything else waits for a human.
                    "review_status": "confirmed" if failed else "pending",
                    "final_verdict": "FAIL" if failed else None,
                    "error_types": ["run_failure"] if failed else [],
                    "review_note": "",
                    "reviewer": "auto:run_failure" if failed else "",
                }
            )

    timing = _timing_summary(results, load_timing, load_total, waited, time.perf_counter() - run_started)
    _log_timing(timing, results)
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
        "serialization": settings.serialization,
        "scope_mode": scope_mode,
        # form.txt changes often, so its path, hash and a copy (form.txt in the run dir) are kept with the run.
        "form": {"path": str(form_path), "sha256": sha256_of(form_path)} if form_path else None,
        "model": settings.model_params(),
        "runs": runs,
        "questions": len(items),
        # Mock runs are pre-tests and stay out of eval-history.
        "pretest": settings.backend == "mock",
        "timing": timing,
    }
    write_jsonl(run_dir / "results.jsonl", results)
    if form is not None:
        (run_dir / "form.txt").write_text(form, encoding="utf-8")
    (run_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (run_dir / "golden.jsonl").write_text(
        "".join(json.dumps(i, ensure_ascii=False) + "\n" for i in items), encoding="utf-8"
    )
    log.info("saved: %s", run_dir)
    return run_dir


def _timing_summary(results: list[dict], load_timing: dict, load_total: float, waited: float, total: float) -> dict:
    latencies = [r["latency_seconds"] for r in results]
    calls = sum(latencies)
    return {
        "total_s": round(total, 2),
        "data_load_s": round(load_total, 2),
        "api_calls_s": round(calls, 2),
        "interval_wait_s": round(waited, 2),
        "calls": len(results),
        "call_avg_s": round(calls / len(results), 2) if results else None,
        "call_max_s": round(max(latencies), 2) if latencies else None,
        "retried_calls": sum(1 for r in results if r["attempts"] > 1),
        "tables": load_timing,
    }


def _log_timing(timing: dict, results: list[dict]) -> None:
    statuses: dict[str, int] = {}
    for r in results:
        statuses[r["call_status"]] = statuses.get(r["call_status"], 0) + 1
    log.info(
        "done: total %.1fs = data load %.1fs + API calls %.1fs + interval wait %.1fs (+ other)",
        timing["total_s"], timing["data_load_s"], timing["api_calls_s"], timing["interval_wait_s"],
    )
    log.info(
        "calls: %d, avg %ss, max %ss, retried %d, status %s",
        timing["calls"], timing["call_avg_s"], timing["call_max_s"], timing["retried_calls"], statuses,
    )
    for r in sorted(results, key=lambda r: r["latency_seconds"], reverse=True)[:3]:
        log.info("slowest: %s %.2fs (attempts %d)", r["question_id"], r["latency_seconds"], r["attempts"])


def setup_logging(log_dir: Path, stage: str) -> tuple[Path, list[logging.Handler]]:
    """Log kosis.* to the terminal and to <log_dir>/<time>-<stage>.log; returns the file path and handlers to close."""
    log_dir.mkdir(parents=True, exist_ok=True)
    path = log_dir / f"{dt.datetime.now():%Y%m%d-%H%M%S}-{stage}.log"
    formatter = logging.Formatter(LOG_FORMAT, datefmt="%H:%M:%S")
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout), logging.FileHandler(path, encoding="utf-8")]
    root = logging.getLogger("kosis")
    root.setLevel(logging.INFO)
    for h in handlers:
        h.setFormatter(formatter)
        root.addHandler(h)
    return path, handlers


def close_logging(handlers: list[logging.Handler]) -> None:
    root = logging.getLogger("kosis")
    for h in handlers:
        root.removeHandler(h)
        h.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m eval.run_eval", description="Run the golden set (calls the model).")
    parser.add_argument("--stage", required=True, help="stage name, e.g. B0")
    parser.add_argument("--golden", default="eval/golden/stat_golden.jsonl", help="golden .jsonl file or a directory of them")
    parser.add_argument("--config", default=None)
    parser.add_argument("--backend", choices=BACKENDS, default=None)
    parser.add_argument("--model", default=None, help="override the HCX model name, e.g. HCX-007")
    parser.add_argument(
        "--thinking", choices=["none", "low", "medium", "high"], default=None, help="override thinking effort (HCX-007 only)"
    )
    parser.add_argument("--max-tokens", type=int, default=None, help="override max output tokens (raise it when thinking is on)")
    parser.add_argument("--runs", type=int, default=None, help="override eval.runs (changing it requires re-running all stages)")
    parser.add_argument("--interval", type=float, default=1.0, help="seconds to wait between model calls (avoids HTTP 429; default 1)")
    parser.add_argument("--form", default=None, help="text file with answer-format instructions appended to each question")
    parser.add_argument("--serialization", choices=["long", "wide"], default=None, help="table text style (default: model.yaml)")
    parser.add_argument(
        "--scope", choices=SCOPE_MODES, default="oracle",
        help="table part per question: oracle (golden `scope`), auto (from the question), full (default: oracle)",
    )
    parser.add_argument("--out", default="eval/results")
    parser.add_argument("--log-dir", default="logs", help="folder for the run log file (default: logs/)")
    args = parser.parse_args(argv)

    load_env()
    settings = load_settings(
        args.config,
        backend=args.backend,
        model_name=args.model,
        thinking_level=args.thinking,
        max_tokens=args.max_tokens,
        serialization=args.serialization,
    )
    log_path, handlers = setup_logging(resolve_path(args.log_dir), args.stage)
    try:
        run_dir = run(
            resolve_path(args.golden),
            args.stage,
            settings,
            resolve_path(args.out),
            runs=args.runs,
            interval=args.interval,
            form_path=resolve_path(args.form) if args.form else None,
            scope_mode=args.scope,
        )
    except Exception:
        log.exception("run failed")
        raise
    finally:
        close_logging(handlers)
    print(f"log: {log_path}")
    print(f"next: python -m eval.review export --run {run_dir.relative_to(PROJECT_ROOT) if run_dir.is_relative_to(PROJECT_ROOT) else run_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
