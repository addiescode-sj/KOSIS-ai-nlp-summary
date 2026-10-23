"""Eval run logging: per-step terminal lines, a log file under --log-dir, and timing in results and manifest."""

from __future__ import annotations

import json

from eval import run_eval
from kosis_chat.data.loader import sha256_of
from kosis_chat.llm.model import HCXClient


def _golden(table, tmp_path):
    golden = tmp_path / "golden" / "v1"
    golden.mkdir(parents=True)
    items = [
        {"id": f"q{i}", "kind": "answerable", "question": "q", "table": str(table), "table_sha256": sha256_of(table),
         "scope": {}}
        for i in (1, 2)
    ]
    (golden / "g.jsonl").write_text("".join(json.dumps(i) + "\n" for i in items), encoding="utf-8")
    return golden


def test_run_writes_step_logs_to_terminal_and_log_dir(pop_path, tmp_path, capsys):
    out, logs = tmp_path / "results", tmp_path / "logs"
    argv = ["--stage", "B0", "--golden", str(_golden(pop_path, tmp_path)), "--backend", "mock",
            "--interval", "0", "--out", str(out), "--log-dir", str(logs)]
    assert run_eval.main(argv) == 0

    terminal = capsys.readouterr().out
    log_files = list(logs.glob("*-B0.log"))
    assert len(log_files) == 1
    text = log_files[0].read_text(encoding="utf-8")
    for line in ("start: 2 questions", "[load 1/1] 101_FAKE_TBL_001.parquet", "data load done", "[call 1/2] q1 ok", "[call 2/2] q2 ok",
                 "done: total", "slowest: q"):
        assert line in text and line in terminal

    run_dir = next(out.iterdir())
    rows = [json.loads(line) for line in (run_dir / "results.jsonl").read_text(encoding="utf-8").splitlines()]
    assert all(r["attempts"] == 1 and r["latency_seconds"] >= 0 for r in rows)
    timing = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))["timing"]
    assert timing["calls"] == 2 and timing["retried_calls"] == 0
    assert timing["tables"][str(pop_path)]["cells"] > 0


def test_logging_handlers_are_removed_after_run(pop_path, tmp_path):
    import logging

    argv = ["--stage", "B0", "--golden", str(_golden(pop_path, tmp_path)), "--backend", "mock",
            "--interval", "0", "--out", str(tmp_path / "r"), "--log-dir", str(tmp_path / "logs")]
    before = list(logging.getLogger("kosis").handlers)
    run_eval.main(argv)
    assert logging.getLogger("kosis").handlers == before


def test_hcx_retries_are_counted_and_logged(mock_settings, server_factory, monkeypatch, caplog):
    from dataclasses import replace

    monkeypatch.setenv("HCX_API_KEY", "nv-dummy")
    srv = server_factory(status=429, body={"status": {"code": "42901"}})

    def fake_sleep(_):
        srv.status = 200
        srv.body = {"status": {"code": "20000"}, "result": {"message": {"role": "assistant", "content": "ok"}}}

    settings = replace(mock_settings, backend="hcx", base_url=srv.base_url, model_name="HCX-DASH-002")
    with caplog.at_level("WARNING", logger="kosis.llm"):
        result = HCXClient(settings, sleep=fake_sleep).generate("s", "u")
    assert result.status == "ok" and result.attempts == 2
    assert "HTTP 429, retry 1/4" in caplog.text
