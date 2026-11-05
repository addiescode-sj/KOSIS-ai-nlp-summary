"""Parity gate: identical runs pass, real changes are reported by path, volatile fields and --ignore paths are skipped."""

from __future__ import annotations

import copy
import json

import pytest

from eval import parity


def _row(question_id: str, latency: float = 0.1, start_ms: float = 0.0) -> dict:
    """A small results.jsonl row with a subset of the keys run_eval writes for --pipeline stat."""
    return {
        "question_id": question_id,
        "kind": "answerable",
        "run_index": 0,
        "call_status": "ok",
        "raw_answer": "2020년 인구는 100명입니다.",
        "decision_metadata": {
            "pipeline": "stat", "route": "rules", "compose": "model",
            "spans": [
                {"name": "judge", "start_ms": start_ms, "duration_ms": latency * 10, "status": "ok", "attributes": {}},
                {"name": "compose", "start_ms": start_ms + 1, "duration_ms": latency * 20, "status": "ok",
                 "attributes": {"gen_ai.request.model": "mock", "llm.attempts": 1}},
            ],
        },
        "judge": {"rule": None, "outcome": "answer", "final": False},
        "stat_answer": {"text": "2020년 인구는 100명입니다.", "facts": [{"value": 100.0, "unit": "명"}],
                        "source": {"org_id": "101", "tbl_id": "DT_1", "survey_nm": "s", "tbl_nm": "t"}},
        "latency_seconds": latency,
        "retry_wait_seconds": 0.0,
        "attempts": 1,
        "usage": {"promptTokens": 10},
        "event_ids": ["E0000002", "E0000003"],
    }


def _write(tmp_path, name: str, rows: list[dict], manifest: bool = True):
    """A run folder; with a completed manifest.json for exactly these rows unless manifest=False."""
    run = tmp_path / name
    run.mkdir()
    (run / "results.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    if manifest:
        _manifest(run)
    return run


def test_identical_runs(tmp_path, capsys):
    rows = [_row("Q1"), _row("Q2")]
    base, run = _write(tmp_path, "a", rows), _write(tmp_path, "b", copy.deepcopy(rows))
    assert parity.main(["--base", str(base), "--run", str(run)]) == 0
    assert capsys.readouterr().out.strip() == "identical"


def test_volatile_fields_are_ignored(tmp_path):
    base = parity.load_rows(_write(tmp_path, "a", [_row("Q1", latency=0.1, start_ms=0.0)]))
    changed = _row("Q1", latency=9.9, start_ms=5.0)  # span start_ms / duration_ms change too
    changed.update(attempts=3, usage={"promptTokens": 99, "completionTokens": 5}, retry_wait_seconds=1.002,
                   started_at="2026-11-05T10:00:00+09:00")
    changed["stat_answer"]["source"]["fetched_at"] = "2026-11-05"
    run = parity.load_rows(_write(tmp_path, "b", [changed]))
    assert parity.compare(base, run)["identical"]


def test_differing_field_is_reported_by_path(tmp_path, capsys):
    changed = _row("Q2")
    changed["stat_answer"]["facts"][0]["value"] = 101.0
    changed["decision_metadata"]["route"] = "llm_fill"
    base, run = _write(tmp_path, "a", [_row("Q1"), _row("Q2")]), _write(tmp_path, "b", [_row("Q1"), changed])
    assert parity.main(["--base", str(base), "--run", str(run)]) == 1
    out = capsys.readouterr().out
    assert "Q2: 2 path(s) differ" in out
    assert "stat_answer.facts[0].value: base=100.0 run=101.0" in out
    assert 'decision_metadata.route: base="rules" run="llm_fill"' in out
    assert "Q1:" not in out


def test_type_change_and_key_order_are_differences(tmp_path):
    base = parity.load_rows(_write(tmp_path, "a", [_row("Q1")]))
    retyped = _row("Q1")
    retyped["stat_answer"]["facts"][0]["value"] = 100  # int instead of float: different bytes
    reordered = _row("Q1")
    reordered["judge"] = {"final": False, "outcome": "answer", "rule": None}
    assert parity.compare(base, parity.load_rows(_write(tmp_path, "b", [retyped])))["diffs"]
    found = parity.compare(base, parity.load_rows(_write(tmp_path, "c", [reordered])))["diffs"][("Q1", 0)]
    assert found[0][0] == "judge (key order)"


def test_missing_question_is_reported(tmp_path, capsys):
    base, run = _write(tmp_path, "a", [_row("Q1"), _row("Q2")]), _write(tmp_path, "b", [_row("Q1"), _row("Q3")])
    assert parity.main(["--base", str(base), "--run", str(run)]) == 1
    out = capsys.readouterr().out
    assert "only in base: Q2" in out
    assert "only in run: Q3" in out


def test_missing_key_and_list_item(tmp_path):
    changed = _row("Q1")
    del changed["judge"]["rule"]
    changed["decision_metadata"]["spans"].append({"name": "render"})
    result = parity.compare(parity.load_rows(_write(tmp_path, "a", [_row("Q1")])),
                            parity.load_rows(_write(tmp_path, "b", [changed])))
    paths = {path: (old, new) for path, old, new in result["diffs"][("Q1", 0)]}
    assert paths["judge.rule"] == (None, parity.MISSING)
    assert paths["decision_metadata.spans[2]"] == (parity.MISSING, {"name": "render"})


def test_nested_list_wildcard_ignore(tmp_path):
    changed = _row("Q1")
    for span in changed["decision_metadata"]["spans"]:
        span["status"] = "error"
    base = parity.load_rows(_write(tmp_path, "a", [_row("Q1")]))
    run = parity.load_rows(_write(tmp_path, "b", [changed]))
    found = parity.compare(base, run)["diffs"][("Q1", 0)]
    assert [p for p, _, _ in found] == ["decision_metadata.spans[0].status", "decision_metadata.spans[1].status"]
    assert parity.compare(base, run, ("decision_metadata.spans[*].status",))["identical"]
    only_first = parity.compare(base, run, ("decision_metadata.spans[0].status",))["diffs"][("Q1", 0)]
    assert [p for p, _, _ in only_first] == ["decision_metadata.spans[1].status"]


def test_dotted_attribute_keys_are_quoted(tmp_path):
    changed = _row("Q1")
    changed["decision_metadata"]["spans"][1]["attributes"]["gen_ai.request.model"] = "HCX-DASH-002"
    result = parity.compare(parity.load_rows(_write(tmp_path, "a", [_row("Q1")])),
                            parity.load_rows(_write(tmp_path, "b", [changed])))
    assert result["diffs"][("Q1", 0)][0][0] == 'decision_metadata.spans[1].attributes["gen_ai.request.model"]'


def test_cli_ignore_flag_and_exit_codes(tmp_path, capsys):
    changed = _row("Q1")
    changed["raw_answer"] = "x" * 500
    base, run = _write(tmp_path, "a", [_row("Q1")]), _write(tmp_path, "b", [changed])
    assert parity.main(["--base", str(base), "--run", str(run)]) == 1
    line = next(l for l in capsys.readouterr().out.splitlines() if l.startswith("  raw_answer"))
    assert line.endswith("...")  # long values are cut
    assert len(line) < 200
    assert parity.main(["--base", str(base), "--run", str(run), "--ignore", "raw_answer"]) == 0
    assert parity.main(["--base", str(base), "--run", str(tmp_path / "nowhere")]) == 2
    with pytest.raises(SystemExit):
        parity.main(["--base", str(base), "--run", str(run), "--ignore", "a[x]"])


def test_runs_are_keyed_by_run_index(tmp_path):
    second = _row("Q1")
    second["run_index"] = 1
    with pytest.raises(parity.ParityError):
        parity.load_rows(_write(tmp_path, "dup", [_row("Q1"), _row("Q1")]))
    rows = parity.load_rows(_write(tmp_path, "a", [_row("Q1"), second]))
    assert list(rows) == [("Q1", 0), ("Q1", 1)]


@pytest.mark.parametrize("bad", ["x", None, [1]])
def test_bad_run_index_is_an_input_error(tmp_path, bad):
    broken = _row("Q1")
    broken["run_index"] = bad
    run = _write(tmp_path, "a", [broken], manifest=False)
    with pytest.raises(parity.ParityError, match="bad run_index"):
        parity.load_rows(run)
    assert parity.main(["--base", str(run), "--run", str(run)]) == 2


def test_nan_equals_nan_and_signed_zero_differs(tmp_path):
    nan_row, neg_zero, pos_zero = _row("Q1"), _row("Q1"), _row("Q1")
    nan_row["stat_answer"]["facts"][0]["value"] = float("nan")
    neg_zero["stat_answer"]["facts"][0]["value"] = -0.0
    pos_zero["stat_answer"]["facts"][0]["value"] = 0.0
    nan_a = parity.load_rows(_write(tmp_path, "a", [nan_row]))
    nan_b = parity.load_rows(_write(tmp_path, "b", [copy.deepcopy(nan_row)]))
    assert parity.compare(nan_a, nan_b)["identical"]
    zeros = parity.compare(parity.load_rows(_write(tmp_path, "c", [neg_zero])),
                           parity.load_rows(_write(tmp_path, "d", [pos_zero])))
    assert [p for p, _, _ in zeros["diffs"][("Q1", 0)]] == ["stat_answer.facts[0].value"]


def test_ignored_key_moving_is_not_key_order(tmp_path):
    moved = _row("Q1", latency=9.9)
    moved = {"latency_seconds": moved.pop("latency_seconds"), **moved}  # volatile key moves to the front
    result = parity.compare(parity.load_rows(_write(tmp_path, "a", [_row("Q1")])),
                            parity.load_rows(_write(tmp_path, "b", [moved])))
    assert result["identical"]


def test_row_order_change_is_reported(tmp_path, capsys):
    base, run = _write(tmp_path, "a", [_row("Q1"), _row("Q2")]), _write(tmp_path, "b", [_row("Q2"), _row("Q1")])
    result = parity.compare(parity.load_rows(base), parity.load_rows(run))
    assert result["order_changed"] and not result["identical"] and not result["diffs"]
    assert parity.main(["--base", str(base), "--run", str(run)]) == 1
    assert "row order differs" in capsys.readouterr().out


def test_width_cuts_values(tmp_path, capsys):
    changed = _row("Q1")
    changed["raw_answer"] = "x" * 500
    base, run = _write(tmp_path, "a", [_row("Q1")]), _write(tmp_path, "b", [changed])
    assert parity.main(["--base", str(base), "--run", str(run), "--width", "20"]) == 1
    line = next(l for l in capsys.readouterr().out.splitlines() if l.startswith("  raw_answer"))
    assert line.endswith('run="' + "x" * 16 + "...")  # 20 characters: quote, 16 x, "..."


def test_parse_path():
    assert parity.parse_path("decision_metadata.spans[*].start_ms") == ("decision_metadata", "spans", "*", "start_ms")
    assert parity.parse_path("a[0][*]") == ("a", 0, "*")
    assert parity.parse_path('attributes["gen_ai.request.model"].x') == ("attributes", "gen_ai.request.model", "x")
    for bad in ["a..b", "", "a.", "a[x]", 'a["b"', 'a["\\q"]']:
        with pytest.raises(ValueError):
            parity.parse_path(bad)


@pytest.mark.parametrize("parts", [
    ("decision_metadata", "spans", 1, "attributes", "gen_ai.request.model"),
    ("decision_metadata", "spans", "*", "attributes", "gen_ai.request.model"),
    ("a.b", 0, "c[d]", 'say "hi"', ""),
    ("stat_answer", "facts", 0, "value"),
])
def test_format_path_round_trips(parts):
    assert parity.parse_path(parity.format_path(parts)) == parts


def test_printed_diff_path_works_as_ignore(tmp_path, capsys):
    changed = _row("Q1")
    changed["decision_metadata"]["spans"][1]["attributes"]["gen_ai.request.model"] = "HCX-DASH-002"
    base, run = _write(tmp_path, "a", [_row("Q1")]), _write(tmp_path, "b", [changed])
    assert parity.main(["--base", str(base), "--run", str(run)]) == 1
    printed = next(l for l in capsys.readouterr().out.splitlines() if l.startswith("  ")).strip().split(": base=")[0]
    assert parity.main(["--base", str(base), "--run", str(run), "--ignore", printed]) == 0


def _manifest(run, **changes):
    """manifest.json for the rows already in run/results.jsonl (question count, runs, status completed) + changes."""
    rows = [json.loads(line) for line in (run / "results.jsonl").read_text(encoding="utf-8").splitlines() if line]
    ids = {str(r["question_id"]) for r in rows}
    runs = max((int(r.get("run_index", 0)) for r in rows), default=0) + 1
    data = {"golden_sha256": "g", "golden_snapshot_sha256": "gs", "data_sha256": {"data/values/x.parquet": "d"},
            "questions": len(ids), "runs": runs, "status": "completed", "pipeline": "stat", "answer_format": "json", "serialization": "long",
            "scope_mode": "oracle", "judge_mode": "rules", "form": None, "interval_seconds": 0.0,
            "model": {"backend": "mock", "model_name": "HCX-DASH-002"}, "run_id": run.name, "started_at": run.name}
    data.update(changes)
    (run / "manifest.json").write_text(json.dumps(data), encoding="utf-8")


def test_matching_manifests_pass_and_bookkeeping_keys_are_free(tmp_path, capsys):
    rows = [_row("Q1"), _row("Q2")]
    base, run = _write(tmp_path, "a", rows), _write(tmp_path, "b", copy.deepcopy(rows))
    _manifest(base)
    _manifest(run)  # run_id and started_at differ: not conditions
    assert parity.main(["--base", str(base), "--run", str(run)]) == 0
    assert capsys.readouterr().out.strip() == "identical"


@pytest.mark.parametrize("key,value", [("golden_sha256", "other"), ("scope_mode", "catalog"),
                                       ("model", {"backend": "hcx", "model_name": "HCX-DASH-002"})])
def test_differing_run_condition_is_an_input_error(tmp_path, key, value):
    base, run = _write(tmp_path, "a", [_row("Q1")]), _write(tmp_path, "b", [_row("Q1")])
    _manifest(base)
    _manifest(run, **{key: value})
    assert parity.main(["--base", str(base), "--run", str(run)]) == 2
    assert parity.main(["--base", str(base), "--run", str(run), "--allow-condition", key]) == 0


def test_manifest_on_one_side_only_is_an_input_error(tmp_path):
    base, run = _write(tmp_path, "a", [_row("Q1")]), _write(tmp_path, "b", [_row("Q1")], manifest=False)
    assert parity.main(["--base", str(base), "--run", str(run)]) == 2
    assert parity.main(["--base", str(base), "--run", str(run), "--allow-missing-manifest"]) == 0


# --- completeness (review 11/05): two runs that lost the same question must not pass ---------------------------


def test_the_same_question_lost_on_both_sides_is_an_input_error(tmp_path):
    base, run = _write(tmp_path, "a", [_row("Q1")]), _write(tmp_path, "b", [_row("Q1")])
    _manifest(base, questions=2)
    _manifest(run, questions=2)  # both say 2 questions, both recorded 1
    assert parity.main(["--base", str(base), "--run", str(run)]) == 2


def test_a_run_that_did_not_complete_is_an_input_error(tmp_path):
    base, run = _write(tmp_path, "a", [_row("Q1")]), _write(tmp_path, "b", [_row("Q1")])
    _manifest(run, status="failed")
    assert parity.main(["--base", str(base), "--run", str(run)]) == 2


def test_listed_question_ids_must_be_the_recorded_ones(tmp_path):
    base, run = _write(tmp_path, "a", [_row("Q1")]), _write(tmp_path, "b", [_row("Q1")])
    _manifest(run, question_ids=["Q2"])
    assert parity.main(["--base", str(base), "--run", str(run)]) == 2
    _manifest(run, question_ids=["Q1"])
    assert parity.main(["--base", str(base), "--run", str(run)]) == 0


def test_a_missing_run_index_is_an_input_error(tmp_path):
    second = {**_row("Q1"), "run_index": 1}
    base, run = _write(tmp_path, "a", [_row("Q1"), second]), _write(tmp_path, "b", [_row("Q1"), second])
    _manifest(run, runs=3)
    _manifest(base, runs=3)
    assert parity.main(["--base", str(base), "--run", str(run)]) == 2


def test_completeness_rules():
    rows = {("Q1", 0): {}, ("Q2", 0): {}}
    assert parity.completeness({"status": "completed", "questions": 2, "runs": 1}, rows) == []
    assert parity.completeness({"status": "completed", "questions": 2, "runs": 1, "question_ids": ["Q1", "Q2"]}, rows) == []
    assert parity.completeness({"status": "running", "questions": 2, "runs": 1}, rows)
    assert parity.completeness({"status": "completed", "runs": 1}, rows)


def test_check_calls_reports_attempts_and_span_status(tmp_path, capsys):
    changed = _row("Q1")
    changed["attempts"] = 2
    base, run = _write(tmp_path, "a", [_row("Q1")]), _write(tmp_path, "b", [changed])
    assert parity.main(["--base", str(base), "--run", str(run)]) == 0  # attempts is volatile for the byte check
    capsys.readouterr()
    assert parity.main(["--base", str(base), "--run", str(run), "--check-calls"]) == 1
    out = capsys.readouterr().out
    assert "Q1: calls differ" in out
    assert "attempts: base=1 run=2" in out
    assert "calls 1" in out


def test_check_calls_identical_runs(tmp_path, capsys):
    base, run = _write(tmp_path, "a", [_row("Q1")]), _write(tmp_path, "b", [_row("Q1", latency=3.0, start_ms=4.0)])
    assert parity.main(["--base", str(base), "--run", str(run), "--check-calls"]) == 0
    assert capsys.readouterr().out.strip() == "identical"
