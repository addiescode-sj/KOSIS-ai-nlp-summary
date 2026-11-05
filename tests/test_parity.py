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


# ---------------------------------------------------------------- --mode semantic (T1-28 S5: chart moved out of StatAnswer)

FACTS = [{"id": "f1", "value": 100, "unit": "명", "period": "2020", "source_ref": "DT_1|T1|00|2020"},
         {"id": "f2", "value": 120, "unit": "명", "period": "2021", "source_ref": "DT_1|T1|00|2021"}]
VEGA = {"$schema": "https://vega.github.io/schema/vega-lite/v5.json", "title": "인구",
        "data": {"values": [{"이름": "2020", "값": 100, "표시": "100명", "강조": "아니오"},
                            {"이름": "2021", "값": 120, "표시": "120명", "강조": "예"}]},
        "layer": [{"mark": {"type": "bar"}, "encoding": {"y": {"field": "값", "title": "값 (명)"}}}]}
SPEC = {"kind": "timeline", "title": "인구", "unit": "명", "fact_ids": ["f1", "f2"], "orientation": None, "sort": None,
        "pattern": None, "vega_lite": VEGA,
        "series": [{"name": "인구", "points": [
            {"label": "2020", "value": 100.0, "display": "100명", "fact_id": "f1", "period": "2020", "highlight": False,
             "share_pct": None},
            {"label": "2021", "value": 120.0, "display": "120명", "fact_id": "f2", "period": "2021", "highlight": True,
             "share_pct": None}]}]}


def _stat_row(question_id: str, layout: str) -> dict:
    """A row as run_eval wrote it before S5 ("old": stat_answer.chart + Vega-Lite row chart) or after ("new")."""
    stat = {"outcome": "answered", "text": "2021년 인구는 120명입니다.", "result": {"facts": copy.deepcopy(FACTS)},
            "parts": [{"kind": "fact", "fact_id": "f2"}], "numbers_origin": "code", "numbers_verified": True}
    row = {"question_id": question_id, "run_index": 0, "call_status": "ok", "judge": {}, "attempts": 1,
           "decision_metadata": {"route": "rules", "chart": "timeline", "spans": []}, "stat_answer": stat}
    if layout == "old":
        stat["chart"] = {"kind": "timeline", "spec": copy.deepcopy(VEGA), "fact_ids": ["f1", "f2"]}
        row["chart"] = copy.deepcopy(VEGA)
    else:
        row["chart"] = copy.deepcopy(SPEC)
    return row


def test_semantic_mode_reads_the_old_and_the_new_layout_alike(tmp_path, capsys):
    base, run = _write(tmp_path, "a", [_stat_row("Q1", "old")]), _write(tmp_path, "b", [_stat_row("Q1", "new")])
    assert parity.main(["--base", str(base), "--run", str(run)]) == 1  # bytes: the layouts differ
    capsys.readouterr()
    assert parity.main(["--base", str(base), "--run", str(run), "--mode", "semantic"]) == 0
    out = capsys.readouterr().out
    assert out.startswith("identical") and "vega_lite: 1/1 charts identical" in out
    form = parity.semantic_row(_stat_row("Q1", "old"))["chart"]
    assert form["unit"] == "명" and [p["fact_id"] for p in form["points"]] == ["f1", "f2"]
    assert form == parity.semantic_row(_stat_row("Q1", "new"))["chart"]


@pytest.mark.parametrize("change, path", [
    (lambda c: c["series"][0]["points"][1].update(value=121.0), "chart.points[1].value"),
    (lambda c: c.update(unit="천명"), "chart.unit"),
    (lambda c: c.update(kind="bar"), "chart.kind"),
    (lambda c: c.update(fact_ids=["f2", "f1"]), "chart.fact_ids"),
])
def test_semantic_mode_reports_a_changed_point_or_chart(tmp_path, capsys, change, path):
    new = _stat_row("Q1", "new")
    change(new["chart"])
    base, run = _write(tmp_path, "a", [_stat_row("Q1", "old")]), _write(tmp_path, "b", [new])
    assert parity.main(["--base", str(base), "--run", str(run), "--mode", "semantic"]) == 1
    assert path in capsys.readouterr().out


def test_semantic_mode_checks_the_fact_a_point_names(tmp_path, capsys):
    new = _stat_row("Q1", "new")
    new["chart"]["series"][0]["points"][0]["fact_id"] = "f2"  # f2's value is 120, the point shows 100
    base, run = _write(tmp_path, "a", [_stat_row("Q1", "old")]), _write(tmp_path, "b", [new])
    assert parity.main(["--base", str(base), "--run", str(run), "--mode", "semantic"]) == 1
    assert "chart links" in capsys.readouterr().out


def test_semantic_mode_compares_the_answer_and_takes_ignore_paths(tmp_path, capsys):
    new = _stat_row("Q1", "new")
    new["stat_answer"]["text"] = "인구는 120명이었습니다."
    new["stat_answer"]["result"]["facts"][1]["value"] = 120.0  # 120 and 120.0 are the same number here
    base, run = _write(tmp_path, "a", [_stat_row("Q1", "old")]), _write(tmp_path, "b", [new])
    assert parity.main(["--base", str(base), "--run", str(run), "--mode", "semantic"]) == 1
    assert "text" in capsys.readouterr().out
    assert parity.main(["--base", str(base), "--run", str(run), "--mode", "semantic", "--ignore", "text"]) == 0


def _semantic(tmp_path, new: dict, old: dict | None = None) -> int:
    base, run = _write(tmp_path, "a", [old or _stat_row("Q1", "old")]), _write(tmp_path, "b", [new])
    return parity.main(["--base", str(base), "--run", str(run), "--mode", "semantic"])


@pytest.mark.parametrize("change, message", [
    (lambda r: r["chart"]["series"][0]["points"][0].update(fact_id=None), "but 'f1' is the charted fact"),  # link lost
    (lambda r: r["chart"]["series"][0]["points"][0].update(period="1999"), "its fact 'f1' at '2020'"),
    (lambda r: r["chart"].update(pattern="zz"), "chart pattern 'zz'"),
    (lambda r: r["chart"].update(orientation="vertical"), "bar layout"),  # a timeline has no bar layout
])
def test_semantic_mode_gates_the_new_rows_links(tmp_path, capsys, change, message):
    new = _stat_row("Q1", "new")
    new["decision_metadata"]["pattern"] = None
    change(new)
    assert _semantic(tmp_path, new) == 1
    assert message in capsys.readouterr().out


def test_semantic_mode_gates_the_vega_lite(tmp_path, capsys):
    new = _stat_row("Q1", "new")
    new["chart"]["vega_lite"]["title"] = "다른 제목"
    assert _semantic(tmp_path, new) == 1
    assert "vega_lite: 0/1 charts identical (differ: Q1)" in capsys.readouterr().out


def test_semantic_mode_does_not_ask_a_tie_to_name_a_fact(tmp_path, capsys):
    """Two points show 120 (the fact f2 and a context row): neither must name f2 (S-EXT-06: a tied candidate)."""
    old, new = _stat_row("Q1", "old"), _stat_row("Q1", "new")
    extra = {"이름": "2022", "값": 120, "표시": "120명", "강조": "아니오"}
    for spec in (old["chart"], old["stat_answer"]["chart"]["spec"], new["chart"]["vega_lite"]):
        spec["data"]["values"].append(dict(extra))
    new["chart"]["series"][0]["points"].append({"label": "2022", "value": 120.0, "display": "120명", "fact_id": None,
                                                "period": "2022", "highlight": False, "share_pct": None})
    new["chart"]["series"][0]["points"][1]["fact_id"] = "f2"
    assert _semantic(tmp_path, new, old) == 0, capsys.readouterr().out


def test_semantic_mode_compares_bar_layout_and_line_periods():
    old, new = _stat_row("Q1", "old"), _stat_row("Q1", "new")
    for row in (old, new):
        row["decision_metadata"].pop("pattern", None)
    bars = {"layer": [{"mark": {"type": "bar"}, "encoding": {"y": {"field": "이름", "sort": "-x"}, "x": {"title": "값 (명)"}}}]}
    old["stat_answer"]["chart"].update(kind="bar", spec={**old["stat_answer"]["chart"]["spec"], **bars})
    new["chart"].update(kind="bar", orientation="vertical", sort="-x")
    assert parity.semantic_row(old)["chart"]["orientation"] == "horizontal"
    assert parity.semantic_row(new)["chart"]["orientation"] == "vertical"  # a difference the gate reports
    line_old, line_new = _stat_row("Q2", "old"), _stat_row("Q2", "new")
    for row in line_old["stat_answer"]["chart"]["spec"]["data"]["values"]:
        row["시점"] = row["이름"]
    line_old["stat_answer"]["chart"]["kind"] = line_new["chart"]["kind"] = "line"
    assert [p["period"] for p in parity.semantic_row(line_old)["chart"]["points"]] == ["2020", "2021"]
    assert parity.semantic_row(line_old)["chart"]["points"] == parity.semantic_row(line_new)["chart"]["points"]
