"""StatModule (T1-28 S5): StatModule(llm, options).answer(StatInput) -> StatOutput, summarize_stat_run, the prepared
table cache by content hash, a replaced Parquet / tables.csv on the public file route, and the tables.csv tbl_id check.

Made-up tables only (tests/conftest.py); the mock model (no key).
"""

from __future__ import annotations

import json
import os

import pyarrow.parquet as pq
import pytest
from pydantic import ValidationError

from kosis_chat import api
from kosis_chat.adapters import parquet as adapters
from kosis_chat.adapters.parquet import load_table, read_catalog
from kosis_chat.chart.vega import to_vega
from kosis_chat.llm.model import MockClient
from kosis_chat.prepare.schemas import SchemaError
from kosis_chat.schemas.io import ParquetTable, StatInput, StatOptions, StatOutput
from kosis_chat.schemas.resources import LLMConfig, LLMKeyMissing

from kosis_ai import paths  # isort: skip
from kosis_ai.compat import legacy_chart, load_stat_answer, without_legacy_chart  # isort: skip
from kosis_ai.schemas import ChartSpec, StatAnswer  # isort: skip

KEY = "101_FAKE_TBL_002"
SEOUL = "2023년 서울특별시 1인 가구 수는?"
AVERAGE = "2015년부터 2024년까지 서울특별시 1인 가구의 연평균은?"
REFUSED = "2030년 서울 1인 가구 수는?"  # no such period: the rules refuse, no model call


def reply(answer: str) -> str:
    return json.dumps({"answer": answer}, ensure_ascii=False)


@pytest.fixture
def workspace(data_dir, monkeypatch):
    monkeypatch.setenv(paths.ENV_VAR, str(data_dir.parent))
    api._cached.cache_clear()
    yield data_dir.parent
    api._cached.cache_clear()


@pytest.fixture
def frame(household_path):
    return pq.read_table(household_path)


def module(mock_llm, replies=None, **options) -> api.StatModule:
    client = MockClient([reply("서울은 {f1}입니다.")] * 4 if replies is None else replies)
    return api.StatModule(mock_llm, StatOptions(**options), client=client)


# ---------------------------------------------------------------- the entry point


def test_answer_is_summarize_stat_on_the_same_table(workspace, frame, mock_llm):
    out = module(mock_llm).answer(StatInput(question=SEOUL, tables=[ParquetTable(frame=frame)]))
    stat = api.summarize_stat(SEOUL, KEY, client=MockClient([reply("서울은 {f1}입니다.")]))
    assert isinstance(out, StatOutput) and out.schema_version == "1"
    assert out.answer == stat and out.answer.outcome == "answered"
    run = api.summarize_stat_run(SEOUL, KEY, client=MockClient([reply("서울은 {f1}입니다.")]))
    assert run.answer == stat and run.chart == out.chart and [s["name"] for s in run.spans] == [s["name"] for s in out.spans]


def test_output_carries_the_chart_and_the_spans(workspace, frame, mock_llm):
    out = module(mock_llm).answer(StatInput(question=AVERAGE, tables=[ParquetTable(frame=frame)]))
    assert isinstance(out.chart, ChartSpec) and out.chart.kind == "timeline" and out.chart.vega_lite is None
    assert "chart" not in out.answer.model_dump() and "chart" not in json.loads(out.model_dump_json())["answer"]
    assert set(out.chart.fact_ids) <= {f.id for f in out.answer.result.facts}
    assert to_vega(out.chart)["data"]["values"]  # Vega-Lite on demand
    names = [s["name"] for s in out.spans]
    assert names[0] == "judge" and "compose" in names and all({"name", "status", "start_ms"} <= set(s) for s in out.spans)


def test_a_refused_answer_has_no_chart_and_needs_no_key(workspace, frame):
    no_key = api.StatModule(LLMConfig(model="HCX-DASH-002"))  # provider hcx, no key
    out = no_key.answer(StatInput(question=REFUSED, tables=[ParquetTable(frame=frame)]))
    assert out.answer.outcome == "refused" and out.chart is None
    with pytest.raises(LLMKeyMissing):  # a question that needs the model
        no_key.answer(StatInput(question=SEOUL, tables=[ParquetTable(frame=frame)]))


def test_input_holds_one_table(frame):
    with pytest.raises(ValidationError):
        StatInput(question=SEOUL, tables=[ParquetTable(frame=frame), ParquetTable(frame=frame)])
    with pytest.raises(ValidationError):
        StatInput(question="", tables=[ParquetTable(frame=frame)])
    assert StatInput(question=SEOUL, tables=[ParquetTable(frame=frame)]).schema_version == "1"


def test_output_keeps_the_old_chart_rule(workspace, frame, mock_llm):
    out = module(mock_llm).answer(StatInput(question=AVERAGE, tables=[ParquetTable(frame=frame)]))
    refused = module(mock_llm).answer(StatInput(question=REFUSED, tables=[ParquetTable(frame=frame)])).answer
    with pytest.raises(ValidationError, match="answered"):
        StatOutput(answer=refused, chart=out.chart)
    other = out.chart.model_copy(update={"fact_ids": ["f99"], "series": []})
    with pytest.raises(ValidationError, match="unknown facts"):
        StatOutput(answer=out.answer, chart=ChartSpec.model_construct(**other.__dict__))


def test_options_that_the_pipeline_cannot_apply_yet_are_refused(mock_llm):
    with pytest.raises(ValueError, match="not supported"):
        api.StatModule(mock_llm, StatOptions(compose_max_attempts=3))
    with pytest.raises(ValueError, match="not supported"):
        api.StatModule(mock_llm, StatOptions(use_fallback_chatbot=False))
    assert "node_overrides" not in StatOptions().model_dump()


def test_node_overrides_reach_the_graph(workspace, frame, mock_llm):
    from kosis_chat.judge.node import step_judge
    seen = []

    def judge(inp, res):
        seen.append(inp.question)
        return step_judge(inp, res)

    out = module(mock_llm, node_overrides={"judge": judge}).answer(StatInput(question=REFUSED, tables=[ParquetTable(frame=frame)]))
    assert seen == [REFUSED] and out.answer.outcome == "refused"


# ---------------------------------------------------------------- the prepared table cache (content_hash)


def test_the_same_rows_are_prepared_once(workspace, frame, mock_llm):
    stat = module(mock_llm)
    first = stat.pipeline(ParquetTable(frame=frame))
    assert stat.pipeline(ParquetTable(frame=pq.read_table(workspace / "data" / "values" / f"{KEY}.parquet"))) is first
    assert len(stat._pipelines) == 1


def test_another_catalog_row_for_the_same_frame_is_prepared_again(workspace, frame, mock_llm, tmp_path):
    stat = module(mock_llm)
    first = stat.pipeline(ParquetTable(frame=frame))
    catalog = workspace / "data" / "tables.csv"
    text = catalog.read_text(encoding="utf-8-sig")
    catalog.write_text(text.replace("가상 가구원수별 가구", "가상 가구원수별 가구 (개정)"), encoding="utf-8-sig")
    os.utime(catalog, ns=(catalog.stat().st_atime_ns, catalog.stat().st_mtime_ns + 10**9))  # a later mtime, surely
    second = stat.pipeline(ParquetTable(frame=frame))
    assert second is not first and second.table.meta.tbl_nm == "가상 가구원수별 가구 (개정)"
    assert first.table.meta.tbl_nm == "가상 가구원수별 가구"


def test_a_dataframe_changed_in_place_is_prepared_again(workspace, frame, mock_llm):
    stat = module(mock_llm)
    df = frame.to_pandas()
    table = ParquetTable(frame=df)
    first = stat.pipeline(table)
    row = df.index[(df["item_id"] == "T1") & (df["dim1_code"] == "11") & (df["period"] == "2023")][0]
    df.loc[row, "value_raw"], df.loc[row, "value"] = "9,999", 9999.0  # the wrapper is frozen, the DataFrame is not
    second = stat.pipeline(table)
    assert second is not first
    out = stat.answer(StatInput(question=SEOUL, tables=[table]))
    assert out.answer.result.facts[0].value == 9999


def test_info_names_the_table_without_a_catalog(frame, mock_llm, tmp_path, monkeypatch):
    from kosis_ai.schemas import TableInfo
    monkeypatch.setenv(paths.ENV_VAR, str(tmp_path / "nowhere"))  # no tables.csv: info is enough
    info = TableInfo(key=KEY, org_id="101", tbl_id="FAKE_TBL_002", survey_name="가상 총조사", organization="가상 기관",
                     table_name="가상 가구원수별 가구", period="년 (2015 ~ 2025)", source="api", coverage="A_정상",
                     rows=1, numeric_rows=1, symbol_rows=0, parquet=f"values/{KEY}.parquet",
                     url="https://example.invalid/fake-table-002", menu_path="가상")
    out = module(mock_llm).answer(StatInput(question=REFUSED, tables=[ParquetTable(frame=frame, info=info)]))
    assert out.answer.source.org_id == "101" and out.answer.source.tbl_nm == "가상 가구원수별 가구"


# ---------------------------------------------------------------- the file route: a replaced file is read again


def _rewrite(path, value: float) -> None:
    """The same file path, Seoul's 2023 one-person households changed (tables.csv untouched)."""
    import pyarrow as pa
    before = path.stat().st_mtime_ns
    df = pq.read_table(path).to_pandas()
    row = df.index[(df["item_id"] == "T1") & (df["dim1_code"] == "11") & (df["period"] == "2023")][0]
    df.loc[row, "value_raw"], df.loc[row, "value"] = f"{value:,.0f}", value
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False), path)
    os.utime(path, ns=(path.stat().st_atime_ns, max(path.stat().st_mtime_ns, before + 10**9)))


def test_a_replaced_parquet_at_the_same_path_is_read_again(workspace, mock_settings):
    first = api.summarize_stat(REFUSED, KEY, settings=mock_settings)  # the cached route (no given client)
    assert api._cached.cache_info().currsize == 1
    pipeline = api._cached(api.table_path(KEY), mock_settings, 0.0)
    assert pipeline.table.cells and first.outcome == "refused"
    catalog = (workspace / "data" / "tables.csv").read_bytes()
    _rewrite(workspace / "data" / "values" / f"{KEY}.parquet", 7777.0)
    assert (workspace / "data" / "tables.csv").read_bytes() == catalog
    api.summarize_stat(REFUSED, KEY, settings=mock_settings)
    rebuilt = api._cached(api.table_path(KEY), mock_settings, 0.0)
    assert rebuilt is not pipeline
    assert any(c.value == 7777 for c in rebuilt.table.cells)


def test_a_replaced_tables_csv_is_read_again(workspace, mock_settings):
    api.summarize_stat(REFUSED, KEY, settings=mock_settings)
    catalog = workspace / "data" / "tables.csv"
    catalog.write_text(catalog.read_text(encoding="utf-8-sig").replace("가상 가구원수별 가구", "가상 가구 (새 이름)"),
                       encoding="utf-8-sig")
    os.utime(catalog, ns=(catalog.stat().st_atime_ns, catalog.stat().st_mtime_ns + 10**9))
    stat = api.summarize_stat(REFUSED, KEY, settings=mock_settings)
    assert stat.source.tbl_nm == "가상 가구 (새 이름)"


def test_cache_clear_still_resets_everything(workspace, mock_settings):
    api.summarize_stat(REFUSED, KEY, settings=mock_settings)
    assert api._cached.cache_info().currsize == 1 and api._STAMPS
    api._cached.cache_clear()  # Phase 2 tests/test_stat_baseline.py
    assert api._cached.cache_info().currsize == 0 and not api._STAMPS


# ---------------------------------------------------------------- tables.csv tbl_id against the rows' (gated)


def _misnamed(workspace) -> None:
    """A catalog row whose tbl_id is not the file rows' tbl_id."""
    catalog = workspace / "data" / "tables.csv"
    text = catalog.read_text(encoding="utf-8-sig")
    catalog.write_text(text.replace(f"{KEY},101,FAKE_TBL_002,", f"{KEY},101,FAKE_TBL_009,"), encoding="utf-8-sig")


def test_the_tbl_id_check_is_off_until_the_scan(workspace, monkeypatch):
    _misnamed(workspace)
    path = workspace / "data" / "values" / f"{KEY}.parquet"
    assert adapters.CHECK_CATALOG_TBL_ID is False  # scripts/scan_tbl_ids.py first (plans/T1-28.md S5)
    assert read_catalog(workspace / "data" / "tables.csv")[KEY]["tbl_id"] == "FAKE_TBL_009"
    assert load_table(path).meta.tbl_id == "FAKE_TBL_009"  # today's file route: the catalog row wins, unchecked
    monkeypatch.setattr(adapters, "CHECK_CATALOG_TBL_ID", True)
    with pytest.raises(SchemaError, match="tables.csv tbl_id"):
        load_table(path)


def test_the_tbl_id_check_passes_a_matching_table(workspace, monkeypatch):
    monkeypatch.setattr(adapters, "CHECK_CATALOG_TBL_ID", True)
    assert load_table(workspace / "data" / "values" / f"{KEY}.parquet").meta.tbl_id == "FAKE_TBL_002"


# ---------------------------------------------------------------- old records (kosis_ai/compat.py)


def test_old_stat_answer_records_load_through_one_boundary(workspace):
    stat = api.summarize_stat(AVERAGE, KEY, client=MockClient([reply("평균 {f5}입니다.")]))
    old_chart = {"kind": "timeline", "spec": {"data": {"values": []}}, "fact_ids": ["f1"]}
    old = stat.model_dump(mode="json") | {"chart": old_chart}
    assert load_stat_answer(old) == stat and legacy_chart(old) == old_chart
    assert without_legacy_chart(old) == stat.model_dump(mode="json")
    assert StatAnswer.model_validate(old) == stat  # Phase 2 evals/runners/stat.py calls model_validate itself
    with pytest.raises(ValidationError):  # only the old chart shape is dropped
        load_stat_answer(stat.model_dump(mode="json") | {"chart": {"kind": "bar", "series": []}})
    with pytest.raises(ValidationError):
        load_stat_answer(stat.model_dump(mode="json") | {"charts": None})


@pytest.mark.parametrize("chart", [
    {},
    {"kind": "bar"},
    {"kind": "bar", "spec": "oops", "fact_ids": ["f1"]},
    {"kind": "bar", "spec": {"mark": "bar"}, "fact_ids": ["f1"]},  # no inline data
    {"kind": "bar", "spec": {"data": {"values": []}}, "fact_ids": []},
    {"kind": "bar", "spec": {"data": {"values": []}}, "fact_ids": ["f1", "f1"]},
    {"kind": "bar", "spec": {"data": {"values": []}}, "fact_ids": ["f99"]},  # a fact the answer does not have
    {"kind": "bar", "spec": {"data": {"values": []}}, "fact_ids": ["f1"], "extra": 1},
    "x",
])
def test_an_old_chart_the_old_contract_refused_still_fails(workspace, chart):
    stat = api.summarize_stat(AVERAGE, KEY, client=MockClient([reply("평균 {f5}입니다.")]))
    with pytest.raises(ValidationError):
        load_stat_answer(stat.model_dump(mode="json") | {"chart": chart})


def test_an_old_chart_on_a_refused_answer_still_fails(workspace):
    refused = api.summarize_stat(REFUSED, KEY, client=MockClient([]))
    old_chart = {"kind": "bar", "spec": {"data": {"values": []}}, "fact_ids": ["f1"]}
    assert load_stat_answer(refused.model_dump(mode="json") | {"chart": None}) == refused  # every old refusal had chart: null
    with pytest.raises(ValidationError):
        load_stat_answer(refused.model_dump(mode="json") | {"chart": old_chart})


def test_the_parquet_stamp_reads_the_footer(workspace):
    path = workspace / "data" / "values" / f"{KEY}.parquet"
    stamp = adapters.parquet_stamp(path)
    assert stamp[:2] == adapters.file_stamp(path) and stamp[2] == pq.read_metadata(path).num_rows
    assert adapters.parquet_stamp(workspace / "nope.parquet") is None


def test_one_table_is_built_once_by_two_threads_and_others_do_not_wait(workspace, frame, mock_llm, monkeypatch):
    import threading
    import time
    stat = module(mock_llm)
    built, real = [], stat._build_pipeline

    def slow(table, catalog):
        built.append(table.tbl_id())
        time.sleep(0.3)
        return real(table, catalog)

    monkeypatch.setattr(stat, "_build_pipeline", slow)
    got = []
    threads = [threading.Thread(target=lambda: got.append(stat.pipeline(ParquetTable(frame=frame)))) for _ in range(2)]
    for t in threads:
        t.start()
    time.sleep(0.05)
    with stat._lock:  # the module lock is free while a table is being built
        pass
    for t in threads:
        t.join()
    assert built == ["FAKE_TBL_002"] and got[0] is got[1] and not stat._building
