"""Work root (KOSIS_WORKSPACE) and summarize_stat(question, table_id) on the made-up tables (plans/T1-27.md)."""

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from eval.golden import GoldenItem, load_items
from kosis_chat import api
from kosis_chat.config import PROJECT_ROOT, load_env, resolve_path
from kosis_chat.data.loader import load_table
from kosis_chat.llm.model import MockClient
from kosis_chat.pipeline import StatPipeline

from kosis_ai import normalize, paths  # isort: skip

QUESTION = "2023년 서울특별시 1인 가구 수는?"


def reply(answer):
    return json.dumps({"answer": answer}, ensure_ascii=False)


@pytest.fixture
def workspace(data_dir, monkeypatch):
    """The made-up data/ (conftest) as the work root's data/."""
    monkeypatch.setenv(paths.ENV_VAR, str(data_dir.parent))
    return data_dir.parent


def test_unset_keeps_every_path_at_the_project_root(monkeypatch):
    monkeypatch.delenv(paths.ENV_VAR, raising=False)
    assert paths.workspace_root() == paths.CODE_ROOT == PROJECT_ROOT
    assert resolve_path("eval/results") == PROJECT_ROOT / "eval" / "results"
    assert GoldenItem.model_construct(table="data/values/x.parquet").path == PROJECT_ROOT / "data" / "values" / "x.parquet"


def test_set_moves_work_paths_but_not_code(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv(paths.ENV_VAR, "ws")  # relative: made absolute once, from the current folder
    root = (tmp_path / "ws").resolve()
    assert paths.workspace_root() == root and paths.data_dir() == root / "data"
    assert resolve_path("eval/results") == root / "eval" / "results"
    assert resolve_path(tmp_path / "abs") == tmp_path / "abs"
    assert GoldenItem.model_construct(table="data/values/x.parquet").path == root / "data" / "values" / "x.parquet"
    assert PROJECT_ROOT == paths.CODE_ROOT  # configs/ and the source snapshot stay with the code


def test_env_file_is_read_from_the_work_root(tmp_path, monkeypatch):
    monkeypatch.setenv(paths.ENV_VAR, str(tmp_path))
    monkeypatch.delenv("T127_FAKE_VALUE", raising=False)
    (tmp_path / ".env").write_text("T127_FAKE_VALUE=from-workspace\n", encoding="utf-8")
    load_env()
    assert os.environ["T127_FAKE_VALUE"] == "from-workspace"


def test_fetch_table_reads_the_work_root_data(workspace):
    df = normalize.fetch_table("101_FAKE_TBL_001", {"periods": ["2023"]})
    assert set(df["period"]) == {"2023"} and not df.empty


def test_summarize_stat_matches_the_pipeline_on_the_same_table(workspace, household_path):
    stat = api.summarize_stat(QUESTION, "101_FAKE_TBL_002", client=MockClient([reply("서울은 {f1}입니다.")]))
    direct = StatPipeline(load_table(household_path), client=MockClient([reply("서울은 {f1}입니다.")]), org_id="101")
    expected = direct.ask(QUESTION).stat_answer
    assert stat.model_dump() == expected.model_dump()
    assert stat.source.org_id == "101" and stat.source.tbl_id == "FAKE_TBL_002"  # as run_eval: from the file name


def test_summarize_stat_refusal_is_an_answer_not_an_error(workspace):
    stat = api.summarize_stat("2030년 서울 1인 가구 수는?", "101_FAKE_TBL_002", client=MockClient([]))
    assert stat.outcome == "refused"


def test_summarize_stat_errors(workspace, tmp_path, monkeypatch):
    with pytest.raises(LookupError, match="not in"):
        api.summarize_stat(QUESTION, "101_NO_SUCH_TABLE", client=MockClient([]))
    (workspace / "data" / "values" / "101_FAKE_TBL_001.parquet").unlink()
    with pytest.raises(LookupError, match="does not exist"):
        api.table_path("101_FAKE_TBL_001")
    down = SimpleNamespace(ok=False, status="api_error", error="down", stat_answer=None)
    monkeypatch.setattr(api, "_build", lambda *a: SimpleNamespace(org_id="101", ask=lambda q: down))
    with pytest.raises(RuntimeError, match="model call failed"):
        api.summarize_stat(QUESTION, "101_FAKE_TBL_002", client=MockClient([]))
    monkeypatch.setenv(paths.ENV_VAR, str(tmp_path / "empty"))
    with pytest.raises(LookupError, match="no tables.csv"):
        api.table_path("101_FAKE_TBL_002")


def test_a_given_client_never_enters_the_cache(workspace):
    api._cached.cache_clear()
    api.summarize_stat("2030년 서울 1인 가구 수는?", "101_FAKE_TBL_002", client=MockClient([]))
    assert api._cached.cache_info().currsize == 0


def test_pipelines_are_cached_per_table_and_settings(workspace, mock_settings, monkeypatch):
    api._cached.cache_clear()
    built = []
    real = api._build
    monkeypatch.setattr(api, "_build", lambda *a: built.append(a) or real(*a))
    for _ in range(2):
        api.summarize_stat("2030년 서울 1인 가구 수는?", "101_FAKE_TBL_002", settings=mock_settings, retry_interval=5)
    assert len(built) == 1 and built[0][1] is mock_settings and built[0][3] == 5
    api.summarize_stat("2030년 서울 1인 가구 수는?", "101_FAKE_TBL_002", settings=mock_settings)
    assert len(built) == 2  # another retry_interval is another pipeline
    assert api._cached(api.table_path("101_FAKE_TBL_002"), mock_settings, 5).retry_interval == 5
    api._cached.cache_clear()


def test_every_golden_table_is_found_by_its_key():
    """Golden table path -> key -> tables.csv `parquet` gives the same file (the route stage 2 search will use)."""
    golden = resolve_path("eval/golden/golden.jsonl")
    if not golden.exists() or not (paths.data_dir() / "tables.csv").exists():
        pytest.skip("golden set / provided data are not in the repo (gitignored)")
    _, items, _ = load_items(golden)
    for item in items:
        assert api.table_path(Path(item.table).stem).resolve() == item.path.resolve(), item.id
