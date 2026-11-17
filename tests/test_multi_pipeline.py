"""T1-30: questions over two or three tables through the multi-table graph (MultiStatPipeline, graph/multi.py).

The model is a MockClient: it writes the plan the test gives (analyze-plan-multi-v1) and nothing usable for the
sentence, so the answer's sentence is the code template.
"""

import json

import pytest

from conftest import write_table
from eval.grade import evaluate
from kosis_chat.analyze import multi
from kosis_chat.data.loader import load_table
from kosis_chat.llm.model import MockClient
from kosis_chat.pipeline import MultiStatPipeline
from kosis_chat.schemas.resources import MultiResources, load_resources

from test_multi_compute import CATALOG, rows  # isort: skip

URLS = {"FAKE_BIRTH": "https://kosis.kr/statHtml/statHtml.do?orgId=101&tblId=FAKE_BIRTH",
        "FAKE_DEATH": "https://kosis.kr/statHtml/statHtml.do?orgId=101&tblId=FAKE_DEATH",
        "FAKE_POPK": "https://kosis.kr/statHtml/statHtml.do?orgId=101&tblId=FAKE_POPK"}
NAMES = {"FAKE_BIRTH": "가상 출생", "FAKE_DEATH": "가상 사망", "FAKE_POPK": "가상 연앙인구"}


@pytest.fixture
def tables(tmp_path):
    folder = tmp_path / "data"
    specs = {"FAKE_BIRTH": ("T1", "출생아수", "명", 1000), "FAKE_DEATH": ("T1", "사망자수", "명", 900),
             "FAKE_POPK": ("T1", "연앙인구", "천명", 50)}
    out = {}
    for tbl_id, spec in specs.items():
        path = write_table(folder, tbl_id, rows(*spec), CATALOG | {"table_name": NAMES[tbl_id], "url": URLS[tbl_id]})
        out[tbl_id] = load_table(path)
    return out


def sel(step_id, table, region="11", period="2024"):
    return {"id": step_id, "op": "select", "table": table, "item": {"codes": ["T1"]},
            "dims": {"행정구역별": {"codes": [region]}}, "periods": {"kind": "list", "periods": [period]}}


def scripted(*plans):
    """A model that writes the given plans in turn and an unusable reply for anything else (the sentence)."""
    queue = [json.dumps(p, ensure_ascii=False) for p in plans]
    seen = []

    def reply(system, user):
        if system == multi.SYSTEM_PROMPT:
            seen.append(user)
            return queue.pop(0)
        return "no json"
    client = MockClient(reply)
    client.seen = seen
    return client


NATURAL = {"steps": [sel("b", "A"), sel("d", "B"), {"id": "n", "op": "combine", "a": "b", "b": "d", "fn": "diff",
                                                     "note": "자연증가"}], "show": ["b", "d", "n"], "present": "compare"}


def pipeline(tables, names, client):
    return MultiStatPipeline([tables[n] for n in names], client=client, org_ids=["101"] * len(names))


def test_births_minus_deaths_is_computed_and_names_both_tables(tables):
    client = scripted(NATURAL)
    answer = pipeline(tables, ["FAKE_BIRTH", "FAKE_DEATH"], client).ask("2024년 서울 출생아에서 사망자를 빼면?")
    stat = answer.stat_answer
    assert stat.outcome == "answered" and stat.numbers_origin == "code"
    facts = {f.id: f for f in stat.result.facts}
    assert facts["f1"].source_ref.startswith("FAKE_BIRTH|") and facts["f2"].source_ref.startswith("FAKE_DEATH|")
    assert facts["f3"].value == 200
    assert [s.tbl_id for s in stat.sources] == ["FAKE_BIRTH", "FAKE_DEATH"] and stat.source == stat.sources[0]
    assert stat.text.count("출처:") == 1 and "가상 출생" in stat.text and "가상 사망" in stat.text
    assert answer.decision_metadata["route"] == "llm_plan_multi" and answer.decision_metadata["pipeline"] == "stat_multi"
    # the prompt shows both tables under their labels
    assert "[표 A] 가상 출생" in client.seen[0] and "[표 B] 가상 사망" in client.seen[0]


def test_per_thousand_people_with_a_conversion_fact(tables):
    plan = {"steps": [sel("b", "A"), sel("p", "B"),
                      {"id": "r", "op": "combine", "a": "b", "b": "p", "fn": "per", "scale": 1000}],
            "show": ["b", "p", "r"], "present": "value"}
    stat = pipeline(tables, ["FAKE_BIRTH", "FAKE_POPK"], scripted(plan)).ask("2024년 서울 인구 천명당 출생아는?").stat_answer
    assert stat.outcome == "answered"
    by_id = {f.id: f.value for f in stat.result.facts}
    for f in stat.result.facts:
        if f.derivation:
            assert abs(round(evaluate(f.derivation, by_id), f.round) - f.value) < 1e-9
    assert any(f.derivation == "f2 * 1000" for f in stat.result.facts)


def test_a_plan_without_a_table_is_said_back_once(tables):
    no_table = {"steps": [sel("b", None), sel("d", "B"), {"id": "n", "op": "combine", "a": "b", "b": "d", "fn": "diff"}],
                "show": ["n"], "present": "compare"}
    client = scripted(no_table, NATURAL)
    answer = pipeline(tables, ["FAKE_BIRTH", "FAKE_DEATH"], client).ask("2024년 서울 자연증가는?")
    assert answer.stat_answer.outcome == "answered"
    assert "table에 표 기호" in answer.decision_metadata["plan_problems"][0]
    assert "이전 계획의 문제" in client.seen[1]


def test_every_table_must_be_read(tables):
    one = {"steps": [sel("b", "A")], "show": ["b"], "present": "value"}
    stat = pipeline(tables, ["FAKE_BIRTH", "FAKE_DEATH"], scripted(one, one)).ask("2024년 서울 자연증가는?").stat_answer
    assert stat.outcome == "refused" and stat.refusal.reason == "class_mismatch"
    assert len(stat.sources) == 2


def test_a_calculation_refused_twice_is_the_answer(tables):
    missing = {"steps": [sel("b", "A", period="2024"), sel("d", "B", period="2019"),
                         {"id": "n", "op": "combine", "a": "b", "b": "d", "fn": "diff"}], "show": ["n"], "present": "compare"}
    stat = pipeline(tables, ["FAKE_BIRTH", "FAKE_DEATH"], scripted(missing, missing)).ask("서울 출생아와 사망자 차이는?").stat_answer
    assert stat.outcome == "refused" and stat.query is not None


def test_a_causal_question_is_refused_without_the_model(tables):
    client = scripted()
    stat = pipeline(tables, ["FAKE_BIRTH", "FAKE_DEATH"], client).ask("서울 출생아가 사망자보다 적은 이유는 뭐야?").stat_answer
    assert stat.outcome == "refused" and stat.refusal.reason == "causal" and client.seen == []


def test_a_period_no_table_has_is_refused(tables):
    stat = pipeline(tables, ["FAKE_BIRTH", "FAKE_DEATH"], scripted()).ask("1990년 서울 자연증가는?").stat_answer
    assert stat.outcome == "refused" and stat.refusal.reason == "no_period"
    assert "가상 출생" in stat.refusal.available and "가상 사망" in stat.refusal.available


def test_a_failed_model_call_raises(tables):
    from kosis_chat.llm.model import LLMResult

    client = MockClient(lambda s, u: LLMResult(text="", status="http_error", error="boom"))
    with pytest.raises(RuntimeError):
        pipeline(tables, ["FAKE_BIRTH", "FAKE_DEATH"], client).ask("2024년 서울 자연증가는?")


def test_multi_resources_hold_two_or_three_distinct_tables(tables):
    birth = load_resources(tables["FAKE_BIRTH"])
    with pytest.raises(ValueError):
        MultiResources((birth,))
    with pytest.raises(ValueError):
        MultiResources((birth, birth))
    res = MultiResources((birth, load_resources(tables["FAKE_DEATH"])))
    assert res.labels == {"A": "FAKE_BIRTH", "B": "FAKE_DEATH"} and res.table is birth.table


# ---------------------------------------------------------------- StatModule / summarize_stats_run (T1-30 S4)


def _parquet(tables_dir, tbl_id):
    import pyarrow.parquet as pq

    from kosis_chat.schemas.io import ParquetTable

    return ParquetTable(frame=pq.read_table(tables_dir / "values" / f"101_{tbl_id}.parquet"))


def test_stat_module_answers_two_tables_and_reuses_their_preparation(tables, tmp_path, mock_llm):
    from kosis_chat import api
    from kosis_chat.schemas.io import StatInput

    folder = tmp_path / "data"
    birth, death = _parquet(folder, "FAKE_BIRTH"), _parquet(folder, "FAKE_DEATH")
    from kosis_chat.schemas.io import StatOptions

    module = api.StatModule(mock_llm, StatOptions(catalog_path=folder / "tables.csv"), client=scripted(NATURAL, NATURAL))
    out = module.answer(StatInput(question="2024년 서울 출생아에서 사망자를 빼면?", tables=[birth, death]))
    assert out.answer.outcome == "answered" and [s.tbl_id for s in out.answer.sources] == ["FAKE_BIRTH", "FAKE_DEATH"]
    assert out.chart is not None and out.spans
    one = module.pipeline(birth)
    again = module.multi_pipeline([birth, death])
    assert again.resources.tables[0] is one.resources  # the table was prepared once
    assert module.multi_pipeline([birth, death]) is again
    # one table: the one-table path, no sources
    single = module.answer(StatInput(question="2024년 서울 출생아 수는?", tables=[birth]))
    assert single.answer.sources == []


def test_stat_input_takes_up_to_three_distinct_tables(tables, tmp_path):
    from pydantic import ValidationError

    from kosis_chat.schemas.io import StatInput

    folder = tmp_path / "data"
    birth, death, pop = (_parquet(folder, t) for t in ("FAKE_BIRTH", "FAKE_DEATH", "FAKE_POPK"))
    assert len(StatInput(question="q", tables=[birth, death, pop]).tables) == 3
    with pytest.raises(ValidationError, match="given twice"):
        StatInput(question="q", tables=[birth, birth])
    with pytest.raises(ValidationError):
        StatInput(question="q", tables=[birth, death, pop, birth])


def test_summarize_stats_run_reads_the_keys(tables, tmp_path, monkeypatch):
    from kosis_chat import api

    from kosis_ai import paths  # isort: skip

    monkeypatch.setenv(paths.ENV_VAR, str(tmp_path))
    out = api.summarize_stats_run("2024년 서울 출생아에서 사망자를 빼면?", ["101_FAKE_BIRTH", "101_FAKE_DEATH"],
                                  client=scripted(NATURAL))
    assert out.answer.outcome == "answered" and out.answer.sources[1].tbl_id == "FAKE_DEATH"
