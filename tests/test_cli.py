"""The terminal entry point (kosis_chat/cli.py, T1-28 S6): questions go to StatModule with the file read as a loaded
table. The B0 chatbot CLI is kosis_chat.legacy.cli (tests/test_chatbot.py). Made-up tables, mock model or no key."""

from __future__ import annotations

import pyarrow.parquet as pq

from kosis_chat import cli
from kosis_chat.adapters.parquet import fetched_at_of, find_catalog
from kosis_chat.api import StatModule
from kosis_chat.schemas.io import ParquetTable, StatInput, StatOptions
from kosis_chat.schemas.resources import DEFAULT_MODEL, LLMConfig

SEOUL = "2023년 서울특별시 1인 가구 수는?"  # needs the model (compose)
REFUSED = "2030년 서울 1인 가구 수는?"  # no such period: the rules refuse, no model call


def module_text(path, question: str) -> str:
    table = ParquetTable(frame=pq.read_table(path), fetched_at=fetched_at_of(path))
    module = StatModule(LLMConfig(provider="mock", model="mock"), StatOptions(catalog_path=find_catalog(path)))
    return module.answer(StatInput(question=question, tables=[table])).answer.text


def test_cli_answers_with_the_stat_module(household_path, capsys):
    assert cli.main(["--table", str(household_path), "--backend", "mock", "-q", SEOUL, "-q", REFUSED]) == 0
    out = capsys.readouterr().out
    assert out.startswith("[가상 가구원수별 가구] backend=mock model=mock\n")
    assert f"> {SEOUL}\n{module_text(household_path, SEOUL)}\n" in out
    assert f"> {REFUSED}\n{module_text(household_path, REFUSED)}\n" in out


def test_cli_reports_a_table_it_cannot_read(tmp_path, household_path, capsys):
    csv = tmp_path / "pop.csv"
    csv.write_text("item,period,value,unit\n", encoding="utf-8")
    assert cli.main(["--table", str(csv), "--backend", "mock", "-q", "x"]) == 2
    assert "unsupported input format" in capsys.readouterr().err
    assert cli.main(["--table", str(tmp_path / "none.parquet"), "--backend", "mock", "-q", "x"]) == 2
    assert "file not found" in capsys.readouterr().err
    lone = tmp_path / "values" / household_path.name  # no tables.csv next to it or one folder up
    lone.parent.mkdir()
    lone.write_bytes(household_path.read_bytes())
    assert cli.main(["--table", str(lone), "--backend", "mock", "-q", "x"]) == 2
    assert "tables.csv" in capsys.readouterr().err


def test_cli_without_a_key_answers_rule_refusals_and_stops_at_the_model(household_path, capsys, monkeypatch):
    monkeypatch.delenv(cli.KEY_ENV, raising=False)
    assert cli.main(["--table", str(household_path), "-q", REFUSED]) == 0
    assert module_text(household_path, REFUSED) in capsys.readouterr().out
    assert cli.main(["--table", str(household_path), "-q", SEOUL]) == 2
    assert cli.KEY_ENV in capsys.readouterr().err


def test_llm_config_of_the_command_line(monkeypatch):
    monkeypatch.setenv(cli.KEY_ENV, "nv-dummy")
    hcx = cli.llm_config(cli.build_parser().parse_args(["--table", "x", "--thinking", "low", "--max-tokens", "5120"]))
    assert (hcx.provider, hcx.model, hcx.thinking_effort, hcx.max_tokens) == ("hcx", DEFAULT_MODEL, "low", 5120)
    assert hcx.api_key.get_secret_value() == "nv-dummy"
    mock = cli.llm_config(cli.build_parser().parse_args(["--table", "x", "--backend", "mock", "--model", "m"]))
    assert (mock.provider, mock.model, mock.api_key) == ("mock", "m", None)
    assert LLMConfig(model=DEFAULT_MODEL).max_tokens == cli.llm_config(cli.build_parser().parse_args(["--table", "x"])).max_tokens
