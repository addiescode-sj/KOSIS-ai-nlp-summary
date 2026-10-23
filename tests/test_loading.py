"""The wide serializer and the serialization setting through Chatbot and run_eval."""

from __future__ import annotations

import json
from dataclasses import replace

import pytest

from eval import run_eval
from kosis_chat.chatbot import Chatbot
from kosis_chat.data import preprocessing, serializer
from kosis_chat.data.loader import load_table, sha256_of
from kosis_chat.llm.model import MockClient


def test_wide_markdown_puts_periods_in_columns(pop_path):
    table = preprocessing.clean(load_table(pop_path))
    text = serializer.serialize(table, "wide")
    assert "단위: 행마다 다름(단위 열)" in text  # 명 and %
    assert "| 항목 | 성별 | 단위 | 2022 | 2023 | 2024 |" in text
    assert "| 가상 인구 | 여자 | 명 |" in text and "560p" in text  # raw values kept
    assert len(text) < len(serializer.serialize(table, "long"))


def test_wide_markdown_writes_single_unit_once_and_splits_many_periods(pop_path):
    table = preprocessing.clean(load_table(pop_path))
    people = replace(table, cells=tuple(c for c in table.cells if c.unit == "명"))
    text = serializer.to_wide_markdown(people, max_period_columns=2)
    assert "단위: 명" in text and "| 단위 |" not in text
    assert "시점 2022 ~ 2023" in text and "시점 2024 ~ 2024" in text


def test_unknown_serialization_is_rejected(pop_path):
    with pytest.raises(ValueError, match="unknown serialization"):
        serializer.serialize(load_table(pop_path), "json")


def test_chatbot_uses_configured_serialization(pop_path, mock_settings):
    client = MockClient("x")
    Chatbot(load_table(pop_path), client=client, settings=replace(mock_settings, serialization="wide")).ask("q")
    assert "| 항목 | 성별 | 단위 | 2022 | 2023 | 2024 |" in client.calls[0][1]


def test_run_eval_records_serialization(pop_path, tmp_path):
    golden = tmp_path / "golden" / "v1"
    golden.mkdir(parents=True)
    item = {"id": "q1", "kind": "answerable", "question": "q", "table": str(pop_path), "table_sha256": sha256_of(pop_path)}
    (golden / "g.jsonl").write_text(json.dumps(item) + "\n", encoding="utf-8")
    out = tmp_path / "results"
    argv = ["--stage", "B0", "--golden", str(golden), "--backend", "mock", "--serialization", "wide", "--scope", "full",
            "--out", str(out), "--log-dir", str(tmp_path / "logs")]
    assert run_eval.main(argv) == 0
    manifest = json.loads((next(out.iterdir()) / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["serialization"] == "wide"
