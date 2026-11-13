"""T1-29: rows a whole table used to be rejected for are settled row by row in the loader (prepare.convert.tidy_rows).

Synthetic tables only (conftest.write_table); each test is one failure type found over the 2,364 provided tables."""

from __future__ import annotations

import pytest

from kosis_chat.adapters import parquet
from kosis_chat.data import preprocessing
from kosis_chat.data.loader import load_table
from kosis_chat.prepare.schemas import SchemaError
from tests.conftest import write_table

CATALOG = {"survey_name": "가상 조사", "organization": "가상 기관", "table_name": "가상 표", "period": "년 (2020 ~ 2021)",
           "url": "https://example.invalid/t"}


def _row(period, raw, *, item="T1", code="0", prd_se="Y", name="가상 인구"):
    blank = raw == ""
    return {"prd_se": prd_se, "period": period, "item_id": item, "item_name": name, "unit": "명", "dim1_name": "성별",
            "dim1_code": code, "dim1_label": "계", "value_raw": raw, "value": None if blank else float(raw),
            "value_kind": "blank" if blank else "ok"}


def _load(tmp_path, rows):
    return preprocessing.clean(load_table(write_table(tmp_path / "data", "FAKE_T", rows, CATALOG)))


def test_rows_without_period_and_value_are_dropped(tmp_path):
    table = _load(tmp_path, [_row("2020", "10"), _row("", ""), _row("", "", item="T2")])
    assert [(c.period_raw, c.value) for c in table.cells] == [("2020", 10.0)]
    assert "시점·값이 빈 행 2 제외" in table.meta.note


def test_a_row_without_period_but_with_a_value_is_still_rejected(tmp_path):
    with pytest.raises(SchemaError, match="period_raw"):
        _load(tmp_path, [_row("2020", "10"), _row("", "7")])


def test_equal_rows_are_kept_once(tmp_path):
    table = _load(tmp_path, [_row("2020", "10"), _row("2020", "10"), _row("2021", "11")])
    assert [(c.period_raw, c.value) for c in table.cells] == [("2020", 10.0), ("2021", 11.0)]
    assert "같은 행 1 제외" in table.meta.note


def test_one_key_with_different_values_becomes_one_missing_cell(tmp_path):
    table = _load(tmp_path, [_row("2020", "10"), _row("2020", "99"), _row("2021", "11")])
    cells = {c.period_raw: c for c in table.cells}
    assert len(table.cells) == 2
    assert (cells["2020"].value, cells["2020"].value_flag, cells["2020"].value_raw) == (None, "missing", "")  # no number is picked
    assert cells["2021"].value == 11.0
    assert "겹친 셀 1개는 값 없음" in table.meta.note


def test_month_and_quarter_writing_one_period_keep_only_the_table_cycle(tmp_path):
    rows = [_row("197001", "1", prd_se="M"), _row("197002", "2", prd_se="M"), _row("197005", "5", prd_se="M"),
            _row("197001", "100", prd_se="Q"), _row("197003", "300", prd_se="Q"),  # 197003: no month row with this key shape
            _row("1970", "9", prd_se="Y")]
    table = _load(tmp_path, rows)
    # every quarter row goes (a quarter left alone would be read as a month); months and years stay as before
    assert sorted((c.period_raw, c.value) for c in table.cells) == [("1970", 9.0), ("197001", 1.0), ("197002", 2.0), ("197005", 5.0)]
    assert table.meta.period_type == "M" and "다른 주기 행 2 제외" in table.meta.note


def test_two_cycles_that_do_not_collide_are_left_as_they_were(tmp_path):
    table = _load(tmp_path, [_row("202001", "1", prd_se="M"), _row("2020", "9", prd_se="Y")])
    assert len(table.cells) == 2 and "행 정리" not in table.meta.note


def test_a_clean_table_is_unchanged(pop_path):
    table = load_table(pop_path)
    assert len(table.cells) == 10 and "행 정리" not in table.meta.note


def test_a_table_over_the_row_limit_is_refused_before_it_is_read(pop_path, monkeypatch):
    monkeypatch.setattr(parquet, "ROW_LIMIT", 5)
    monkeypatch.setattr(parquet, "read_parquet", lambda path: pytest.fail("the file was read"))
    with pytest.raises(SchemaError, match="over the 5-row limit"):
        load_table(pop_path)
