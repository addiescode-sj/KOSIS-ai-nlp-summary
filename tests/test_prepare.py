"""prepare (T1-28 S2): a loaded table (ParquetTable) -> TableInput without opening a file, and its content hash.

Made-up tables only (tests/conftest.py). The file route (adapters.parquet.load_table) is the reference: the same
rows passed as a frame give the same cells and meta, except file_sha256 (the content hash) and fetched_at.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import replace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from pydantic import ValidationError

from kosis_chat.adapters.parquet import catalog_row, load_table
from kosis_chat.prepare.clean import clean
from kosis_chat.prepare.node import PARTIAL_NOTE, content_hash, prepare, table_input
from kosis_chat.prepare.schemas import SchemaError
from kosis_chat.schemas.io import ParquetTable, StatInput
from tests.conftest import pop_rows, write_table

from kosis_ai.schemas import TableInfo  # isort: skip

POP_CATALOG = {"survey_name": "가상 인구조사", "organization": "가상 기관", "table_name": "가상 성별 인구",
               "period": "년 (2022 ~ 2024)", "url": "https://example.invalid/fake-table-001"}


def _info(**changes) -> TableInfo:
    fields = {"key": "101_FAKE_TBL_001", "org_id": "101", "tbl_id": "FAKE_TBL_001", "survey_name": "가상 인구조사",
              "organization": "가상 기관", "table_name": "가상 성별 인구", "period": "년 (2022 ~ 2024)", "source": "api",
              "coverage": "A_정상", "rows": 10, "numeric_rows": 8, "symbol_rows": 2,
              "parquet": "values/101_FAKE_TBL_001.parquet", "url": "https://example.invalid/fake-table-001",
              "menu_path": "가상"}
    return TableInfo(**{**fields, **changes})


@pytest.fixture
def frame(pop_path) -> pa.Table:
    return pq.read_table(pop_path)


def test_full_frame_gives_the_file_cells_and_meta(pop_path, frame):
    from_file = load_table(pop_path)
    given = table_input(ParquetTable(frame=frame, info=_info(), fetched_at=from_file.meta.fetched_at))
    assert given.cells == from_file.cells
    assert replace(given.meta, file_sha256=from_file.meta.file_sha256) == from_file.meta
    assert clean(given).cells == clean(from_file).cells


def test_catalog_row_stands_in_for_info(pop_path, frame):
    from_file = load_table(pop_path)
    given = table_input(ParquetTable(frame=frame), catalog={"101_FAKE_TBL_001": catalog_row(pop_path)})
    assert given.cells == from_file.cells
    assert given.meta.fetched_at == ""  # tables.csv has no fetch date and no file date is known here
    assert replace(given.meta, file_sha256="x", fetched_at="x") == replace(from_file.meta, file_sha256="x", fetched_at="x")


def test_fetched_at_is_the_file_date_or_empty(pop_path, frame):
    assert load_table(pop_path).meta.fetched_at == dt.date.fromtimestamp(pop_path.stat().st_mtime).isoformat()
    assert table_input(ParquetTable(frame=frame, info=_info())).meta.fetched_at == ""
    assert table_input(ParquetTable(frame=frame, info=_info(), fetched_at="2026-11-05")).meta.fetched_at == "2026-11-05"


def test_no_info_and_no_catalog_row_is_an_error(frame):
    with pytest.raises(SchemaError, match="FAKE_TBL_001"):
        table_input(ParquetTable(frame=frame), catalog={})


def test_partial_rows_pass_and_are_converted(pop_path, frame):
    part = ParquetTable(frame=frame.slice(0, 3), info=_info(), coverage="partial")
    prepared = prepare(part)
    assert prepared.cells == clean(load_table(pop_path)).cells[:3]
    assert prepared.meta.note.endswith(PARTIAL_NOTE)  # recorded only; no calculation changes (plan 미결 12)


def test_dataframe_gives_the_same_cells(frame):
    as_arrow = table_input(ParquetTable(frame=frame, info=_info()))
    as_pandas = table_input(ParquetTable(frame=frame.to_pandas(), info=_info()))
    assert as_pandas.cells == as_arrow.cells


@pytest.mark.parametrize("drop", ["unit", "tbl_id", "dim1_code", "value_kind"])
def test_missing_required_column_is_an_error(frame, drop):
    with pytest.raises(ValidationError, match=drop):
        ParquetTable(frame=frame.select([c for c in frame.column_names if c != drop]))


def test_no_rows_or_two_tables_is_an_error(frame, household_path):
    with pytest.raises(ValidationError, match="no rows"):
        ParquetTable(frame=frame.slice(0, 0))
    with pytest.raises(ValidationError, match="one tbl_id"):
        ParquetTable(frame=pa.concat_tables([frame, pq.read_table(household_path)]))


def test_stat_input_holds_one_table(frame):
    table = ParquetTable(frame=frame, info=_info())
    assert StatInput(question="2023년 여자 가상 인구는?", tables=[table]).tables[0] is table
    with pytest.raises(ValidationError):
        StatInput(question="", tables=[table])
    with pytest.raises(ValidationError):
        StatInput(question="q", tables=[table, table])


# --- the content hash: a different table never gets the same key ------------------------------------------------


def test_same_content_same_hash_and_column_order_does_not_matter(pop_path, frame):
    first = ParquetTable(frame=frame, info=_info())
    again = ParquetTable(frame=pq.read_table(pop_path), info=_info())
    reordered = ParquetTable(frame=frame.select(list(reversed(frame.column_names))), info=_info())
    assert content_hash(first) == content_hash(again) == content_hash(reordered)
    assert content_hash(first) is content_hash(first)  # computed once per object


def test_one_row_differs(tmp_path, frame):
    rows = pop_rows()
    rows[0] = {**rows[0], "value_raw": "1,001", "value": 1001.0}
    other = pq.read_table(write_table(tmp_path / "other" / "data", "FAKE_TBL_001", rows, POP_CATALOG))
    assert other.num_rows == frame.num_rows
    assert content_hash(ParquetTable(frame=other, info=_info())) != content_hash(ParquetTable(frame=frame, info=_info()))


def test_a_column_name_differs(frame):
    blank = pa.array([""] * frame.num_rows)
    a = ParquetTable(frame=frame.append_column("note_a", blank), info=_info())
    b = ParquetTable(frame=frame.append_column("note_b", blank), info=_info())
    assert content_hash(a) != content_hash(b)


def test_info_or_coverage_differs(frame):
    base = content_hash(ParquetTable(frame=frame, info=_info()))
    assert content_hash(ParquetTable(frame=frame, info=_info(tbl_id="FAKE_TBL_009"))) != base
    assert content_hash(ParquetTable(frame=frame, info=_info(table_name="다른 표"))) != base
    assert content_hash(ParquetTable(frame=frame)) != base
    assert content_hash(ParquetTable(frame=frame, info=_info(), coverage="partial")) != base
