"""Module input (plans/T1-28.md "입출력 계약"): a loaded Parquet table and the question. T1-28 S2, not yet the entry point.

ParquetTable is a table already read (all or part of its rows), not a file path: the module never opens files
(kosis_chat/adapters/parquet.py does). Only the column contract is checked here (the columns prepare reads, at least
one row, one tbl_id); the rows themselves are checked by prepare (kosis_chat/prepare/node.py).
StatOutput, StatOptions and ChartSpec come with the entry point (T1-28 S5).
"""

from __future__ import annotations

from typing import Literal

import pandas as pd
import pyarrow as pa
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, model_validator

from kosis_chat.prepare.convert import PARQUET_COLUMNS

from kosis_ai.schemas import TableInfo  # isort: skip


def _columns(frame: pa.Table | pd.DataFrame) -> list[str]:
    return list(frame.column_names) if isinstance(frame, pa.Table) else [str(c) for c in frame.columns]


def _tbl_ids(frame: pa.Table | pd.DataFrame) -> set:
    values = frame.column("tbl_id").to_pylist() if isinstance(frame, pa.Table) else frame["tbl_id"].tolist()
    return {v for v in values if v is not None and v == v}  # v == v drops NaN


class ParquetTable(BaseModel):
    """A loaded Parquet table (all or part of its rows). Not a file path."""

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True, extra="forbid")

    frame: pa.Table | pd.DataFrame
    info: TableInfo | None = None  # None: the catalog row (tables.csv) is looked up (D23 c)
    # "partial": the frame holds part of the table's rows. Totals, ranks, max/min and shares assume the whole table;
    # until that is decided (plans/T1-28.md 미결 12, recommended (다)) it is only recorded (TableMeta.note), not acted on.
    coverage: Literal["full", "partial"] = "full"
    # Date the values were fetched (YYYY-MM-DD). Kept here, not on kosis_ai TableInfo: Phase 2 writes TableInfo into
    # tables.jsonl, so a new TableInfo field would change its files. None: TableMeta.fetched_at = "".
    fetched_at: str | None = None
    # The content hash (prepare.node.content_hash), computed once per object.
    _content_hash: str | None = PrivateAttr(default=None)

    @model_validator(mode="after")
    def _column_contract(self) -> ParquetTable:
        missing = [c for c in PARQUET_COLUMNS if c not in set(_columns(self.frame))]
        if missing:
            raise ValueError(f"frame lacks required columns {missing}")
        if len(self.frame) < 1:
            raise ValueError("frame has no rows")
        ids = _tbl_ids(self.frame)
        if len(ids) != 1:
            raise ValueError(f"frame must hold one table (one tbl_id), found {sorted(map(str, ids))}")
        return self


class StatInput(BaseModel):
    """One question about one loaded table (E1: a list of tables, of length 1 for now)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal["1"] = "1"  # E2: a change that breaks a field raises the version
    question: str = Field(min_length=1)
    tables: list[ParquetTable] = Field(min_length=1, max_length=1)
