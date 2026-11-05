"""Module input (plans/T1-28.md "입출력 계약"): a loaded Parquet table and the question. T1-28 S2, not yet the entry point.

ParquetTable is a table already read (all or part of its rows), not a file path: the module never opens files
(kosis_chat/adapters/parquet.py does). Only the column contract is checked here (the columns prepare reads, at least
one row, one tbl_id on every row, info.tbl_id = that tbl_id); the rows themselves are checked by prepare (kosis_chat/prepare/node.py).
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


def _tbl_ids(frame: pa.Table | pd.DataFrame) -> tuple[set, int]:
    """(the distinct tbl_id values, the number of rows without one: None, NaN or blank)."""
    values = frame.column("tbl_id").to_pylist() if isinstance(frame, pa.Table) else frame["tbl_id"].tolist()
    present = [v for v in values if v is not None and v == v and str(v).strip()]  # v == v drops NaN
    return set(present), len(values) - len(present)


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
    # Hash of the frame's rows (prepare.node), kept only for a pyarrow Table: a DataFrame can be changed in place after
    # the wrapper is made, so its rows are hashed again on every call.
    _frame_hash: str | None = PrivateAttr(default=None)

    def tbl_id(self) -> str:
        """The one tbl_id every row of the frame carries (checked when the object is made)."""
        (tbl_id,) = _tbl_ids(self.frame)[0]
        return str(tbl_id)

    @model_validator(mode="after")
    def _column_contract(self) -> ParquetTable:
        missing = [c for c in PARQUET_COLUMNS if c not in set(_columns(self.frame))]
        if missing:
            raise ValueError(f"frame lacks required columns {missing}")
        if len(self.frame) < 1:
            raise ValueError("frame has no rows")
        ids, blank = _tbl_ids(self.frame)
        if blank:
            raise ValueError(f"{blank} row(s) of the frame have no tbl_id; every row must name its table")
        if len(ids) != 1:
            raise ValueError(f"frame must hold one table (one tbl_id), found {sorted(map(str, ids))}")
        # The meta (table name, URL, source_ref) comes from info: it must describe the table the rows come from.
        (tbl_id,) = ids
        if self.info is not None and str(self.info.tbl_id) != str(tbl_id):
            raise ValueError(f"info.tbl_id {self.info.tbl_id!r} is not the frame's tbl_id {tbl_id!r}")
        return self


class StatInput(BaseModel):
    """One question about one loaded table (E1: a list of tables, of length 1 for now)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal["1"] = "1"  # E2: a change that breaks a field raises the version
    question: str = Field(min_length=1)
    tables: list[ParquetTable] = Field(min_length=1, max_length=1)
