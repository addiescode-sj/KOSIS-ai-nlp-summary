"""Module input and output (plans/T1-28.md "입출력 계약"): a loaded Parquet table and the question in, the answer, its
chart and spans out. T1-28 S2 (input), S5 (StatOutput, StatOptions; the entry point is kosis_chat.api.StatModule).

ParquetTable is a table already read (all or part of its rows), not a file path: the module never opens files
(kosis_chat/adapters/parquet.py does). Only the column contract is checked here (the columns prepare reads, at least
one row, one tbl_id on every row, info.tbl_id = that tbl_id); the rows themselves are checked by prepare (kosis_chat/prepare/node.py).
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Literal

import pandas as pd
import pyarrow as pa
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, model_validator

from kosis_chat.prepare.convert import PARQUET_COLUMNS, tbl_id_counts

from kosis_ai.schemas import ChartSpec, StatAnswer, TableInfo  # isort: skip


def _columns(frame: pa.Table | pd.DataFrame) -> list[str]:
    return list(frame.column_names) if isinstance(frame, pa.Table) else [str(c) for c in frame.columns]


def _tbl_ids(frame: pa.Table | pd.DataFrame) -> tuple[set, int]:
    """(the distinct tbl_id values, the number of rows without one: None, NaN or blank), by pyarrow.compute."""
    return tbl_id_counts(frame)


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


class StatOptions(BaseModel):
    """How StatModule runs (E3). The defaults are the pipeline's behaviour; only retry_interval, node_overrides and
    catalog_path can differ today (the other fields are the knobs' places: StatModule refuses another value until the
    pipeline reads them)."""

    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)

    retry_interval: float = Field(default=0.0, ge=0)  # seconds before a second model call for one question (HCX 429)
    compose_max_attempts: int = 2
    fill_retry: bool = True
    plan_retry: bool = True
    use_fallback_chatbot: bool = True  # S6 decides the plan-failure answer (plans/T1-28.md S6 (b))
    prompt_versions: dict[str, str] = Field(default_factory=dict)  # empty: each stage's prompts.py default
    node_overrides: dict[str, Callable] = Field(default_factory=dict, exclude=True)  # E4: node name -> step
    catalog_path: Path | None = None  # tables.csv for tables without info; None: kosis_ai.paths.data_dir()/tables.csv


class StatOutput(BaseModel):
    """What StatModule.answer returns: the shared StatAnswer (no chart field since T1-28 S5), the chart as data (Phase 3
    make_chart input; Vega-Lite by kosis_chat.chart.vega.to_vega) and the spans (decision_metadata["spans"] shape, D7)."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1"] = "1"
    answer: StatAnswer
    chart: ChartSpec | None = None
    spans: list[dict] = Field(default_factory=list)

    @model_validator(mode="after")
    def _chart_of_the_answer(self) -> StatOutput:
        # The rule StatAnswer kept for its chart before T1-28 S5, now between the two fields.
        if self.chart is not None:
            if self.answer.outcome != "answered" or self.answer.numbers_origin != "code":
                raise ValueError("a chart belongs to an answered answer whose numbers code computed")
            unknown = [i for i in self.chart.fact_ids if self.answer.result.get(i) is None]
            if unknown:
                raise ValueError(f"chart.fact_ids reference unknown facts: {unknown}")
        return self
