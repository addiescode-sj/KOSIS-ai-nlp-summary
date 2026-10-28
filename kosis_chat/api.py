"""summarize_stat(question, table_id) -> StatAnswer: the stage 1 public API (T1-11, plans/T1-27.md).

table_id is a data/tables.csv key ("101_DT_1B8000H"), the key stage 2 search returns (TableCandidate.key). The
table is read from the work root's data/ (KOSIS_WORKSPACE, src/kosis_ai/paths.py) and answered by the C pipeline,
as `python -m eval.run_eval --pipeline stat` does for one golden table (without a golden scope: the compact-v2
fallback picks its own table part).

Errors are part of the contract: an unknown table raises LookupError, a failed model call raises RuntimeError.
A question the table cannot answer is not an error: it comes back as a refused or clarify StatAnswer.
"""

from __future__ import annotations

import csv
from functools import lru_cache
from pathlib import Path

from kosis_chat.config import Settings
from kosis_chat.contract import org_id_from_table_path, to_stat_answer
from kosis_chat.data.loader import load_table
from kosis_chat.llm.model import LLMClient
from kosis_chat.pipeline import StatPipeline

from kosis_ai import paths  # isort: skip
from kosis_ai.schemas import StatAnswer  # isort: skip


def table_path(table_id: str) -> Path:
    """The Parquet file of a tables.csv key, from the tables.csv `parquet` column (relative to data/)."""
    data_dir = paths.data_dir()
    catalog = data_dir / "tables.csv"
    if not catalog.exists():
        raise LookupError(f"no tables.csv in {data_dir} (set {paths.ENV_VAR} to the folder that holds data/)")
    with catalog.open(encoding="utf-8-sig", newline="") as f:
        row = next((r for r in csv.DictReader(f) if r.get("key") == table_id), None)
    if row is None:
        raise LookupError(f"{table_id} is not in {catalog}")
    path = data_dir / row["parquet"]
    if not path.exists():
        raise LookupError(f"{table_id}: {path} does not exist")
    return path


def _build(path: Path, settings: Settings | None, client: LLMClient | None, retry_interval: float = 0.0) -> StatPipeline:
    table = load_table(path)
    return StatPipeline(table, client=client, settings=settings, org_id=org_id_from_table_path(path, table.meta.tbl_id),
                        retry_interval=retry_interval)


@lru_cache(maxsize=16)
def _cached(path: Path, settings: Settings | None, retry_interval: float) -> StatPipeline:
    """Pipelines without a given client, kept per table file and settings (loading and indexing a big table takes seconds)."""
    # ponytail: no lock; two threads may build the same table twice. Add one when a server calls this (T2-10).
    return _build(path, settings, None, retry_interval)


def summarize_stat(question: str, table_id: str, *, settings: Settings | None = None,
                   client: LLMClient | None = None, retry_interval: float = 0.0) -> StatAnswer:
    """Answer `question` from one provided table.

    retry_interval: seconds to wait before a second model call for the same question (HCX 429), as run_eval --interval.
    A given client (tests, a recording client) builds a fresh, uncached pipeline.
    """
    path = table_path(table_id)
    bot = _cached(path, settings, retry_interval) if client is None else _build(path, settings, client, retry_interval)
    answer = bot.ask(question)
    stat = to_stat_answer(question, answer, org_id=bot.org_id)
    if stat is None:
        raise RuntimeError(f"{table_id}: the model call failed ({answer.status}: {answer.error})")
    return stat
