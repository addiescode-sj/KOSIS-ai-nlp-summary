"""summarize_stat(question, table_id) -> StatAnswer: the stage 1 public API (T1-11, plans/T1-27.md).

table_id is a data/tables.csv key ("101_DT_1B8000H"), the key stage 2 search returns (TableCandidate.key). The
table is read from the work root's data/ (KOSIS_WORKSPACE, src/kosis_ai/paths.py) and answered by the C pipeline,
as `python -m eval.run_eval --pipeline stat` does for one golden table (without a golden scope: the compact-v2
fallback picks its own table part).

Errors are part of the contract: an unknown table raises LookupError, a failed model call raises RuntimeError.
A question the table cannot answer is not an error: it comes back as a refused or clarify StatAnswer.

The model (T1-28 S2c): pass llm=LLMConfig(api_key=..., model=..., provider=...). Without a key, a question the rules
refuse or ask back is still answered; one that needs the model raises LLMKeyMissing (a RuntimeError). settings= (a
kosis_chat.config.Settings, Phase 2) keeps working: its key is the shell environment variable it names. client= (a
ready client: tests, a recording client) wins over both.

T1-28 S5: StatModule(llm, options).answer(StatInput) -> StatOutput is the module entry point for a table already loaded
(Phase 3): the answer, the chart as data (ChartSpec; StatAnswer has no chart any more) and the spans. Prepared tables are
kept per module by content_hash(table, catalog) (kosis_chat/prepare/node.py), so the same rows and meta are prepared
once and a changed frame or catalog row is prepared again. summarize_stat_run(question, table_id, ...) -> StatOutput is
summarize_stat with the chart and spans (it replaces Phase 2's use of the private _build + ask). summarize_stat keeps
the file route (adapters.parquet.load_table, the pipelines cached per file in _cached); a Parquet file or tables.csv
replaced at the same path (another mtime or size) is read again: the caches are cleared when a stamp changes.
"""

from __future__ import annotations

import csv
import threading
from collections import OrderedDict
from functools import lru_cache
from pathlib import Path

from kosis_chat.adapters.parquet import file_stamp, load_table, parquet_stamp, read_catalog
from kosis_chat.config import Settings
from kosis_chat.legacy.chatbot import Answer
from kosis_chat.legacy.contract_text import to_stat_answer
from kosis_chat.llm.model import LLMClient
from kosis_chat.graph.build import build_graph
from kosis_chat.pipeline import StatPipeline
from kosis_chat.prepare.node import _catalog_row, content_hash, table_input
from kosis_chat.render.source import org_id_from_table_path
from kosis_chat.schemas.io import ParquetTable, StatInput, StatOptions, StatOutput
from kosis_chat.schemas.resources import LLMConfig

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


def _build(path: Path, settings: Settings | None, client: LLMClient | None, retry_interval: float = 0.0,
           llm: LLMConfig | None = None) -> StatPipeline:
    # Deprecated for callers outside this module (Phase 2 evals/trace.py): use summarize_stat_run. Signature kept.
    # adapters open the file (read_parquet + tables.csv), prepare converts; StatPipeline cleans and indexes once
    # (kosis_chat/schemas/resources.py). T1-28 S2. The client is built on first use from llm or settings (S2c).
    table = load_table(path)
    return StatPipeline(table, client=client, settings=settings, org_id=org_id_from_table_path(path, table.meta.tbl_id),
                        retry_interval=retry_interval, llm=llm)


@lru_cache(maxsize=16)
def _cached(path: Path, settings: Settings | None, retry_interval: float) -> StatPipeline:
    """Pipelines without a given client, kept per table file and settings (loading and indexing a big table takes seconds)."""
    # ponytail: no lock; two threads may build the same table twice. Add one when a server calls this (T2-10).
    return _build(path, settings, None, retry_interval)


@lru_cache(maxsize=16)
def _cached_llm(path: Path, llm: LLMConfig, retry_interval: float) -> StatPipeline:
    """_cached for an injected LLMConfig (frozen, hashable; the key stays a SecretStr)."""
    return _build(path, None, None, retry_interval, llm=llm)


def _clear_all() -> None:
    """_cached.cache_clear() also drops the LLMConfig pipelines and the file stamps (Phase 2 tests reset the workspace
    with it); it does not touch StatModule caches (each module owns its own)."""
    _cached.__wrapped_clear__()
    _cached_llm.cache_clear()
    with _STAMP_LOCK:
        _STAMPS.clear()


# (mtime_ns, size, footer rows, footer size) of the Parquet file and (mtime_ns, size) of tables.csv each cached pipeline was built from, by file path.
_STAMPS: dict[Path, tuple] = {}
_STAMP_LOCK = threading.Lock()


def _stamp(path: Path) -> tuple:
    return parquet_stamp(path), file_stamp(paths.data_dir() / "tables.csv")


def _check_fresh(path: Path) -> None:
    """Drop the cached pipelines when the table file or tables.csv changed since this path was last answered.

    One stamp changed clears every cached pipeline (an lru_cache cannot drop one key): files change rarely, and a
    cleared cache only costs a reload. The Parquet stamp also has the footer's row count and size; a file rewritten
    inside one mtime tick with the same size, row count and footer size is not seen."""
    stamp = _stamp(path)
    with _STAMP_LOCK:
        old = _STAMPS.get(path)
    if old is not None and old != stamp:
        _cached.cache_clear()
    with _STAMP_LOCK:
        _STAMPS[path] = stamp


_cached.__wrapped_clear__ = _cached.cache_clear
_cached.cache_clear = _clear_all


def summarize_stat(question: str, table_id: str, *, settings: Settings | None = None,
                   client: LLMClient | None = None, retry_interval: float = 0.0,
                   llm: LLMConfig | None = None) -> StatAnswer:
    """Answer `question` from one provided table.

    retry_interval: seconds to wait before a second model call for the same question (HCX 429), as run_eval --interval.
    A given client (tests, a recording client) builds a fresh, uncached pipeline.
    """
    return summarize_stat_run(question, table_id, settings=settings, client=client, retry_interval=retry_interval,
                              llm=llm).answer


def summarize_stat_run(question: str, table_id: str, *, settings: Settings | None = None,
                       client: LLMClient | None = None, retry_interval: float = 0.0,
                       llm: LLMConfig | None = None) -> StatOutput:
    """summarize_stat with the chart (ChartSpec) and the spans: the same arguments, errors and answer (T1-28 S5)."""
    path = table_path(table_id)
    if client is None:
        _check_fresh(path)
    if llm is None:  # the calls as before S2c (tests monkeypatch _build and _cached with these arguments)
        bot = _cached(path, settings, retry_interval) if client is None else _build(path, settings, client, retry_interval)
    elif client is None and settings is None:
        bot = _cached_llm(path, llm, retry_interval)
    else:
        bot = _build(path, settings, client, retry_interval, llm=llm)
    return _output(question, bot.ask(question), bot.org_id, table_id)


def _output(question: str, answer: Answer, org_id: str | None, label: str) -> StatOutput:
    """The StatOutput of one Answer; RuntimeError when the model call failed (summarize_stat's contract)."""
    stat = to_stat_answer(question, answer, org_id=org_id)
    if stat is None:
        raise RuntimeError(f"{label}: the model call failed ({answer.status}: {answer.error})")
    spans = (answer.decision_metadata or {}).get("spans") or []
    return StatOutput(answer=stat, chart=answer.chart if answer.stat_answer is not None else None, spans=list(spans))


# ---------------------------------------------------------------- the module entry point (T1-28 S5)

# Options StatModule cannot apply yet: the value the pipeline has today (plans/T1-28.md 입출력 계약 StatOptions).
_FIXED_OPTIONS = {"compose_max_attempts": 2, "fill_retry": True, "plan_retry": True, "use_fallback_chatbot": True,
                  "prompt_versions": {}}


class StatModule:
    """The stat module (D25): StatModule(llm=LLMConfig(...), options=StatOptions()).answer(StatInput) -> StatOutput.

    The module opens no table file: the table comes loaded (ParquetTable). Its meta is ParquetTable.info, else the
    tables.csv row with the frame's tbl_id (options.catalog_path, read once per module and again when the file changes).
    Prepared pipelines are kept by content_hash(table, catalog) (at most `cache_size`, least recently used dropped;
    the module lock guards the cache dicts; a table is built under its own key's lock, outside the module lock). client= (tests, a recording client) is used instead of
    building one from llm. A key is needed only when an LLM step runs (LLMKeyMissing, a RuntimeError)."""

    def __init__(self, llm: LLMConfig, options: StatOptions | None = None, *, client: LLMClient | None = None,
                 cache_size: int = 16) -> None:
        self.llm = llm
        self.options = options or StatOptions()
        for name, value in _FIXED_OPTIONS.items():
            if getattr(self.options, name) != value:
                raise ValueError(f"StatOptions.{name}={getattr(self.options, name)!r} is not supported yet (only {value!r})")
        self.client = client
        self.cache_size = cache_size
        self._pipelines: OrderedDict[str, StatPipeline] = OrderedDict()
        self._catalog: tuple[tuple | None, dict] | None = None
        self._lock = threading.Lock()  # the cache dicts only
        self._building: dict[str, threading.Lock] = {}

    def catalog(self) -> dict[str, dict]:
        """tables.csv rows by key (read on first use and again when the file's mtime or size changes)."""
        path = self.options.catalog_path or paths.data_dir() / "tables.csv"
        stamp = file_stamp(path)
        if self._catalog is None or self._catalog[0] != stamp:
            self._catalog = (stamp, read_catalog(path))
        return self._catalog[1]

    def prepared_key(self, table: ParquetTable) -> tuple[str, dict | None]:
        """(content_hash(table, catalog), the catalog used: None when table.info is given)."""
        catalog = None if table.info is not None else self.catalog()
        return content_hash(table, catalog), catalog

    def _cached_pipeline(self, key: str) -> StatPipeline | None:
        with self._lock:
            bot = self._pipelines.get(key)
            if bot is not None:
                self._pipelines.move_to_end(key)
            return bot

    def pipeline(self, table: ParquetTable) -> StatPipeline:
        """The prepared pipeline of a table (built once per content hash).

        The module lock guards only the cache; a table is prepared under its own key's lock, so two threads asking about
        the same table build it once and threads asking about other tables do not wait."""
        key, catalog = self.prepared_key(table)
        bot = self._cached_pipeline(key)
        if bot is not None:
            return bot
        with self._lock:
            building = self._building.setdefault(key, threading.Lock())
        with building:
            bot = self._cached_pipeline(key)  # another thread may have built it meanwhile
            if bot is None:
                bot = self._build_pipeline(table, catalog)
                with self._lock:
                    self._pipelines[key] = bot
                    while len(self._pipelines) > self.cache_size:
                        self._pipelines.popitem(last=False)
        with self._lock:
            self._building.pop(key, None)
        return bot

    def _build_pipeline(self, table: ParquetTable, catalog: dict | None) -> StatPipeline:
        prepared = table_input(table, catalog)  # cleaned and indexed by StatPipeline (load_resources)
        org_id = table.info.org_id if table.info is not None else _catalog_row(prepared.meta.tbl_id, catalog).get("org_id")
        bot = StatPipeline(prepared, client=self.client, org_id=org_id or None,
                           retry_interval=self.options.retry_interval, llm=self.llm)
        if self.options.node_overrides:
            bot._graph = build_graph(bot.resources, node_overrides=dict(self.options.node_overrides))
        return bot

    def answer(self, inp: StatInput) -> StatOutput:
        """One question about the input's table: the StatAnswer, its chart (ChartSpec or None) and spans.

        Errors as summarize_stat: a failed model call raises RuntimeError; a bad table raises SchemaError (ValueError)."""
        (table,) = inp.tables
        bot = self.pipeline(table)
        return _output(inp.question, bot.ask(inp.question), bot.org_id, table.tbl_id())
