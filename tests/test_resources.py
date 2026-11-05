"""StatResources (T1-28 S2): one cleaned table, one index and one set of lookups per table, shared by the stages.

The lookups that compute/ kept in module-global dicts keyed by id(table) now live in TableInput.caches.
"""

from __future__ import annotations

import re
from dataclasses import replace
from pathlib import Path

import pytest

from kosis_chat.adapters.parquet import load_table
from kosis_chat.compute import engine, resolve, semantics
from kosis_chat.llm.model import LLMResult, MockClient
from kosis_chat.pipeline import StatPipeline
from kosis_chat.prepare import index as index_module
from kosis_chat.prepare.index import index_of
from kosis_chat.schemas.resources import StatResources, load_resources


@pytest.fixture
def builds(monkeypatch) -> list:
    """Every index built through index_of (the one cache of the index)."""
    built = []
    real = index_module.build_index
    monkeypatch.setattr(index_module, "build_index", lambda table: built.append(table) or real(table))
    return built


def test_same_table_asked_twice_does_not_rebuild_the_index(pop_path, mock_settings, builds):
    bot = StatPipeline(load_table(pop_path), client=MockClient(LLMResult(text="not json")), settings=mock_settings,
                       org_id="101")
    assert len(builds) == 1
    first = bot.index
    for _ in range(2):
        bot.ask("2023년 여자 가상 인구는?")
    assert len(builds) == 1
    assert bot.index is first is bot.resources.index is semantics.index_of(bot.table)


def test_resources_hold_the_cleaned_table_and_its_index(pop_path, builds):
    raw = load_table(pop_path)
    resources = load_resources(raw, org_id="101")
    assert isinstance(resources, StatResources) and resources.org_id == "101"
    assert resources.table is not raw and all(c.value_flag is not None for c in resources.table.cells)
    assert resources.index is index_of(resources.table) and resources.index.table is resources.table
    assert resources.caches is resources.table.caches
    assert len(builds) == 1


def test_lookups_are_kept_per_table_object(pop_path):
    table = load_resources(load_table(pop_path)).table
    assert engine._indexed(table)[1] is engine._indexed(table)[1]
    assert resolve._cells_by_key(table) is resolve._cells_by_key(table)
    other = replace(table, cells=table.cells)  # a new TableInput: its own, empty caches
    assert len(other.caches) == 0 and other.caches is not table.caches
    assert engine._indexed(other)[1] is not engine._indexed(table)[1]
    assert engine._indexed(other)[1] == engine._indexed(table)[1]
    assert index_of(other) is not index_of(table)


CHAT = Path(__file__).resolve().parents[1] / "kosis_chat"
# api._cached: the pipeline per table file, Phase 2 calls _cached.cache_clear() (docs/compat-surface.md).
KEPT_CACHES = {"api.py"}


def test_no_module_cache_keyed_by_table_object():
    """grep: no cache keyed by id(table); every other lru_cache says why it stays (a '# ponytail:' line above it)."""
    problems = []
    for path in sorted(CHAT.rglob("*.py")):
        rel = path.relative_to(CHAT).as_posix()
        lines = path.read_text(encoding="utf-8").splitlines()
        problems += [f"{rel}:{n}" for n, line in enumerate(lines, 1) if re.search(r"[\[(]id\(table\)[\])]", line)]
        problems += [f"{rel}:{n}" for n, line in enumerate(lines, 1) if line.startswith("@lru_cache")
                     and rel not in KEPT_CACHES and not any("# ponytail:" in above for above in lines[max(0, n - 4):n - 1])]
    assert not problems, problems
