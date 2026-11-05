"""What the stages share for one table (T1-28 S2): the cleaned TableInput, its index and its lookup caches.

StatResources is built once per table (load_resources) and held by StatPipeline. The lookups the compute stage
derives from a table (cells by key, series, the analysis lookup, the index) live in table.caches (LookupCaches), one
per TableInput object, in place of the module-global dicts keyed by id(table) they replaced; resources.caches is the
same object. Concurrency: no lock (see LookupCaches): a table may be prepared twice by two threads, with equal results.
The prepared-pipeline cache by file stays kosis_chat.api._cached (an lru_cache, Phase 2 calls cache_clear()).
"""

from __future__ import annotations

from dataclasses import dataclass

from kosis_chat.prepare.clean import clean
from kosis_chat.prepare.index import TableIndex, index_of
from kosis_chat.prepare.schemas import LookupCaches, TableInput


@dataclass(frozen=True)
class StatResources:
    table: TableInput  # cleaned
    index: TableIndex  # index_of(table): the same object every stage gets
    org_id: str | None = None

    @property
    def caches(self) -> LookupCaches:
        return self.table.caches


def load_resources(table: TableInput, org_id: str | None = None) -> StatResources:
    """Clean the table and index it (the index is kept in the cleaned table's caches)."""
    cleaned = clean(table)
    return StatResources(table=cleaned, index=index_of(cleaned), org_id=org_id)


__all__ = ["LookupCaches", "StatResources", "load_resources"]
