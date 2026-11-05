"""Old path of the table loader, split in T1-28 S2 (every name re-exported here):

- kosis_chat/adapters/parquet.py opens the files: load_table (read_parquet + prepare.convert), sha256_of, the
  tables.csv lookup and the file date.
- kosis_chat/prepare/convert.py turns the rows into cells: infer_unit and the column / period constants.
"""

from __future__ import annotations

from kosis_chat.adapters.parquet import catalog_row, fetched_at_of, find_catalog, load_parquet, load_table, parquet_meta
from kosis_chat.adapters.parquet import read_parquet, sha256_of
from kosis_chat.prepare.convert import (
    _PAREN,
    DIM_LEVELS,
    KNOWN_UNITS,
    PARQUET_COLUMNS,
    PRD_SE,
    UNIT_OVERRIDES,
    UNKNOWN_UNIT,
    _unit_in_parens,
    build_meta,
    infer_unit,
)
from kosis_chat.prepare.schemas import Cell, Problem, SchemaError, TableInput, TableMeta, validate, validate_meta_dict

# The private names the loader had before the split (same objects).
_DIM_LEVELS = DIM_LEVELS
_PRD_SE = PRD_SE
_PARQUET_COLUMNS = PARQUET_COLUMNS
_build_meta = build_meta
_find_catalog = find_catalog
_parquet_meta = parquet_meta
_load_parquet = load_parquet

__all__ = ["KNOWN_UNITS", "UNIT_OVERRIDES", "UNKNOWN_UNIT", "Cell", "Problem", "SchemaError", "TableInput", "TableMeta",
           "_DIM_LEVELS", "_PAREN", "_PARQUET_COLUMNS", "_PRD_SE", "_build_meta", "_find_catalog", "_load_parquet",
           "_parquet_meta", "_unit_in_parens", "catalog_row", "fetched_at_of", "infer_unit", "load_table",
           "read_parquet", "sha256_of", "validate", "validate_meta_dict"]
