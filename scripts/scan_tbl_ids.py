"""Read-only scan before the tables.csv tbl_id check (plans/T1-28.md S5, 검토 반영 2): every table's rows against its row.

    python scripts/scan_tbl_ids.py                      # data/ of the work root (KOSIS_WORKSPACE, else this folder)
    python scripts/scan_tbl_ids.py --data D:\\kosis\\data --out tbl_id_scan.csv

For every tables.csv row: the Parquet file it names, the distinct tbl_id values of the file's rows, the rows without a
tbl_id (None, NaN, blank), and whether they are exactly the row's tbl_id; also whether the file name
(<org_id>_<tbl_id>.parquet) agrees. Nothing is written except the optional --out CSV of the problem tables. Exit 0
when no table breaks the rule (then kosis_chat/adapters/parquet.py CHECK_CATALOG_TBL_ID can be set True), 1 when some
do, 2 when the catalog cannot be read. Reads only the tbl_id column of each file.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def tbl_id_counts(column) -> tuple[list[str], int]:
    """(the distinct tbl_id values, sorted; the rows without one: null, NaN or blank) by pyarrow.compute, without a
    Python value per row (a 50M-row table stays in Arrow memory)."""
    import pyarrow as pa
    import pyarrow.compute as pc

    if pa.types.is_floating(column.type):
        column = pc.if_else(pc.is_nan(column), pa.scalar(None, column.type), column)
    if not (pa.types.is_string(column.type) or pa.types.is_large_string(column.type)):
        column = pc.cast(column, pa.string())
    if len(column) == 0:
        return [], 0
    blank = pc.fill_null(pc.equal(pc.utf8_trim_whitespace(column), ""), False)
    present = pc.filter(column, pc.and_(pc.invert(blank), pc.is_valid(column)))
    return sorted(pc.unique(present).to_pylist()), column.null_count + int(pc.sum(blank).as_py() or 0)


def scan_one(data_dir: Path, row: dict) -> dict:
    import pyarrow.parquet as pq

    out = {"key": row.get("key", ""), "catalog_tbl_id": row.get("tbl_id", ""), "parquet": row.get("parquet", ""),
           "row_tbl_ids": "", "blank_rows": 0, "rows": 0, "problem": ""}
    path = data_dir / (row.get("parquet") or "")
    if not row.get("parquet") or not path.is_file():
        out["problem"] = "no_file"
        return out
    try:
        column = pq.read_table(path, columns=["tbl_id"]).column("tbl_id")
        ids, blank = tbl_id_counts(column)
    except Exception as exc:  # unreadable file or no tbl_id column
        out["problem"] = f"unreadable: {type(exc).__name__}: {exc}"[:200]
        return out
    out.update(rows=len(column), blank_rows=blank, row_tbl_ids="|".join(ids[:5]))
    problems = []
    if out["blank_rows"]:
        problems.append("blank_tbl_id")
    if ids != [str(row.get("tbl_id"))]:
        problems.append("tbl_id_mismatch" if len(ids) == 1 else f"{len(ids)}_tbl_ids")
    stem = path.stem
    if "_" in stem and stem.split("_", 1)[1] != str(row.get("tbl_id")):
        problems.append("file_name_mismatch")  # reported, not part of the check
    out["problem"] = ",".join(problems)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python scripts/scan_tbl_ids.py", description=__doc__.splitlines()[0])
    parser.add_argument("--data", default=None, help="the data/ folder (default: <KOSIS_WORKSPACE or repo>/data)")
    parser.add_argument("--out", default=None, help="CSV of the problem tables (default: none written)")
    args = parser.parse_args(argv)
    data_dir = Path(args.data) if args.data else Path(os.environ.get("KOSIS_WORKSPACE") or ROOT) / "data"
    catalog = data_dir / "tables.csv"
    try:
        with catalog.open(encoding="utf-8-sig", newline="") as f:
            rows = list(csv.DictReader(f))
    except OSError as exc:
        print(f"cannot read {catalog}: {exc}", file=sys.stderr)
        return 2
    results = []
    for n, row in enumerate(rows, start=1):
        results.append(scan_one(data_dir, row))
        if n % 200 == 0:
            print(f"  {n}/{len(rows)}", flush=True)
    kinds = Counter(p for r in results for p in r["problem"].split(",") if p)
    gated = [r for r in results if any(p in r["problem"] for p in ("blank_tbl_id", "tbl_id_mismatch", "_tbl_ids", "no_file", "unreadable"))]
    print(f"tables.csv rows {len(rows)}, files read {sum(r['rows'] > 0 for r in results)}")
    for kind, count in sorted(kinds.items()):
        print(f"  {kind}: {count}")
    for r in gated[:20]:
        print(f"  {r['key']}: {r['problem']} (catalog {r['catalog_tbl_id']}, rows {r['row_tbl_ids'] or '-'})")
    if args.out:
        with open(args.out, "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(results[0]) if results else ["key"])
            writer.writeheader()
            writer.writerows(r for r in results if r["problem"])
        print(f"saved: {args.out}")
    print("check can be enabled (0 tables break it)" if not gated else f"{len(gated)} table(s) break the check")
    return 0 if not gated else 1


if __name__ == "__main__":
    sys.exit(main())
