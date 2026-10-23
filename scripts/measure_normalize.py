"""역할: T1-05 정규화(src/kosis_ai/normalize.py)를 제공 데이터 2,364표 전체에 돌려 plans/T1-05.md의 M1·M2·measure_kind 분포를 잰다.
Data In:  data/tables.csv, data/values/<key>.parquet, data/table_metadata/<key>.json
Data Out: 표준 출력 요약 + --out 경로(기본 data/build/normalize_report.csv)에 표별 한 줄 (gitignore 대상)
표마다 (단위, 항목, 분류 라벨) 고유 조합만 읽어 정규화하므로 셀 전체를 읽지 않는다. 결과는 셀 단위 정규화와 같다
(normalize는 같은 조합에 같은 값을 준다).
실행: 레포 루트에서  python scripts/measure_normalize.py
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from kosis_ai.normalize import LABEL_COLUMNS, _known, normalize  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="data/build/normalize_report.csv")
    args = ap.parse_args()
    data = Path(args.data)
    with open(data / "tables.csv", encoding="utf-8-sig", newline="") as f:
        tables = list(csv.DictReader(f))

    started, rows = time.time(), []
    for t in tables:
        tb = pq.read_table(data / "values" / f"{t['key']}.parquet", columns=["tbl_id", "unit", "item_name", "value_kind", *LABEL_COLUMNS])
        combos = tb.to_pandas().drop_duplicates(["unit", "item_name", *LABEL_COLUMNS])
        out = normalize(combos, t["key"], data)
        null_unit = out[out["unit"].isna()]
        fake = sorted({u for u in out["unit"].dropna().unique() if _known(u) is None})
        rows.append({
            "key": t["key"], "source": t["source"],
            "has_null_unit": bool(len(null_unit)),
            "null_unit_resolved": bool(len(null_unit)) and bool(null_unit["unit_resolved"].notna().all()),
            "unit_sources": "|".join(sorted(null_unit["unit_source"].unique())),
            "fake_units": "|".join(fake),
            "kinds": "|".join(sorted(str(k) for k in out["measure_kind"].unique())),
            "kind_unknown": bool(out["measure_kind"].isna().any()),
        })  # fmt: skip

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    null = [r for r in rows if r["has_null_unit"]]
    print(f"tables {len(rows)}, {time.time() - started:.0f}s -> {out_path}")
    print(f"M1 tables with null-unit cells: {len(null)}, all resolved: {sum(r['null_unit_resolved'] for r in null)}")
    print("   sources used for null-unit cells:", Counter(s for r in null for s in r["unit_sources"].split("|")))
    print(f"M2 tables with a non-unit value in the unit column (dropped): {sum(bool(r['fake_units']) for r in rows)}")
    print("measure_kind per table (any cell):", Counter(k for r in rows for k in r["kinds"].split("|")))
    print(f"tables with some cells of unknown measure_kind: {sum(r['kind_unknown'] for r in rows)}")


if __name__ == "__main__":
    main()
