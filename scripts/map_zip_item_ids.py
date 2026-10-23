"""역할: ZIP에서 만든 Parquet 표의 item_id를 공식 메타데이터 코드(T1 등)로 바꿔 API 표와 같은 키를 쓰게 한다.
ZIP CSV에는 항목 코드가 없어서 build_parquet.py가 item_id에 항목명(예: '1인 (가구)')을 넣었다.
이 스크립트는 항목명에서 끝의 단위 괄호와 공백을 뗀 뒤 table_metadata의 itemInfo.itmList(scrKor)와 맞춰 코드로 바꾼다.
Data In (기본 --data data):
  tables.csv                    source=zip 인 표만 고른다
  table_metadata/<key>.json     항목 코드표
  values/<key>.parquet          고칠 표 (item_name은 그대로 두고 item_id만 바꾼다)
Data Out:
  values/<key>.parquet          제자리에서 다시 쓴다 (임시 파일에 쓰고 이름을 바꿈)
  build/zip_item_map.csv        표·항목명별 결과: mapped / nomatch(메타에 없는 이름) / dup(같은 이름이 여러 코드)
  build/build_log.txt           진행 기록을 이어 붙인다
nomatch·dup 항목은 이름을 그대로 둔다. 이미 코드로 바뀐 항목(item_id != item_name)은 건드리지 않아 여러 번 돌려도 같다.
실행: 레포 루트에서
  python scripts/map_zip_item_ids.py --dry-run   # 파일은 안 바꾸고 build/zip_item_map.csv와 요약만
  python scripts/map_zip_item_ids.py
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

BATCH_ROWS = 500_000
UNIT_PAREN = re.compile(r"\s*\([^()]*\)\s*$")


def norm(name: str) -> str:
    return re.sub(r"\s+", "", name)


def meta_codes(meta_path: Path) -> dict[str, list[str]]:
    """역할: 공백을 뗀 항목명 -> 코드 목록. 같은 이름이 여러 코드면 목록이 2개 이상."""
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    codes: dict[str, list[str]] = {}
    for itm in meta["itemInfo"]["itmList"]:
        codes.setdefault(norm(itm.get("scrKor") or ""), []).append(str(itm["itmId"]))
    return codes


def match(name: str, codes: dict[str, list[str]]) -> tuple[str | None, str]:
    ids = codes.get(norm(name)) or codes.get(norm(UNIT_PAREN.sub("", name)))
    if not ids:
        return None, "nomatch"
    if len(ids) > 1:
        return None, "dup"
    return ids[0], "mapped"


def distinct_names(path: Path) -> list[str]:
    """역할: 아직 코드로 안 바뀐(item_id == item_name) 항목명만 모은다."""
    names: set[str] = set()
    with pq.ParquetFile(path) as src:
        for batch in src.iter_batches(batch_size=BATCH_ROWS, columns=["item_id", "item_name"]):
            same = pc.equal(batch.column("item_id"), batch.column("item_name"))
            names.update(pc.unique(pc.filter(batch.column("item_id"), same)).to_pylist())
    names.discard(None)
    return sorted(names)


def replace_with_retry(tmp: Path, path: Path, tries: int = 10) -> None:
    """역할: tmp로 path를 덮어쓴다. OneDrive 동기화·백신이 잠깐 파일을 잡고 있으면 Windows가 거부하므로 몇 번 다시 시도한다."""
    for attempt in range(1, tries + 1):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            if attempt == tries:
                raise
            time.sleep(0.5 * attempt)


def rewrite(path: Path, mapping: dict[str, str]) -> int:
    """역할: item_id만 mapping대로 바꿔 같은 스키마로 다시 쓴다. 바꾼 행 수를 돌려준다.
    Windows는 열려 있는 파일을 덮어쓸 수 없어서 원본을 닫은 뒤에 이름을 바꾼다."""
    tmp = path.with_suffix(".parquet.tmp")
    keys, values = pa.array(list(mapping)), pa.array(list(mapping.values()))
    changed = 0
    with pq.ParquetFile(path) as src:
        writer = pq.ParquetWriter(tmp, src.schema_arrow, compression="zstd")
        try:
            for batch in src.iter_batches(batch_size=BATCH_ROWS):
                table = pa.Table.from_batches([batch], schema=src.schema_arrow)
                item = table.column("item_id").combine_chunks()
                idx = pc.index_in(item, value_set=keys)
                hit = pc.is_valid(idx)
                changed += pc.sum(hit).as_py() or 0
                new = pc.if_else(hit, pc.take(values, idx), item)
                table = table.set_column(table.schema.get_field_index("item_id"), "item_id", new)
                writer.write_table(table)
        except BaseException:
            writer.close()
            tmp.unlink(missing_ok=True)
            raise
        writer.close()
    replace_with_retry(tmp, path)
    return changed


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data", help="Parquet 데이터 폴더 (tables.csv, values/, table_metadata/)")
    ap.add_argument("--dry-run", action="store_true", help="파일은 바꾸지 않고 결과표와 요약만 만든다")
    args = ap.parse_args(argv)
    data = Path(args.data)
    with (data / "tables.csv").open(encoding="utf-8-sig") as f:
        keys = [r["key"] for r in csv.DictReader(f) if r["source"] == "zip" and r["parquet"]]
    (data / "build").mkdir(exist_ok=True)
    log_file = (data / "build" / "build_log.txt").open("a", encoding="utf-8")

    def log(msg: str) -> None:
        line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [map_zip_item_ids] {msg}"
        print(line, flush=True)
        log_file.write(line + "\n")

    log(f"data={data} zip tables={len(keys)} dry_run={args.dry_run}")
    started, status, rows_changed, tables_changed = time.time(), Counter(), 0, 0
    with (data / "build" / "zip_item_map.csv").open("w", encoding="utf-8-sig", newline="") as f:
        out = csv.writer(f)
        out.writerow(["key", "item_name", "item_id", "status"])
        for i, key in enumerate(keys, 1):
            path = data / "values" / f"{key}.parquet"
            codes = meta_codes(data / "table_metadata" / f"{key}.json")
            mapping = {}
            for name in distinct_names(path):
                code, st = match(name, codes)
                status[st] += 1
                out.writerow([key, name, code or "", st])
                if code:
                    mapping[name] = code
            if mapping and not args.dry_run:
                rows_changed += rewrite(path, mapping)
                tables_changed += 1
            if i % 25 == 0 or i == len(keys):
                log(f"[{i}/{len(keys)}] {key} elapsed {(time.time() - started) / 60:.1f} min")
    log(f"finished: items {dict(status)}, tables rewritten {tables_changed}, rows changed {rows_changed:,}")
    log_file.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
