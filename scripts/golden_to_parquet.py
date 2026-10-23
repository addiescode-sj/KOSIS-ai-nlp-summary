"""역할: 골든셋을 Parquet 데이터 기준으로 바꾼다 (표 경로, 파일 해시, 문항별 oracle 범위 `scope`).
Data In:
  --golden eval/golden/stat_golden.jsonl   바꿀 골든셋 (예전 data/cache/*.json 경로도 읽음)
  --data data                              values/<org>_<tbl>.parquet, tables.csv
Data Out:
  --golden 파일을 제자리에서 다시 쓴다. 처음 바꿀 때 원본은 eval/golden/backup/<이름>.cache-v1.jsonl로 남긴다.
  --report (기본 eval/golden/scope_report.md)  문항별 범위·칸 수·글자 수 검토표
범위 규칙 (사용자 결정 10/23: 지역은 같은 수준 전부):
  정답 근거(source_ref)가 있는 문항: 근거의 항목만, 각 분류는 근거 코드와 같은 수준의 코드 전부,
    시점은 근거 시점과 질문에 적힌 연도를 모두 덮는 범위. default_allowed는 최근 5개 시점까지 넓힌다.
  근거가 없는 문항(not_answerable, clarify)과 규칙이 안 맞는 문항: SCOPE_OVERRIDES에 직접 적는다.
데이터를 다시 빌드해 해시가 바뀌었을 때는 --rehash 로 해시만 갱신한다 (scope는 그대로).
실행: 레포 루트에서
  python scripts/golden_to_parquet.py
  python scripts/golden_to_parquet.py --rehash
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kosis_chat.data import scope, serializer  # noqa: E402
from kosis_chat.data.loader import load_table  # noqa: E402
from kosis_chat.data.schema import TableInput  # noqa: E402

CACHE_NAME = re.compile(r"^(?P<key>\d+_[A-Za-z0-9_]+?)_[0-9a-f]{10}\.json$")
REGION = "행정구역별(읍면동)"
# "level_of": every code at the level of that code (scope.same_level). "max": the table's last period.
SCOPE_OVERRIDES: dict[str, dict] = {
    "S-EXT-01": {"periods": ["1970", "max"]},  # 1970년 이후 가장 낮은 해: 이후 시점 전부를 봐야 함
    "S-CLR-01": {"items": ["T10", "T11", "T12"], "dims": {"행정구역별": {"level_of": "21"}}, "periods": ["2023", "2023"]},
    "S-CLR-02": {},
    "S-REF-01": {},
    "S-REF-03": {"items": ["T40"], "dims": {"행정구역별": {"level_of": "29"}}, "periods": ["2005", "2015"]},
    "S-REF-04": {"items": ["T7"], "dims": {REGION: {"level_of": "00"}}, "periods": ["2020", "max"]},
    "S-REF-05": {"periods": ["2022", "2022"]},
    "S-REF-06": {"items": ["T11"], "dims": {"행정구역별": {"level_of": "00"}}, "periods": ["2024", "2024"]},
    "S-REF-07": {"periods": ["2020", "max"]},
    "S-REF-08": {},
    "S-REF-09": {"items": ["T7"], "dims": {REGION: {"level_of": "29010390"}}, "periods": ["2020", "2020"]},
    "S-REF-10": {"periods": ["2021", "max"]},
    "S-REF-11": {"periods": ["2020", "max"]},
}


def parquet_path(table: str, data: Path) -> str:
    """data/cache/101_DT_1B8000H_b2861f86cc.json -> data/values/101_DT_1B8000H.parquet (already Parquet: unchanged)."""
    name = Path(table).name
    if name.endswith(".parquet"):
        return table
    m = CACHE_NAME.match(name)
    if not m:
        raise ValueError(f"cannot map {table!r} to a Parquet table")
    return f"{data.as_posix()}/values/{m.group('key')}.parquet"


def _levels(table: TableInput, name: str, code: str) -> list[str]:
    codes = list(scope.dim_codes(table)[name])
    return scope.same_level(codes, code)


def resolve_override(override: dict, table: TableInput) -> dict:
    periods = sorted({c.period_raw for c in table.cells})
    out: dict = {}
    if override.get("items"):
        out["items"] = list(override["items"])
    for name, spec in (override.get("dims") or {}).items():
        out.setdefault("dims", {})[name] = _levels(table, name, spec["level_of"]) if isinstance(spec, dict) else list(spec)
    if override.get("periods"):
        start, end = override["periods"]
        out["periods"] = [periods[0] if start == "min" else start, periods[-1] if end == "max" else end]
    return out


def oracle_scope(item: dict, table: TableInput) -> dict:
    if item["id"] in SCOPE_OVERRIDES:
        return resolve_override(SCOPE_OVERRIDES[item["id"]], table)
    refs = {c.source_ref: c for c in table.cells}
    facts = [f for f in item.get("required_facts", []) + item.get("aux_facts", []) if f.get("source_ref")]
    if not facts:
        raise ValueError(f"{item['id']}: no source_ref facts; add it to SCOPE_OVERRIDES")
    cells = [refs[f["source_ref"]] for f in facts]
    items = list(dict.fromkeys(c.item_code for c in cells))
    dims: dict[str, list[str]] = {}
    for name in table.dim_names:
        wanted = dict.fromkeys(code for c in cells for code in _levels(table, name, c.dim_codes[name]))
        dims[name] = list(wanted)
    periods = sorted({c.period_raw for c in table.cells})
    lows = [c.period_raw for c in cells]
    asked = scope.question_periods(item["question"], periods)
    if asked:
        lows += list(asked)
    start, end = min(lows), max(lows)
    if item["kind"] == "default_allowed":  # "최근": the model should pick the latest period itself
        start, end = min(start, periods[-scope.RECENT_PERIODS]), periods[-1]
    return {"items": items, "dims": dims, "periods": [start, end]}


def describe(chosen: dict, table: TableInput) -> str:
    labels = scope.dim_codes(table)
    parts = []
    if chosen.get("items"):
        names = {c.item_code: c.item for c in table.cells}
        parts.append("항목 " + ", ".join(names.get(i, i) for i in chosen["items"]))
    for name, codes in (chosen.get("dims") or {}).items():
        shown = [labels[name].get(c, c) for c in codes]
        parts.append(f"{name} {len(codes)}개 ({', '.join(shown[:4])}{' …' if len(shown) > 4 else ''})")
    if chosen.get("periods"):
        parts.append("시점 {} ~ {}".format(*chosen["periods"]))
    return " / ".join(parts) or "표 전체"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--golden", default="eval/golden/stat_golden.jsonl")
    ap.add_argument("--data", default="data")
    ap.add_argument("--report", default="eval/golden/scope_report.md")
    ap.add_argument("--rehash", action="store_true", help="update table_sha256 only (after a data rebuild)")
    args = ap.parse_args(argv)
    golden, data = Path(args.golden), Path(args.data)
    items = [json.loads(line) for line in golden.read_text(encoding="utf-8").splitlines() if line.strip()]

    backup = golden.parent / "backup" / f"{golden.stem}.cache-v1.jsonl"
    if not args.rehash and not backup.exists():
        backup.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(golden, backup)
        print(f"backup: {backup}")

    tables: dict[str, TableInput] = {}
    rows = []
    for item in items:
        path = parquet_path(item["table"], data)
        if path not in tables:
            tables[path] = load_table(path)
            print(f"loaded {path}: {len(tables[path].cells):,} cells")
        table = tables[path]
        item["table"] = path
        item["table_sha256"] = table.meta.file_sha256
        if args.rehash:
            continue
        item["scope"] = oracle_scope(item, table)
        cut = scope.apply(table, scope.Scope.from_dict(item["scope"]))
        chars = len(serializer.serialize(cut, "wide"))
        rows.append((item, Path(path).stem, len(cut.cells), chars, describe(item["scope"], table)))

    golden.write_text("".join(json.dumps(i, ensure_ascii=False) + "\n" for i in items), encoding="utf-8")
    print(f"written: {golden} ({len(items)} items)")
    if rows:
        lines = [
            "# 골든셋 oracle 범위 검토표",
            "",
            f"- 골든셋: `{golden.as_posix()}` / 데이터: `{data.as_posix()}` / 글자 수는 넓은 형식(wide) 직렬화 기준",
            "- 지역 등 분류는 정답과 같은 수준 전부(사용자 결정 10/23). 직접 적은 범위는 `scripts/golden_to_parquet.py` SCOPE_OVERRIDES",
            "",
            "| id | kind | 질문 | 표 | 범위 | 칸 | 글자 |",
            "|---|---|---|---|---|---|---|",
        ]
        for item, tbl, cells, chars, text in rows:
            mark = " (직접)" if item["id"] in SCOPE_OVERRIDES else ""
            question = item["question"].replace("|", "\\|")
            lines.append(f"| {item['id']}{mark} | {item['kind']} | {question} | {tbl} | {text} | {cells:,} | {chars:,} |")
        biggest = max(rows, key=lambda r: r[3])
        lines += ["", f"가장 긴 범위: {biggest[0]['id']} {biggest[3]:,}자"]
        Path(args.report).write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"report: {args.report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
