"""역할: data_fix의 전제조건(인구 통계표 목록·수록기간)과 실제 받은 수치 데이터를 대조해 표별로 분류하고 누락을 뽑는다.
Data In:
  - data_fix/population_files/data/population_table_catalog.jsonl : 전제조건(2,364표, 표별 수록기간)
  - data_fix/tables/<org>_<tbl>/ : 표 폴더 2,364개를 한곳에 모음
      status.json + *.json.gz 가 있으면 API 수집분, *.zip 이면 대용량 ZIP 수집분
  - data_fix/population/population/bulk_download_status.jsonl : ZIP 파일별 시작·종료 시점
  - data_fix/population/population/통계표별_수치수집현황.csv : 수집기가 기록한 상태(교차 확인용)
Data Out (data_fix/coverage/):
  - table_coverage.csv : 2,364표 전체 + 조건 밖 폴더, 표마다 분류·누락 내용
  - missing_tables.csv : 정상이 아닌 표만
분류: A 정상 / B 수록시점차이(조건은 '년 (시작~끝)'이지만 KOSIS가 일부 연도만 수록, 누락 아님)
      C 수집누락(요청했는데 못 받은 시점) / D 파일누락(등록된 조각·ZIP이 폴더에 없음) / E 데이터없음 / X 조건밖폴더
  - summary.json : 분류별·조사별 개수
실행: 레포 루트에서 python scripts/check_data_coverage.py [--root data_fix]
"""

import argparse
import csv
import glob
import gzip
import json
import os
import re
import unicodedata
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

# 전제조건의 주기 이름 → 비교용 주기 묶음
LABEL_KIND = {"년": "Y", "월": "M", "분기": "Q", "5년": "F", "2년": "F", "부정기": "F", "IR": "F"}
# API 요청/응답 주기 → 비교용 주기 묶음 (Y는 응답에서 A로 올 수 있음, 부정기는 IR로 요청)
API_KIND = {"Y": "Y", "A": "Y", "M": "M", "Q": "Q", "F": "F", "IR": "F"}

CATEGORY = {
    "A": "A_정상",
    "B": "B_수록시점차이",
    "C": "C_수집누락",
    "D": "D_파일누락",
    "E": "E_데이터없음",
    "X": "X_조건밖폴더",
}


def nfc_glob(folder: Path, pattern: str) -> Path:
    """역할: 맥에서 만든 NFD 한글 파일명을 NFC 이름으로 찾는다."""
    for name in os.listdir(folder):
        if unicodedata.normalize("NFC", name) == pattern:
            return folder / name
    raise FileNotFoundError(folder / pattern)


def expand(kind: str, start: str, end: str) -> set[str]:
    """역할: 시작~종료를 주기별 시점 집합으로 펼친다. F(5년·2년·부정기)는 양 끝만 기대한다."""
    if kind == "Y":
        return {str(y) for y in range(int(start[:4]), int(end[:4]) + 1)}
    if kind == "M":
        out, y, m = set(), int(start[:4]), int(start[4:6])
        while (y, m) <= (int(end[:4]), int(end[4:6])):
            out.add(f"{y}{m:02d}")
            y, m = (y + 1, 1) if m == 12 else (y, m + 1)
        return out
    if kind == "Q":
        out, y, q = set(), int(start[:4]), int(start[4:6])
        while (y, q) <= (int(end[:4]), int(end[4:6])):
            out.add(f"{y}{q:02d}")
            y, q = (y + 1, 1) if q == 4 else (y, q + 1)
        return out
    return {start[:4], end[:4]}


def norm_raw_period(kind: str, text: str) -> str:
    """역할: ZIP 목록의 시점 표기(2026.01, 2026 1/4)를 API 표기(202601, 202601)로 맞춘다."""
    text = text.strip()
    if kind == "M":
        m = re.fullmatch(r"(\d{4})\.?(\d{2})", text)
        return f"{m[1]}{m[2]}" if m else text
    if kind == "Q":
        m = re.fullmatch(r"(\d{4})\s*(\d)/4", text) or re.fullmatch(r"(\d{4})0?(\d)", text)
        return f"{m[1]}{int(m[2]):02d}" if m else text
    return text[:4]


def compress(periods: set[str], kind: str) -> str:
    """역할: 누락 시점을 '2011~2015, 2018'처럼 연속 구간으로 줄여 적는다."""
    if not periods:
        return ""
    ordered = sorted(periods)
    nxt = {"Y": lambda p: str(int(p) + 1)}
    if kind == "M":
        nxt["M"] = lambda p: (f"{int(p[:4]) + 1}01" if p[4:] == "12" else f"{p[:4]}{int(p[4:]) + 1:02d}")
    if kind == "Q":
        nxt["Q"] = lambda p: (f"{int(p[:4]) + 1}01" if p[4:] == "04" else f"{p[:4]}{int(p[4:]) + 1:02d}")
    step = nxt.get(kind)
    runs, first, last = [], ordered[0], ordered[0]
    for p in ordered[1:]:
        if step and p == step(last):
            last = p
            continue
        runs.append(first if first == last else f"{first}~{last}")
        first = last = p
    runs.append(first if first == last else f"{first}~{last}")
    return ", ".join(runs)


def load_condition(path: Path) -> list[dict]:
    """역할: 전제조건 표 목록을 읽고 주기별 기대 시점을 붙인다."""
    tables = []
    for line in path.open(encoding="utf-8"):
        r = json.loads(line)
        expected: dict[str, set[str]] = defaultdict(set)
        for p in r["catalog_periods"]:
            kind = LABEL_KIND[p["frequency_label"]]
            expected[kind] |= expand(kind, p["start"], p["end"])
        r["expected"] = expected
        tables.append(r)
    return tables


def read_api(folder: Path) -> dict:
    """역할: API 표 폴더의 status.json에 등록된 조각만 보고 받은 시점과 빠진 파일을 센다."""
    status = json.loads((folder / "status.json").read_text(encoding="utf-8"))
    present = set(os.listdir(folder))
    got: dict[str, set[str]] = defaultdict(set)
    requested: dict[str, set[str]] = defaultdict(set)
    for k, v in status.get("requested_periods", {}).items():
        requested[API_KIND.get(k, k)] |= set(v)
    missing_files, n_files = [], 0
    for c in status.get("chunks", []):
        n_files += 1
        if c.get("file") not in present:
            missing_files.append(c.get("file"))
            continue
        if c.get("status") != "complete":
            continue
        kind = API_KIND.get(c["query"].get("prdSe", ""), "?")
        got[kind] |= set(c.get("periods_returned") or [])
    return {
        "status": status.get("status", "?"),
        "errors": status.get("errors") or [],
        "got": got,
        "requested": requested,
        "n_files": n_files,
        "missing_files": missing_files,
    }


def read_raw(folder: Path, entries: list[dict], open_zip: bool) -> dict:
    """역할: ZIP 목록에 있는 파일이 실제로 있는지 보고, 파일별 시작~종료로 받은 시점을 만든다."""
    present = set(os.listdir(folder)) if folder.exists() else set()
    got: dict[str, set[str]] = defaultdict(set)
    missing_files, broken = [], []
    for e in entries:
        name = Path(e["path"]).name
        if name not in present:
            missing_files.append(name)
            continue
        if open_zip:
            try:
                with zipfile.ZipFile(folder / name) as z:
                    if not z.namelist():
                        broken.append(name)
                        continue
            except zipfile.BadZipFile:
                broken.append(name)
                continue
        kind = API_KIND.get(e["filename"].split("_")[-2], "F")
        s, t = norm_raw_period(kind, e["start"]), norm_raw_period(kind, e["end"])
        got[kind] |= expand(kind, s, t) if kind in "YMQ" else {s, t}
    extra_files = sorted(present - {Path(e["path"]).name for e in entries} - {".DS_Store"})
    return {
        "status": "complete" if entries else "no_index",
        "errors": [],
        "got": got,
        "requested": {},
        "n_files": len(entries),
        "missing_files": missing_files + [f"{b}(열 수 없음)" for b in broken],
        "extra_files": extra_files,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data_fix")
    ap.add_argument("--no-zip-open", action="store_true", help="ZIP을 열어 보는 확인을 건너뛴다")
    args = ap.parse_args()
    root = Path(args.root)
    api_dir = raw_dir = root / "tables"
    pop_dir = root / "population" / "population"
    out_dir = root / "coverage"
    out_dir.mkdir(exist_ok=True)

    tables = load_condition(root / "population_files" / "data" / "population_table_catalog.jsonl")
    bulk: dict[str, list[dict]] = defaultdict(list)
    for line in (pop_dir / "bulk_download_status.jsonl").open(encoding="utf-8"):
        e = json.loads(line)
        bulk[f"{e['org_id']}_{e['table_id']}"].append(e)
    collector = {}
    with nfc_glob(pop_dir, "통계표별_수치수집현황.csv").open(encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            collector[f"{r['기관ID']}_{r['통계표ID']}"] = r
    folders = {n for n in os.listdir(api_dir) if (api_dir / n).is_dir()}
    api_folders = {n for n in folders if (api_dir / n / "status.json").exists()}
    raw_folders = folders - api_folders

    rows = []
    for t in tables:
        key = f"{t['org_id']}_{t['tbl_id']}"
        info, source = None, "없음"
        if key in api_folders and (api_dir / key / "status.json").exists():
            info, source = read_api(api_dir / key), "API"
        elif key in raw_folders or key in bulk:
            info, source = read_raw(raw_dir / key, bulk.get(key, []), not args.no_zip_open), "ZIP"

        # real: 수집기가 KOSIS 메타 기준으로 요청했는데 못 받은 시점(진짜 누락)
        # gap: 조건 범위 안이지만 KOSIS가 수록하지 않아 요청 대상도 아니었던 시점(예: 5년 주기 표의 중간 연도)
        missing_by_kind, gap_by_kind, extra_by_kind = {}, {}, {}
        if info:
            for kind, exp in t["expected"].items():
                got = info["got"].get(kind, set())
                if kind == "F":
                    # 5년·2년·부정기는 조건의 양 끝 시점이 있는지만 본다(중간 시점은 메타에 따라 다름)
                    got = got | info["got"].get("Y", set())
                real = info["requested"].get(kind, set()) - info["got"].get(kind, set())
                gap = exp - got - real
                if real:
                    missing_by_kind[kind] = real
                if gap:
                    gap_by_kind[kind] = gap
            for kind, got in info["got"].items():
                exp = t["expected"].get(kind, set())
                extra = {p for p in got if exp and p > max(exp)}
                if extra:
                    extra_by_kind[kind] = extra

        if info is None:
            cat = "E"
        elif info["status"] != "complete":
            cat = "C"
        elif info["missing_files"]:
            cat = "D"
        elif missing_by_kind:
            cat = "C"
        elif gap_by_kind:
            cat = "B"
        else:
            cat = "A"
        coll = collector.get(key, {})
        rows.append({
            "분류": CATEGORY[cat],
            "조사": t["survey_name"],
            "기관ID": t["org_id"],
            "통계표ID": t["tbl_id"],
            "통계표명": t["table_name"],
            "조건_수록기간": t["period"],
            "받은곳": source,
            "수집상태": info["status"] if info else "",
            "수집기_기록상태": coll.get("상태", ""),
            "파일수": info["n_files"] if info else 0,
            "빠진파일수": len(info["missing_files"]) if info else "",
            "빠진파일": "; ".join(info["missing_files"][:20]) if info else "",
            "빠진시점": " | ".join(f"{k}: {compress(v, k)}" for k, v in sorted(missing_by_kind.items())),
            "빠진시점수": sum(len(v) for v in missing_by_kind.values()),
            "미수록시점(누락아님)": " | ".join(f"{k}: {compress(v, k)}" for k, v in sorted(gap_by_kind.items())),
            "조건보다_최신시점": " | ".join(f"{k}: {compress(v, k)}" for k, v in sorted(extra_by_kind.items())),
            "오류": "; ".join(str(e) for e in (info["errors"] if info else []))[:300],
            "URL": t["paths"][0]["kosis_table_url"],
        })

    condition_keys = {f"{t['org_id']}_{t['tbl_id']}" for t in tables}
    for key in sorted((api_folders | raw_folders) - condition_keys - {".DS_Store"}):
        org, tbl = key.split("_", 1)
        rows.append({"분류": CATEGORY["X"], "기관ID": org, "통계표ID": tbl,
                     "받은곳": "API" if key in api_folders else "ZIP"})

    fields = list(rows[0].keys())
    for name, subset in [("table_coverage.csv", rows),
                         ("missing_tables.csv", [r for r in rows if r["분류"] != CATEGORY["A"]])]:
        with (out_dir / name).open("w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(subset)

    by_cat = Counter(r["분류"] for r in rows)
    by_survey: dict[str, Counter] = defaultdict(Counter)
    for r in rows:
        by_survey[r.get("조사") or "(조건 밖)"][r["분류"]] += 1
    summary = {
        "조건_표수": len(tables),
        "분류별": dict(sorted(by_cat.items())),
        "조사별": {k: dict(sorted(v.items())) for k, v in sorted(by_survey.items())},
        "받은곳별": dict(Counter(r["받은곳"] for r in rows)),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
