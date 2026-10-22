"""Full-data EDA: profile every KOSIS table in the shared dump, one table at a time, safe for long unattended runs.

Input (default): data_fix/
  tables/<org>_<tbl>/status.json + *.json.gz        OpenAPI rows (long format), read only the chunks in status.json
  tables/<org>_<tbl>/*.zip                          KOSIS bulk CSV (wide format, CP949 or UTF-8)
  population/population/table_metadata/<org>_<tbl>.json   official table metadata
  (the older layout numeric_api/ + numeric_raw/ + table_metadata/ under --root still works)
Output (default): data/eda_full/
  profiles/<org>_<tbl>.json   one profile per table (written atomically; existing ones are skipped -> resume)
  errors.jsonl                one line per failed table (the run keeps going)
  run_log.txt                 progress log with timestamps and ETA
  summary/*.csv               flat tables for the notebook (tables, dims, units, symbols, periods, issues)

Run from the repo root:
  python scripts/eda_full.py                 # all tables, resumes where it stopped
  python scripts/eda_full.py --limit 20      # quick trial on 20 tables
  python scripts/eda_full.py --summarize-only
Stop any time with Ctrl+C; run the same command again to continue.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import gzip
import io
import json
import math
import os
import re
import sys
import time
import traceback
import unicodedata
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

PROFILE_VERSION = "eda-full-v1"
TOTAL_NAME = re.compile(r"^\s*(계|합계|총계|전체|전국|소계|총인구|총수|total|Total|TOTAL)\s*$|^(총|전체)\s|-계$|_계$")
PAREN_UNIT = re.compile(r"\(([^()]{1,15})\)\s*$")
NUMERIC = re.compile(r"^[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?$")
PROVISIONAL = re.compile(r"^(?P<num>[+-]?[\d.,]+)\s*\(?\s*(p|e|P|E)\s*\)?$")
SYMBOL_GROUPS = {"": "blank", "-": "dash", "–": "dash", "－": "dash", "x": "x_secret", "X": "x_secret", "Ｘ": "x_secret",
                 "…": "ellipsis", "...": "ellipsis", "..": "ellipsis", "*": "asterisk", "(X)": "x_secret"}
MAX_EXAMPLES = 15


# ---------------------------------------------------------------- helpers

def nfc(s: str) -> str:
    return unicodedata.normalize("NFC", s)


def now() -> str:
    return dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


class Log:
    def __init__(self, path: Path):
        self.path = path

    def __call__(self, msg: str) -> None:
        line = f"[{now()}] {msg}"
        try:
            print(line, flush=True)
        except UnicodeEncodeError:  # Windows console without UTF-8
            print(line.encode("ascii", "replace").decode(), flush=True)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")


def atomic_write_json(path: Path, obj) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    os.replace(tmp, path)


def iter_json_array(text_stream, chunk_size: int = 1 << 20):
    """Yield objects of a top-level JSON array without loading the whole file (keeps memory flat for big chunks)."""
    decoder = json.JSONDecoder()
    buf = ""
    started = False
    eof = False
    pos = 0
    while True:
        if not eof and len(buf) - pos < chunk_size:
            more = text_stream.read(chunk_size)
            if more:
                buf = buf[pos:] + more
                pos = 0
            else:
                eof = True
        # skip separators
        while pos < len(buf) and buf[pos] in " \t\r\n,":
            pos += 1
        if pos >= len(buf):
            if eof:
                return
            continue
        ch = buf[pos]
        if not started:
            if ch != "[":
                raise ValueError(f"not a JSON array (starts with {ch!r})")
            started = True
            pos += 1
            continue
        if ch == "]":
            return
        try:
            obj, end = decoder.raw_decode(buf, pos)
        except json.JSONDecodeError:
            if eof:
                raise
            more = text_stream.read(chunk_size)
            if not more:
                eof = True
            buf = buf[pos:] + more
            pos = 0
            continue
        yield obj
        pos = end


def classify_value(raw) -> tuple[float | None, str]:
    """(numeric value or None, kind). kind: ok / provisional / one of SYMBOL_GROUPS values / other_text."""
    if raw is None:
        return None, "blank"
    s = str(raw).strip()
    if s in SYMBOL_GROUPS:
        return None, SYMBOL_GROUPS[s]
    t = s.replace(",", "")
    if NUMERIC.match(t):
        try:
            v = float(t)
            return (v, "ok") if math.isfinite(v) else (None, "other_text")
        except ValueError:
            return None, "other_text"
    m = PROVISIONAL.match(s)
    if m:
        try:
            return float(m.group("num").replace(",", "")), "provisional"
        except ValueError:
            pass
    return None, "other_text"


def decimals_of(raw) -> int:
    s = str(raw).strip().replace(",", "")
    if "." not in s:
        return 0
    frac = s.split(".", 1)[1]
    frac = frac.rstrip("0") if re.fullmatch(r"\d+", frac or "") else frac
    return len(re.match(r"\d*", frac).group(0))


def period_shape(p: str) -> str:
    return re.sub(r"\d", "9", p)


def period_gaps(prd_se: str, periods: list[str]) -> dict:
    """Missing periods inside [first, last] for regular cycles. Irregular cycles (F, IR, ...) are not checked."""
    try:
        if prd_se in ("Y", "A"):
            ys = sorted({int(p[:4]) for p in periods if p[:4].isdigit()})
            if len(ys) < 2:
                return {"checked": False}
            full = set(range(ys[0], ys[-1] + 1))
            missing = sorted(full - set(ys))
            step = Counter(b - a for a, b in zip(ys, ys[1:])).most_common(1)[0][0]
            return {"checked": True, "missing": len(missing), "missing_examples": [str(m) for m in missing[:MAX_EXAMPLES]],
                    "typical_step_years": step}
        if prd_se == "M":
            idx = sorted({int(p[:4]) * 12 + int(p[4:6]) - 1 for p in periods if len(p) >= 6 and p[:6].isdigit()})
            if len(idx) < 2:
                return {"checked": False}
            missing = sorted(set(range(idx[0], idx[-1] + 1)) - set(idx))
            return {"checked": True, "missing": len(missing),
                    "missing_examples": [f"{m // 12}{m % 12 + 1:02d}" for m in missing[:MAX_EXAMPLES]]}
        if prd_se in ("Q", "H"):
            n = 4 if prd_se == "Q" else 2
            idx = []
            for p in periods:
                if len(p) >= 6 and p[:6].isdigit() and 1 <= int(p[4:6]) <= n:
                    idx.append(int(p[:4]) * n + int(p[4:6]) - 1)
            idx = sorted(set(idx))
            if len(idx) < 2:
                return {"checked": False}
            missing = sorted(set(range(idx[0], idx[-1] + 1)) - set(idx))
            return {"checked": True, "missing": len(missing),
                    "missing_examples": [f"{m // n}{m % n + 1:02d}" for m in missing[:MAX_EXAMPLES]]}
    except Exception as exc:  # never fail a table because of an odd period code
        return {"checked": False, "error": str(exc)[:200]}
    return {"checked": False}


# ---------------------------------------------------------------- readers -> uniform observations
# observation = (dims: tuple[(dim_name, code, label)], item_id, item_name, unit, prd_se, period, raw_value)

API_PRD = {"A": "Y", "Y": "Y", "M": "M", "Q": "Q", "H": "H", "F": "F", "IR": "IR", "D": "D", "S": "S"}


def read_api(folder: Path, stats: dict):
    status = json.loads((folder / "status.json").read_text(encoding="utf-8"))
    stats["api_status"] = status.get("status")
    stats["api_status_rows"] = status.get("rows")
    stats["api_errors"] = [str(e)[:300] for e in (status.get("errors") or [])][:10]
    stats["api_chunks"] = len(status.get("chunks") or [])
    stats["api_requested_periods"] = {k: len(v) for k, v in (status.get("requested_periods") or {}).items()} \
        if isinstance(status.get("requested_periods"), dict) else None
    field_counter = stats.setdefault("api_fields", Counter())
    bytes_read = 0
    chunks = status.get("chunks") or []
    stats["api_chunk_source"] = "status.json"
    if not chunks and status.get("status") == "running":
        # The collector was stopped mid-table: status.json has no chunk list yet, but each finished
        # request left a <hash>.query.json marked complete. Use those so the partial data is profiled.
        chunks = []
        for q in sorted(folder.glob("*.query.json")):
            try:
                info = json.loads(q.read_text(encoding="utf-8"))
            except Exception:
                continue
            # completed_at >= started_at keeps only this run's pieces, not cache left from older runs
            if (info.get("status") == "complete" and (info.get("query") or {}).get("tblId") == status.get("table_id")
                    and str(info.get("completed_at") or "") >= str(status.get("started_at") or "")):
                chunks.append(info)
        stats["api_chunk_source"] = "query.json (running table)"
        stats["api_chunks"] = len(chunks)
    for chunk in chunks:
        file = chunk.get("file")
        if not file or chunk.get("status") not in (None, "complete"):
            stats.setdefault("skipped_chunks", 0)
            stats["skipped_chunks"] += 1
            continue
        path = folder / file
        if not path.exists():
            stats.setdefault("missing_chunk_files", []).append(file)
            continue
        bytes_read += path.stat().st_size
        with gzip.open(path, "rt", encoding="utf-8", errors="replace") as stream:
            for row in iter_json_array(stream):
                if not isinstance(row, dict):
                    continue
                if len(field_counter) < 200:
                    field_counter.update(row.keys())
                dims = []
                for lv in range(1, 9):
                    code = row.get(f"C{lv}")
                    if code is None and row.get(f"C{lv}_NM") is None:
                        continue
                    dims.append((str(row.get(f"C{lv}_OBJ_NM") or f"C{lv}"), str(code), str(row.get(f"C{lv}_NM") or "")))
                prd = str(row.get("PRD_SE") or "")
                yield (tuple(dims), str(row.get("ITM_ID") or ""), str(row.get("ITM_NM") or ""),
                       row.get("UNIT_NM"), API_PRD.get(prd, prd), str(row.get("PRD_DE") or ""), row.get("DT"))
    stats["bytes"] = bytes_read


def detect_encoding(head: bytes) -> str:
    cut = head.rfind(b"\n")
    sample = head[:cut] if cut > 0 else head
    try:
        sample.decode("utf-8")
        return "utf-8-sig"
    except UnicodeDecodeError:
        return "cp949"


def parse_bulk_csv(binary_stream, prd_se: str, stats: dict, source: str):
    head = binary_stream.read(1 << 16)
    enc = detect_encoding(head)
    stats.setdefault("encodings", Counter())[enc] += 1
    rest = io.BufferedReader(_Concat(head, binary_stream))
    text = io.TextIOWrapper(rest, encoding=enc, errors="replace", newline="")
    reader = csv.reader(text)
    title = None
    header = None
    for row in reader:
        if not row or not any(c.strip() for c in row):
            continue
        if header is None and "시점" in [c.strip() for c in row]:
            header = [c.strip() for c in row]
            break
        if title is None:
            title = row[0].strip()
    if header is None:
        stats.setdefault("bulk_files_without_header", []).append(source)
        return
    stats.setdefault("bulk_titles", Counter())[title or ""] += 1
    period_col = header.index("시점")
    dim_cols = []  # (code_idx, name_idx, dim_name)
    i = 0
    used = {period_col}
    while i < len(header) - 1:
        if i not in used and header[i].startswith("C") and header[i][1:] == header[i + 1]:
            dim_cols.append((i, i + 1, header[i + 1]))
            used.update((i, i + 1))
            i += 2
        else:
            i += 1
    item_cols = [(j, header[j]) for j in range(len(header)) if j not in used]
    stats.setdefault("bulk_item_columns", Counter()).update(name for _, name in item_cols)
    for row in reader:
        if not row or len(row) <= period_col:
            continue
        dims = tuple((name, row[ci].lstrip("'").strip(), row[ni].strip()) for ci, ni, name in dim_cols if ci < len(row) and ni < len(row))
        period = row[period_col].lstrip("'").strip().replace(".", "")
        for j, name in item_cols:
            raw = row[j] if j < len(row) else ""
            m = PAREN_UNIT.search(name)
            yield (dims, name, name, m.group(1).strip() if m else None, prd_se, period, raw)


class _Concat(io.RawIOBase):
    """Raw stream = bytes already read + the rest of another stream (zip members are not seekable)."""

    def __init__(self, first: bytes, rest):
        self.first, self.rest, self.pos = first, rest, 0

    def readable(self):
        return True

    def readinto(self, b):
        if self.pos < len(self.first):
            n = min(len(b), len(self.first) - self.pos)
            b[:n] = self.first[self.pos:self.pos + n]
            self.pos += n
            return n
        data = self.rest.read(len(b))
        n = len(data)
        b[:n] = data
        return n


def read_bulk(folder: Path, stats: dict):
    files = sorted(folder.iterdir())
    zip_stems = {p.stem for p in files if p.suffix.lower() == ".zip"}
    bytes_read = 0
    stats["bulk_files"] = 0
    for p in files:
        name = nfc(p.name)
        if name.startswith("."):
            continue
        if p.suffix.lower() == ".csv" and p.stem in zip_stems:
            stats.setdefault("skipped_loose_csv", []).append(name)
            continue
        m = re.match(r".*_([A-Z]+)_(\d+)\.(zip|csv)$", name, re.IGNORECASE)
        prd_se = m.group(1).upper() if m else "?"
        stats.setdefault("bulk_prd_files", Counter())[prd_se] += 1
        bytes_read += p.stat().st_size
        stats["bulk_files"] += 1
        if p.suffix.lower() == ".zip":
            try:
                zf = zipfile.ZipFile(p)
            except zipfile.BadZipFile:
                stats.setdefault("bad_zip", []).append(name)
                continue
            with zf:
                for info in zf.infolist():
                    if info.is_dir() or not info.filename.lower().endswith(".csv"):
                        continue
                    stats.setdefault("bulk_uncompressed_bytes", 0)
                    stats["bulk_uncompressed_bytes"] += info.file_size
                    with zf.open(info) as raw:
                        yield from parse_bulk_csv(raw, prd_se, stats, f"{name}:{info.filename}")
        elif p.suffix.lower() == ".csv":
            with p.open("rb") as raw:
                yield from parse_bulk_csv(raw, prd_se, stats, name)
    stats["bytes"] = bytes_read


# ---------------------------------------------------------------- profiler

class Profiler:
    """Consumes observations of one table and keeps bounded-size statistics."""

    def __init__(self, max_store: int):
        self.max_store = max_store
        self.n = 0
        self.dims = {}  # dim name -> {"codes": {code: label}, "len": Counter}
        self.dim_order = []
        self.items = {}  # item id -> name
        self.units = Counter()
        self.units_by_item = defaultdict(set)
        self.kinds = Counter()
        self.symbols = Counter()
        self.other_text = Counter()
        self.periods = defaultdict(set)
        self.decimals = Counter()
        self.decimals_by_item = defaultdict(int)
        self.neg = self.zero = 0
        self.vmin = math.inf
        self.vmax = -math.inf
        self.abs_log10 = Counter()
        self.label_chars = 0
        self.store: dict | None = {}
        self.dup_keys = 0
        self.dims_per_obs = Counter()

    def add(self, obs) -> None:
        dims, item_id, item_name, unit, prd_se, period, raw = obs
        self.n += 1
        self.dims_per_obs[len(dims)] += 1
        for name, code, label in dims:
            d = self.dims.get(name)
            if d is None:
                d = self.dims[name] = {"codes": {}, "len": Counter()}
                self.dim_order.append(name)
            if code not in d["codes"]:
                if len(d["codes"]) < 200_000:
                    d["codes"][code] = label
                    d["len"][len(code)] += 1
        if item_id not in self.items and len(self.items) < 50_000:
            self.items[item_id] = item_name
        u = "" if unit is None else str(unit).strip()
        self.units[u if unit is not None else "<필드 없음>"] += 1
        if len(self.units_by_item) < 50_000:
            self.units_by_item[item_id].add(u)
        if len(self.periods[prd_se]) < 100_000:
            self.periods[prd_se].add(period)
        value, kind = classify_value(raw)
        self.kinds[kind] += 1
        if kind not in ("ok", "provisional"):
            s = "" if raw is None else str(raw).strip()
            self.symbols[s[:20]] += 1
            if kind == "other_text" and len(self.other_text) < 500:
                self.other_text[s[:40]] += 1
        else:
            dec = decimals_of(raw)
            self.decimals[dec] += 1
            if dec > self.decimals_by_item[item_id]:
                self.decimals_by_item[item_id] = dec
            if value < 0:
                self.neg += 1
            elif value == 0:
                self.zero += 1
            self.vmin = min(self.vmin, value)
            self.vmax = max(self.vmax, value)
            if value != 0:
                self.abs_log10[int(math.floor(math.log10(abs(value))))] += 1
        # rough prompt size: labels + period + value + unit per row
        self.label_chars += sum(len(lbl) for _, _, lbl in dims) + len(item_name) + len(period) + len(str(raw or "")) + len(u) + 8
        if self.store is not None:
            key = (tuple(c for _, c, _ in dims), item_id, prd_se, period)
            if key in self.store:
                self.dup_keys += 1
            elif len(self.store) >= self.max_store:
                self.store = None  # too big: skip key-level checks for this table
            else:
                self.store[key] = value if kind in ("ok", "provisional") else None

    # ---- key-level checks (only when the table fit in the store)
    def sum_checks(self) -> list[dict]:
        if not self.store:
            return []
        out = []
        n_dims = len(self.dim_order)
        axes = [(i, self.dim_order[i]) for i in range(n_dims)] + [(-1, "__항목__")]
        for axis, axis_name in axes:
            if axis >= 0:
                labels = self.dims[axis_name]["codes"]
            else:
                labels = self.items
            totals = [c for c, lbl in labels.items() if TOTAL_NAME.search(lbl or "")]
            if not totals or len(labels) < 3:
                continue
            others = [c for c in labels if c not in totals]
            if not others:
                continue
            min_len = min(len(c) for c in others)
            siblings = set(c for c in others if len(c) == min_len)
            total_set = set(totals)
            groups = defaultdict(dict)
            for (codes, item_id, prd_se, period), v in self.store.items():
                if axis >= 0:
                    if axis >= len(codes):
                        continue
                    code = codes[axis]
                    rest = (codes[:axis] + codes[axis + 1:], item_id, prd_se, period)
                else:
                    code = item_id
                    rest = (codes, prd_se, period)
                if code in total_set or code in siblings:
                    groups[rest][code] = v
            checked = exact = close = 0
            ratios = []
            for g in groups.values():
                t = next((g[c] for c in totals if c in g), None)
                kids = [g.get(c) for c in siblings]
                if t is None or any(k is None for k in kids) or not kids:
                    continue
                s = sum(kids)
                checked += 1
                diff = abs(t - s)
                exact += diff < 1e-9
                close += diff <= max(0.5 * len(kids) * 10 ** -3, abs(t) * 1e-3)
                if t:
                    ratios.append(s / t)
            if checked:
                ratios.sort()
                out.append({"axis": axis_name, "total_labels": sorted({labels[c] for c in totals})[:5],
                            "siblings": len(siblings), "sibling_code_len": min_len, "checked": checked,
                            "exact": exact, "within_0.1pct": close,
                            "ratio_sum_over_total_p10_p50_p90": [round(ratios[int(len(ratios) * q)], 4) for q in (0.1, 0.5, 0.9)] if ratios else None})
        return out

    def completeness(self) -> dict:
        if not self.store:
            return {"checked": False}
        out = {"checked": True, "by_cycle": {}}
        per = defaultdict(lambda: {"obs": 0, "codes": defaultdict(set), "items": set(), "periods": set()})
        for (codes, item_id, prd_se, period), v in self.store.items():
            p = per[prd_se]
            p["obs"] += 1
            for i, c in enumerate(codes):
                p["codes"][i].add(c)
            p["items"].add(item_id)
            p["periods"].add(period)
        for prd_se, p in per.items():
            expected = len(p["items"]) * len(p["periods"])
            for s in p["codes"].values():
                expected *= len(s)
            out["by_cycle"][prd_se] = {"present": p["obs"], "expected_full_grid": expected,
                                       "missing_share": round(1 - p["obs"] / expected, 4) if expected else None}
        return out

    def result(self) -> dict:
        dims = []
        for name in self.dim_order:
            d = self.dims[name]
            labels = list(d["codes"].values())
            label_dup = sum(1 for _, c in Counter(labels).items() if c > 1)
            dims.append({"name": name, "codes": len(d["codes"]), "code_lengths": dict(sorted(d["len"].items())),
                         "total_like": sorted({l for l in labels if TOTAL_NAME.search(l or "")})[:10],
                         "labels_reused_by_several_codes": label_dup,
                         "paren_units_in_labels": dict(Counter(m.group(1) for l in labels if (m := PAREN_UNIT.search(l or ""))).most_common(10)),
                         "examples": labels[:MAX_EXAMPLES]})
        items = list(self.items.values())
        periods = {}
        for prd_se, ps in self.periods.items():
            ps = sorted(ps)
            periods[prd_se] = {"count": len(ps), "first": ps[0] if ps else None, "last": ps[-1] if ps else None,
                               "shapes": sorted({period_shape(p) for p in ps}), "gaps": period_gaps(prd_se, ps)}
        numeric = self.kinds.get("ok", 0) + self.kinds.get("provisional", 0)
        return {
            "observations": self.n,
            "dims": dims,
            "n_dims": len(dims),
            "dims_per_observation": dict(self.dims_per_obs),
            "items": {"count": len(self.items), "examples": items[:MAX_EXAMPLES],
                      "total_like": sorted({i for i in items if TOTAL_NAME.search(i or "")})[:10],
                      "paren_units_in_names": dict(Counter(m.group(1) for i in items if (m := PAREN_UNIT.search(i or ""))).most_common(10))},
            "units": dict(self.units.most_common(30)),
            "items_with_several_units": sum(1 for s in self.units_by_item.values() if len(s) > 1),
            "periods": periods,
            "values": {"kinds": dict(self.kinds), "numeric_share": round(numeric / self.n, 6) if self.n else None,
                       "symbols": dict(self.symbols.most_common(20)), "other_text_examples": dict(self.other_text.most_common(15)),
                       "negative": self.neg, "zero": self.zero,
                       "min": None if self.vmin == math.inf else self.vmin, "max": None if self.vmax == -math.inf else self.vmax,
                       "abs_log10_histogram": dict(sorted(self.abs_log10.items())),
                       "decimals": dict(sorted(self.decimals.items())),
                       "max_decimals_by_item_top": dict(sorted(((self.items.get(k, k), v) for k, v in self.decimals_by_item.items()), key=lambda x: -x[1])[:10])},
            "approx_prompt_tokens": self.label_chars // 2,
            "key_checks": {"fit_in_memory": self.store is not None, "duplicate_keys": self.dup_keys if self.store is not None else None},
            "completeness": self.completeness(),
            "sum_checks": self.sum_checks(),
        }


# ---------------------------------------------------------------- metadata

def load_reference(repo: Path) -> dict:
    ref = {}
    meta = repo / "data" / "통계표_메타데이터.jsonl"
    if meta.exists():
        with meta.open(encoding="utf-8-sig") as f:
            for line in f:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                ref[(str(r.get("organization_id")), str(r.get("table_id")))] = {
                    "survey": r.get("survey_name"), "title": r.get("table_name"), "period_label": r.get("period"),
                    "unit_meta": r.get("unit"), "updated_at": r.get("updated_at")}
    return ref


def official_meta(path: Path) -> dict:
    if not path.exists():
        return {"found": False}
    try:
        m = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"found": True, "error": str(exc)[:200]}
    classes = []
    for c in m.get("classInfoList") or []:
        classes.append({"name": c.get("classNm"), "itmCnt": c.get("itmCnt"), "depthLvl": c.get("depthLvl"),
                        "listed": len(c.get("itmList") or [])})
    out = {"found": True, "classes": classes}
    for k in ("tblNm", "unitNm", "prdSe", "dimCo", "downloadable", "analyzable"):
        if k in m:
            out[k] = m[k]
    return out


# ---------------------------------------------------------------- driver

def table_dirs(root: Path) -> list[tuple[str, Path]]:
    out = []
    if (root / "tables").is_dir():
        # data_fix layout: every table folder in one place; API folders carry status.json
        for p in sorted((root / "tables").iterdir()):
            if p.is_dir() and not p.name.startswith("."):
                out.append(("api" if (p / "status.json").exists() else "bulk", p))
        return out
    for kind, sub in (("api", "numeric_api"), ("bulk", "numeric_raw")):
        base = root / sub
        if not base.is_dir():
            continue
        for p in sorted(base.iterdir()):
            if p.is_dir() and not p.name.startswith("."):
                out.append((kind, p))
    return out


def meta_dir(root: Path) -> Path:
    if (root / "table_metadata").is_dir():
        return root / "table_metadata"
    return root / "population" / "population" / "table_metadata"


def dir_bytes(kind: str, path: Path) -> int:
    try:
        if kind == "api":
            return sum(f.stat().st_size for f in path.iterdir() if f.name.endswith(".json.gz"))
        return sum(f.stat().st_size for f in path.iterdir() if f.suffix.lower() in (".zip", ".csv"))
    except OSError:
        return 0


def profile_table(kind: str, path: Path, root: Path, ref: dict, max_store: int) -> dict:
    key = nfc(path.name)
    org, _, tbl = key.partition("_")
    stats: dict = {}
    started = time.time()
    prof = Profiler(max_store)
    reader = read_api(path, stats) if kind == "api" else read_bulk(path, stats)
    for obs in reader:
        prof.add(obs)
    result = prof.result()
    for k, v in list(stats.items()):
        if isinstance(v, Counter):
            stats[k] = dict(v.most_common(40))
    r = ref.get((org, tbl), {})
    return {"profile_version": PROFILE_VERSION, "key": key, "org_id": org, "table_id": tbl, "source": kind,
            "survey": r.get("survey"), "title": r.get("title"), "period_label": r.get("period_label"),
            "unit_meta": r.get("unit_meta"), "status": "ok",
            "data_state": ("partial" if stats.get("api_status") in ("running", "incomplete") else "full")
            if result.get("observations") else "no_data",
            "seconds": round(time.time() - started, 1),
            "read": stats, "official_meta": official_meta(meta_dir(root) / f"{key}.json"), **result}


def run(args) -> int:
    repo = Path(args.repo).resolve()
    root = (repo / args.root).resolve() if not Path(args.root).is_absolute() else Path(args.root)
    out = (repo / args.out).resolve() if not Path(args.out).is_absolute() else Path(args.out)
    prof_dir = out / "profiles"
    prof_dir.mkdir(parents=True, exist_ok=True)
    log = Log(out / "run_log.txt")
    if not root.is_dir():
        log(f"input folder not found: {root}")
        return 2
    ref = load_reference(repo)
    tables = table_dirs(root)
    if args.only:
        wanted = set(args.only)
        tables = [t for t in tables if nfc(t[1].name) in wanted or nfc(t[1].name).partition("_")[2] in wanted]
    todo = []
    for kind, p in tables:
        target = prof_dir / f"{nfc(p.name)}.json"
        if target.exists() and not args.force:
            if args.retry_errors:
                try:
                    if json.loads(target.read_text(encoding="utf-8")).get("status") == "error":
                        todo.append((kind, p))
                except Exception:
                    todo.append((kind, p))
            continue
        todo.append((kind, p))
    skipped = len(tables) - len(todo)
    if args.limit:
        todo = todo[: args.limit]
    sizes = {str(p): dir_bytes(k, p) for k, p in todo}
    total_bytes = sum(sizes.values()) or 1
    log(f"start: {len(tables)} tables found, {skipped} already profiled, {len(todo)} to do "
        f"({total_bytes / 1e9:.2f} GB on disk). input={root} output={out}")
    done_bytes = 0
    t0 = time.time()
    ok = err = 0
    try:
        for i, (kind, p) in enumerate(todo, 1):
            key = nfc(p.name)
            target = prof_dir / f"{key}.json"
            try:
                result = profile_table(kind, p, root, ref, args.max_store)
                ok += 1
            except MemoryError:
                result = {"profile_version": PROFILE_VERSION, "key": key, "source": kind, "status": "error",
                          "error": "MemoryError (rerun with a smaller --max-store)"}
                err += 1
            except Exception as exc:
                result = {"profile_version": PROFILE_VERSION, "key": key, "source": kind, "status": "error",
                          "error": f"{type(exc).__name__}: {exc}"[:500], "traceback": traceback.format_exc()[-3000:]}
                err += 1
            if result.get("status") == "error":
                with (out / "errors.jsonl").open("a", encoding="utf-8") as f:
                    f.write(json.dumps({"at": now(), "key": key, "source": kind, "error": result["error"]}, ensure_ascii=False) + "\n")
            atomic_write_json(target, result)
            done_bytes += sizes.get(str(p), 0)
            if i % args.log_every == 0 or i == len(todo) or result.get("status") == "error":
                el = time.time() - t0
                rate = done_bytes / el if el else 0
                eta = (total_bytes - done_bytes) / rate if rate else 0
                log(f"[{i}/{len(todo)}] {key} ({kind}) {result.get('status')} obs={result.get('observations', '-')} "
                    f"| ok {ok} err {err} | {done_bytes / 1e9:.2f}/{total_bytes / 1e9:.2f} GB | elapsed {el / 60:.1f} min, ETA {eta / 60:.1f} min")
    except KeyboardInterrupt:
        log("stopped by user (Ctrl+C). Run the same command again to continue.")
        return 130
    log(f"profiling finished: ok {ok}, errors {err}, {(time.time() - t0) / 60:.1f} min")
    summarize(out, log)
    return 0


# ---------------------------------------------------------------- summaries for the notebook

def _write_csv(path: Path, rows: list[dict]) -> None:
    cols = []
    for r in rows:
        for k in r:
            if k not in cols:
                cols.append(k)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: (json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v) for k, v in r.items()})


def summarize(out: Path, log: Log | None = None) -> None:
    prof_dir = out / "profiles"
    sdir = out / "summary"
    sdir.mkdir(parents=True, exist_ok=True)
    tables, dims, units, symbols, periods, sums = [], [], [], [], [], []
    for p in sorted(prof_dir.glob("*.json")):
        try:
            r = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        base = {"key": r.get("key"), "source": r.get("source"), "survey": r.get("survey"), "title": r.get("title")}
        if r.get("status") != "ok":
            tables.append({**base, "status": r.get("status"), "error": r.get("error")})
            continue
        v = r.get("values", {})
        kinds = v.get("kinds", {})
        rd = r.get("read", {})
        cyc = r.get("periods", {})
        comp = r.get("completeness", {}).get("by_cycle", {}) if r.get("completeness", {}).get("checked") else {}
        tables.append({
            **base, "status": "ok", "data_state": r.get("data_state"), "org_id": r.get("org_id"),
            "observations": r.get("observations"), "api_status": rd.get("api_status"),
            "api_chunk_source": rd.get("api_chunk_source"), "bytes": rd.get("bytes"), "seconds": r.get("seconds"),
            "n_dims": r.get("n_dims"), "dim_names": " | ".join(d["name"] for d in r.get("dims", [])),
            "max_codes_in_dim": max((d["codes"] for d in r.get("dims", [])), default=0),
            "n_items": r.get("items", {}).get("count"), "cycles": ",".join(sorted(cyc)),
            "n_periods_max": max((c["count"] for c in cyc.values()), default=0),
            "first_period": min((c["first"] for c in cyc.values() if c.get("first")), default=None),
            "last_period": max((c["last"] for c in cyc.values() if c.get("last")), default=None),
            "period_gaps": sum((c.get("gaps", {}).get("missing") or 0) for c in cyc.values()),
            "n_units": len([u for u in r.get("units", {}) if u not in ("", "<필드 없음>")]),
            "unit_field_missing_obs": r.get("units", {}).get("<필드 없음>", 0), "unit_blank_obs": r.get("units", {}).get("", 0),
            "items_with_several_units": r.get("items_with_several_units"), "unit_meta": r.get("unit_meta"),
            "numeric_share": v.get("numeric_share"), "provisional": kinds.get("provisional", 0),
            "blank": kinds.get("blank", 0), "dash": kinds.get("dash", 0), "x_secret": kinds.get("x_secret", 0),
            "ellipsis": kinds.get("ellipsis", 0), "other_text": kinds.get("other_text", 0),
            "negative": v.get("negative"), "zero": v.get("zero"), "max_decimals": max(map(int, v.get("decimals", {}) or {0: 0})),
            "approx_prompt_tokens": r.get("approx_prompt_tokens"),
            "fit_in_memory": r.get("key_checks", {}).get("fit_in_memory"), "duplicate_keys": r.get("key_checks", {}).get("duplicate_keys"),
            "grid_missing_share_max": max((c.get("missing_share") or 0 for c in comp.values()), default=None),
            "sum_axes_checked": len(r.get("sum_checks", [])),
            "sum_exact_share": (sum(s["exact"] for s in r.get("sum_checks", [])) / sum(s["checked"] for s in r.get("sum_checks", [])))
            if r.get("sum_checks") else None,
            "meta_classes": len(r.get("official_meta", {}).get("classes", []) or []),
        })
        for d in r.get("dims", []):
            dims.append({**base, "dim": d["name"], "codes": d["codes"], "code_lengths": d["code_lengths"],
                         "n_code_lengths": len(d["code_lengths"]), "total_like": d["total_like"],
                         "labels_reused": d["labels_reused_by_several_codes"], "paren_units": d["paren_units_in_labels"]})
        for u, n in r.get("units", {}).items():
            units.append({**base, "unit": u, "observations": n})
        for s, n in v.get("symbols", {}).items():
            symbols.append({**base, "symbol": s, "observations": n})
        for c, info in cyc.items():
            periods.append({**base, "cycle": c, "count": info["count"], "first": info["first"], "last": info["last"],
                            "shapes": info["shapes"], "gaps_checked": info.get("gaps", {}).get("checked"),
                            "gaps": info.get("gaps", {}).get("missing"), "gap_examples": info.get("gaps", {}).get("missing_examples")})
        for s in r.get("sum_checks", []):
            sums.append({**base, **s})
    for name, rows in (("tables", tables), ("dims", dims), ("units", units), ("symbols", symbols), ("periods", periods), ("sum_checks", sums)):
        _write_csv(sdir / f"{name}.csv", rows)
    if log:
        log(f"summary written: {sdir} ({len(tables)} tables)")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", default=str(Path(__file__).resolve().parents[1]), help="repo root (default: parent of scripts/)")
    ap.add_argument("--root", default="data_fix", help="dump folder (relative to repo)")
    ap.add_argument("--out", default="data/eda_full", help="output folder (relative to repo)")
    ap.add_argument("--limit", type=int, default=0, help="profile at most N tables this run (0 = all)")
    ap.add_argument("--only", nargs="*", help="only these tables (folder name like 101_DT_1B040A3 or table id)")
    ap.add_argument("--force", action="store_true", help="re-profile tables that already have a profile")
    ap.add_argument("--retry-errors", action="store_true", help="re-profile only tables whose profile is an error")
    ap.add_argument("--max-store", type=int, default=1_000_000,
                    help="max observations kept for key-level checks (duplicates, grid, sums); bigger tables skip them")
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--summarize-only", action="store_true")
    args = ap.parse_args(argv)
    if args.summarize_only:
        repo = Path(args.repo).resolve()
        out = (repo / args.out).resolve()
        summarize(out, Log(out / "run_log.txt"))
        return 0
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
