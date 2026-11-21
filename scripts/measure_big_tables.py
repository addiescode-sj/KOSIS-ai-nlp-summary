"""Measure every big table of a data folder against the acceptance targets: peak memory and seconds per read (T1-31).

    python scripts/measure_big_tables.py --data <workspace>/data --out <file.json> [--min-rows 500000]
        [--tables KEY ...] [--index-dir DIR] [--fresh] [--timeout 1800] [--max-gb 3] [--max-seconds 10]

    python scripts/measure_big_tables.py --data "D:\\내 작업\\kosis\\data" --out big_tables.json --fresh
    python scripts/measure_big_tables.py --data "D:\\내 작업\\kosis\\data" --out big_tables.json --tables 101_DT_1IN1503

Which tables: the rows of <data>/tables.csv whose Parquet file has at least --min-rows rows by its footer (only the
footer is read), measured smallest first, so the biggest come last. --tables names keys to measure whatever their
size; their results replace those keys in an existing --out file and the other keys are kept.

Each table is measured in a child process of the same Python (this file run with --child), so a peak belongs to one
table, and a crash or a timeout (--timeout seconds, default 1800) of one table is recorded (exit code or "timeout")
and the run goes on. The first child:
1. loads the table with kosis_chat.adapters.parquet.load_table, as the pipeline does, and records the route it took:
   "file" (the cells stay in the file: a table of BIG_TABLE_ROWS rows or more that its scan finds eligible),
   "memory" (read whole) with the reason it is not file-backed, or "refused" with the reason; whether the scan of
   the file was a saved one or a new one, and the size of the saved scan files;
2. builds the table index as the pipeline does (schemas.resources.load_resources: clean, then prepare.index.index_of),
   and records whether a file-backed table's index was built from its descriptors ("built") or loaded from the
   saved index file beside the saved scan ("loaded"), with that file's size;
3. on a file-backed table, runs fixed requests chosen from the table's own descriptors (evenly spaced, no randomness):
   one cell, 20 cells of one period, one series, series_at for 5 keys, a summary of one item over all periods, a
   summary of one item at one period (the item and the period of a cell that exists), and by_refs for 20 refs of
   cells read before. Each request goes to a new store over the same scan, so no request is answered from the memo
   of another; each records its seconds, the cells it returned, the row groups and rows it read, and the rows it
   turned into cells (rows_converted);
4. reports its own peak memory (Windows: PeakWorkingSetSize, and PeakPagefileUsage as commit; Linux: VmHWM; macOS:
   ru_maxrss).
A second child loads each file-backed table again and gets its index: the load that reuses the saved scan and the
saved index, as every later run of the chatbot does. --fresh deletes the table's saved scan and saved index before
the first child, so the first child always scans and builds the index; without it a saved scan or index found by the
first child is used and reported as such.

The saved scans go where the chatbot puts them (<data>/index, beside tables.csv) unless --index-dir names another
folder. --fresh deletes only the measured table's saved files there (<key>.json, <key>.descriptors.parquet,
<key>.merge.parquet, <key>.table_index.json).

Output: one line per table as it finishes, a summary at the end, and --out (JSON, UTF-8) rewritten after every table,
so a stopped run keeps what it measured. Only table keys, counts, seconds, sizes and condition names are printed or
saved: no cell value, no label, no path.

Targets (--max-gb, default 3, in GiB of peak working set; --max-seconds, default 10): a table is over a target when a
child's peak is over --max-gb, or when one of these takes longer than --max-seconds: a request, the load that reuses
the saved scan (file route) or the load (memory route), the index (the reuse child's on the file route: loaded
from the saved index when it could be saved; the build on the memory route). The first load of a file-backed table (the
scan, done once per file) is reported but not judged. Exit 0 when every table was measured and met the targets, 1
otherwise (over a target, refused, crashed, timed out), 2 for bad arguments or a missing data folder or tables.csv.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import platform
import subprocess
import sys
import tempfile
import time
import traceback
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

FORMAT = 1  # of the --out file
PROBE_DESCRIPTORS = 26  # evenly spaced descriptors the requests use: 0-5 for series, 6-25 for the 20 cells
MB = 1024 * 1024
BIG_ROWS: int | None = None  # --big-rows (tests only): load_table's big_rows; None: the library's BIG_TABLE_ROWS
PROBE_ORDER = ["one cell", "20 cells of one period", "one series", "series_at 5 keys",
               "summary of one item, all periods", "summary of one item, one period", "by_refs 20"]


# ---------------------------------------------------------------- peak memory


def peak_memory() -> dict:
    """This process's peak memory in bytes: {"peak": working set / resident, "commit": private commit (Windows)}."""
    if sys.platform == "win32":
        try:
            return _windows_peak()
        except Exception:
            pass
        try:
            import psutil  # optional

            info = psutil.Process().memory_info()
            return {"peak": int(getattr(info, "peak_wset", info.rss)), "commit": getattr(info, "peak_pagefile", None),
                    "how": "psutil"}
        except Exception:
            return {"peak": None, "commit": None, "how": "none"}
    status = Path("/proc/self/status")
    if status.exists():  # Linux: VmHWM is this process's own peak; ru_maxrss keeps the peak of the image before exec
        for line in status.read_text().splitlines():
            if line.startswith("VmHWM:"):
                return {"peak": int(line.split()[1]) * 1024, "commit": None, "how": "VmHWM"}
    try:
        import resource  # not on Windows: imported here only

        maxrss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return {"peak": int(maxrss if sys.platform == "darwin" else maxrss * 1024), "commit": None, "how": "ru_maxrss"}
    except Exception:
        return {"peak": None, "commit": None, "how": "none"}


def _windows_peak() -> dict:
    import ctypes
    from ctypes import wintypes

    class Counters(ctypes.Structure):
        _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    try:
        get_info = kernel32.K32GetProcessMemoryInfo  # Windows 7 and later
    except AttributeError:
        get_info = ctypes.WinDLL("psapi", use_last_error=True).GetProcessMemoryInfo
    get_info.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
    get_info.restype = wintypes.BOOL
    counters = Counters()
    counters.cb = ctypes.sizeof(Counters)
    if not get_info(kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
        raise OSError(ctypes.get_last_error())
    return {"peak": int(counters.PeakWorkingSetSize), "commit": int(counters.PeakPagefileUsage),
            "how": "GetProcessMemoryInfo"}


def _mb(n: int | None) -> float | None:
    return None if n is None else round(n / MB, 1)


# ---------------------------------------------------------------- the library's facts


def big_rows() -> int:
    """The row count over which load_table tries the file route."""
    from kosis_chat.adapters import parquet

    return parquet.BIG_TABLE_ROWS if BIG_ROWS is None else BIG_ROWS


def saved_scan_files(path: Path) -> list[Path]:
    """The saved-scan files of this table in the folder load_table uses (parquet_store.INDEX_DIR, else <data>/index)."""
    from kosis_chat.adapters import parquet, parquet_store

    folder = parquet.index_folder(path)
    if folder is None:
        return []
    try:
        return list(parquet_store._index_files(folder, path.stem))
    except AttributeError:  # same names as parquet_store._index_files at f73a010
        return [folder / f"{path.stem}.json", folder / f"{path.stem}.descriptors.parquet"]


def saved_index_file(path: Path) -> Path | None:
    """The saved table index of this table in that folder (beside the saved scan), None when there is no folder."""
    from kosis_chat.adapters import parquet, parquet_store

    folder = parquet.index_folder(path)
    if folder is None:
        return None
    try:
        return parquet_store._table_index_file(folder, path.stem)
    except AttributeError:  # a library without the saved index
        return folder / f"{path.stem}.table_index.json"


def why_not_file_backed(path: Path, error: Exception | None = None) -> str:
    """The condition that keeps this table off the file route: a parquet_store.ELIGIBILITY key, as
    kosis_chat.adapters.parquet.why_not_file_backed gives it (from the scan load_table has just saved, so no new scan
    when the index folder is writable), or one of: "small" (under the big-table row count, never scanned),
    "row_limit" (refused over ROW_LIMIT with no condition found), "error" (the file or its catalog row cannot be read,
    or another load error). error: the SchemaError of a refused load."""
    from kosis_chat.adapters import parquet
    from kosis_chat.prepare.schemas import SchemaError

    rows = parquet.footer_rows(path)
    if error is None and rows is not None and rows < big_rows():
        return "small"
    try:
        found = parquet.why_not_file_backed(path)
    except SchemaError:
        return "error"
    if found is not None:
        return found[0]
    return "row_limit" if error is not None and "row limit" in str(error) else "error"


# ---------------------------------------------------------------- the child


def _stamp(path: Path) -> tuple | None:
    try:
        st = path.stat()
    except OSError:
        return None
    return st.st_mtime_ns, st.st_size


def _frames(exc: BaseException) -> list[str]:
    """Where an error happened: file name, line and function of each frame (no message: it may hold values)."""
    return [f"{Path(f.filename).name}:{f.lineno}:{f.name}" for f in traceback.extract_tb(exc.__traceback__)][-8:]


class Child:
    """One measurement in this process; the result file is rewritten after every step, so a crash keeps the steps."""

    def __init__(self, path: Path, result: Path) -> None:
        self.path, self.result_path = path, result
        self.out: dict = {"stage": "start"}

    def save(self, stage: str) -> None:
        self.out["stage"] = stage
        memory = peak_memory()
        self.out["peak_mb"] = _mb(memory["peak"])
        self.out["peak_commit_mb"] = _mb(memory["commit"])
        self.out["memory_how"] = memory["how"]
        self.result_path.write_text(json.dumps(self.out), encoding="utf-8")

    def failed(self, stage: str, exc: BaseException) -> None:
        self.out["error_type"] = type(exc).__name__
        self.out["error_frames"] = _frames(exc)
        self.save(stage)

    def run(self, mode: str, fresh: bool) -> int:
        from kosis_chat.adapters import parquet
        from kosis_chat.prepare.schemas import SchemaError

        files = saved_scan_files(self.path)
        index_file = saved_index_file(self.path)
        if fresh:
            for f in files + ([index_file] if index_file is not None else []):  # this table's saved scan and index
                f.unlink(missing_ok=True)
        before = [_stamp(f) for f in files]
        self.out["mode"] = mode
        self.save("load")
        started = time.perf_counter()
        try:
            table = parquet.load_table(self.path, big_rows=BIG_ROWS)
        except SchemaError as exc:
            self.out["load_s"] = round(time.perf_counter() - started, 3)
            self.out["route"] = "refused"
            self.out["condition"] = why_not_file_backed(self.path, exc)
            self.out["problems"] = len(exc.problems)
            self._scan_facts(files, before)
            self.save("done")
            return 0
        except BaseException as exc:  # MemoryError, a broken file: recorded, the parent goes on
            self.out["load_s"] = round(time.perf_counter() - started, 3)
            self.failed("load", exc)
            return 3
        self.out["load_s"] = round(time.perf_counter() - started, 3)
        self._scan_facts(files, before)
        if table.file_backed:
            self.out["route"] = "file"
        else:
            self.out["route"] = "memory"
            self.out["condition"] = why_not_file_backed(self.path)
        self.out["peak_after_load_mb"] = _mb(peak_memory()["peak"])
        self.save("index")

        try:
            from kosis_chat.schemas.resources import load_resources

            started = time.perf_counter()
            resources = load_resources(table)  # the pipeline's step: clean, then index_of
            self.out["index_s"] = round(time.perf_counter() - started, 3)
            index = resources.index
            self.out["periods"] = len(index.periods)
            self.out["axes"] = len(index.axes)
            if table.file_backed:  # "loaded" only when the saved index was used, not built again
                state = getattr(table.store, "index_state", None)
                self.out["index_from"] = "loaded" if state == "loaded" else "built"
                self.out["index_state"] = state  # the store's: "loaded", "saved", "not saved" or None
                stamp = _stamp(index_file) if index_file is not None else None
                self.out["saved_index_bytes"] = stamp[1] if stamp else None
        except BaseException as exc:
            self.failed("index", exc)
            return 3
        self.out["peak_after_index_mb"] = _mb(peak_memory()["peak"])
        if mode == "measure" and table.file_backed:
            self.out["descriptors"] = table.store.facts.descriptor_count
            self.save("probes")
            try:
                self.out["probes"] = probes(resources.table, index)
            except BaseException as exc:
                self.failed("probes", exc)
                return 3
        self.save("done")
        return 0

    def _scan_facts(self, files: list[Path], before: list) -> None:
        """Whether load_table used a saved scan ("saved"), made a new one ("fresh", "fresh_not_saved"), or none."""
        from kosis_chat.adapters import parquet

        after = [_stamp(f) for f in files]
        rows = parquet.footer_rows(self.path)
        if rows is not None and rows < big_rows():
            self.out["scan"] = "none"
        elif files and before[0] is not None and before[0] == after[0]:
            self.out["scan"] = "saved"
        elif files and after[0] is not None:
            self.out["scan"] = "fresh"
        else:
            self.out["scan"] = "fresh_not_saved"
        self.out["saved_scan_bytes"] = sum(s[1] for s in after if s is not None) if any(after) else None


def probes(table, index) -> list[dict]:
    """The fixed requests on a file-backed table (see the module docstring), each on a new store over the same scan."""
    from kosis_chat.prepare.store import ITEM_AXIS, dim_key

    store = table.store

    def new_store():
        try:
            return type(store)(store.path, store.meta, store.facts)
        except Exception:  # a store that cannot be made again: the memo may answer some requests
            return store

    out: list[dict] = []

    def timed(name: str, call, count) -> object:
        s = new_store()
        before = {k: getattr(s, k, None) for k in ("reads", "row_groups_read", "rows_read", "rows_masked", "rows_converted")}
        started = time.perf_counter()
        got = call(s)
        record = {"name": name, "s": round(time.perf_counter() - started, 3), "cells": count(got)}
        for k, v in before.items():
            if v is not None:
                record[k] = getattr(s, k) - v
        out.append(record)
        return got

    picks = descriptor_cells(store, PROBE_DESCRIPTORS)
    keys = [(c.item_code, dim_key(c)) for c in picks]
    cells_in = lambda got: sum(len(v) for v in got.values())  # noqa: E731

    five = timed("series_at 5 keys", lambda s: s.series_at(keys[:5]), cells_in)
    one = timed("one series", lambda s: s.series(*keys[5]), len) if len(keys) > 5 else ()
    serieses = [five.get(k, ()) for k in keys[:5]] + [one]
    # the period most of these six series have (the latest of those on a tie), and the first key that has it
    counts = Counter(c.period_raw for cells in serieses for c in cells)
    period = max(counts, key=lambda p: (counts[p], p)) if counts else index.periods[-1]
    at = next((k for k, cells in zip(keys, serieses) if any(c.period_raw == period for c in cells)), keys[0])
    found_one = timed("one cell", lambda s: s.cell((at[0], at[1], period)), lambda c: int(c is not None))
    twenty = timed("20 cells of one period", lambda s: s.cells_at([(i, d, period) for i, d in keys[6:26]]), len)
    timed("summary of one item, all periods", lambda s: s.summary({ITEM_AXIS: {at[0]}}, index.periods),
          lambda r: r.count)
    timed("summary of one item, one period", lambda s: s.summary({ITEM_AXIS: {at[0]}}, [period]), lambda r: r.count)
    refs: list[str] = []
    for cell in [found_one, *twenty.values(), *(c for cells in serieses for c in cells)]:
        if cell is not None and cell.source_ref not in refs:
            refs.append(cell.source_ref)
        if len(refs) == 20:
            break
    timed("by_refs 20", lambda s: s.by_refs(refs), len)
    return sorted(out, key=lambda p: PROBE_ORDER.index(p["name"]))


def descriptor_cells(store, count: int) -> list:
    """`count` descriptor cells (one per distinct cell but its period and value) evenly spaced over the table, by the
    store's own conversion: index_cells of a store whose scan holds only those descriptors. They are taken from the
    descriptor table the scan holds, else from the saved descriptors file, reading only the row groups that hold them
    (a table may have millions of descriptors)."""
    import dataclasses
    import itertools

    facts = store.facts
    n = facts.descriptor_count
    positions = sorted({i * n // count for i in range(count)}) if n else []
    try:
        picked = _descriptor_rows(facts, positions)
        small = dataclasses.replace(facts, descriptors=picked, descriptor_count=picked.num_rows, descriptors_file=None)
        return list(type(store)(store.path, store.meta, small).index_cells())
    except Exception:  # another Facts layout: the first descriptors instead of evenly spaced ones
        return list(itertools.islice(store.index_cells(), count))


def _descriptor_rows(facts, positions: list[int]):
    """The rows at `positions` (sorted) of the scan's descriptor table, as a pyarrow Table."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    if facts.descriptors is not None:
        return facts.descriptors.take(positions)
    parts, start = [], 0
    with pq.ParquetFile(facts.descriptors_file) as pf:
        for group in range(pf.num_row_groups):
            size = pf.metadata.row_group(group).num_rows
            local = [p - start for p in positions if start <= p < start + size]
            if local:
                parts.append(pf.read_row_group(group).take(local))
            start += size
    return pa.concat_tables(parts)


def child_main(args: argparse.Namespace) -> int:
    global BIG_ROWS
    BIG_ROWS = args.big_rows
    if args.index_dir:
        from kosis_chat.adapters import parquet_store

        parquet_store.INDEX_DIR = Path(args.index_dir)
    child = Child(Path(args.path), Path(args.result))
    try:
        return child.run(args.child, args.fresh)
    except BaseException as exc:
        child.failed(child.out.get("stage", "start"), exc)
        return 3


# ---------------------------------------------------------------- the parent


def list_tables(data: Path, min_rows: int, keys: list[str] | None) -> tuple[list[dict], list[str], list[str], int]:
    """(tables to measure, smallest first: key, rows, file_mb, path; keys whose file is missing; keys whose file
    exists but whose footer cannot be read; tables.csv rows)."""
    import pyarrow.parquet as pq

    with (data / "tables.csv").open(encoding="utf-8-sig", newline="") as f:
        rows = [r for r in csv.DictReader(f) if r.get("key")]
    by_key = {r["key"]: r for r in rows}
    wanted = keys if keys else list(by_key)
    out, missing, unreadable = [], [], []
    for key in wanted:
        row = by_key[key]
        path = data / row["parquet"] if row.get("parquet") else data / "values" / f"{key}.parquet"
        if not path.is_file():
            missing.append(key)
            continue
        try:
            meta = pq.read_metadata(path)
        except Exception:
            unreadable.append(key)
            continue
        if keys or meta.num_rows >= min_rows:
            out.append({"key": key, "rows": meta.num_rows, "file_mb": _mb(path.stat().st_size), "path": path})
    out.sort(key=lambda t: (t["rows"], t["key"]))
    return out, missing, unreadable, len(rows)


def run_child(mode: str, table: dict, args: argparse.Namespace, fresh: bool) -> dict:
    """Run one child; its result file, plus "exit" (the exit code or "timeout") and "wall_s"."""
    handle, name = tempfile.mkstemp(prefix="measure_big_tables.", suffix=".json")
    os.close(handle)
    result = Path(name)
    command = [sys.executable, str(Path(__file__).resolve()), "--child", mode, "--path", str(table["path"]),
               "--result", str(result)]
    if args.index_dir:
        command += ["--index-dir", str(Path(args.index_dir).resolve())]
    if args.big_rows is not None:
        command += ["--big-rows", str(args.big_rows)]
    if fresh:
        command.append("--fresh")
    started = time.perf_counter()
    try:
        # the child's own output is dropped: an error text may quote cell values; the result file names the frames
        code: int | str = subprocess.run(command, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                         stderr=subprocess.DEVNULL, timeout=args.timeout).returncode
    except subprocess.TimeoutExpired:
        code = "timeout"
    wall = round(time.perf_counter() - started, 3)
    try:
        out = json.loads(result.read_text(encoding="utf-8") or "{}")
    except (OSError, ValueError):
        out = {}
    finally:
        try:
            result.unlink()
        except OSError:
            pass
    return out | {"exit": code, "wall_s": wall}


def measure(table: dict, args: argparse.Namespace) -> dict:
    first = run_child("measure", table, args, args.fresh)
    record = {"key": table["key"], "rows": table["rows"], "file_mb": table["file_mb"], "measured": _now(),
              "route": first.get("route"), "condition": first.get("condition"), "first": first, "reuse": None}
    if first.get("route") == "file" and _finished(first):
        record["reuse"] = run_child("reload", table, args, False)
    record["over"] = over_targets(record, args)
    record["ok"] = ok(record) and not record["over"]
    return record


def _finished(child: dict) -> bool:
    return child.get("exit") == 0 and child.get("stage") == "done"


def ok(record: dict) -> bool:
    """Every child finished and the table was read (not refused)."""
    children = [c for c in (record["first"], record["reuse"]) if c is not None]
    return record["route"] in ("file", "memory") and all(_finished(c) for c in children)


def over_targets(record: dict, args: argparse.Namespace) -> list[dict]:
    """What is over --max-gb or --max-seconds (the module docstring lists what is judged)."""
    out = []
    limit_mb = args.max_gb * 1024
    for which in ("first", "reuse"):
        child = record[which]
        if child and child.get("peak_mb") is not None and child["peak_mb"] > limit_mb:
            out.append({"what": f"peak ({which} load)", "value": child["peak_mb"], "limit": limit_mb, "unit": "MB"})
    first, reuse = record["first"], record["reuse"]
    timed = []
    if record["route"] == "memory":
        timed += [("load", first.get("load_s")), ("index", first.get("index_s"))]
    if reuse:
        timed += [("load with the saved scan", reuse.get("load_s")), ("index", reuse.get("index_s"))]
    timed += [(f"request: {p['name']}", p.get("s")) for p in first.get("probes", [])]
    for what, seconds in timed:
        if seconds is not None and seconds > args.max_seconds:
            out.append({"what": what, "value": seconds, "limit": args.max_seconds, "unit": "s"})
    return out


def _now() -> str:
    return dt.datetime.now().isoformat(timespec="seconds")


def _s(x) -> str:
    return "-" if x is None else f"{x:.1f}"


def line(record: dict, done: int, total: int) -> str:
    first, reuse = record["first"], record["reuse"]
    route = record["route"] or "-"
    if record["condition"]:
        route += f" [{record['condition']}]"
    if not _finished(first):
        route += f" FAILED at {first.get('stage', 'start')} (exit {first.get('exit')})"
    probes = first.get("probes", [])
    slowest = max((p["s"] for p in probes), default=None)
    text = (f"[{done}/{total}] {record['key']}  {record['rows']:,} rows  {route}  load {_s(first.get('load_s'))} s"
            f" (scan {first.get('scan', '-')})  index {_s(first.get('index_s'))} s ({first.get('index_from', '-')})"
            f"  slowest request {_s(slowest)} s"
            f"  peak {first.get('peak_mb', '-')} MB")
    if reuse:
        if not _finished(reuse):
            text += f"  | reuse failed ({reuse.get('stage')}, exit {reuse.get('exit')})"
        else:
            text += (f"  | reuse: load {_s(reuse.get('load_s'))} s (scan {reuse.get('scan')})"
                     f" index {_s(reuse.get('index_s'))} s ({reuse.get('index_from', '-')})"
                     f" peak {reuse.get('peak_mb')} MB")
    if record["over"]:
        text += f"  OVER: {', '.join(o['what'] for o in record['over'])}"
    return text


def summarize(tables: dict) -> dict:
    records = list(tables.values())
    routes = Counter(r["route"] or "failed" for r in records)
    conditions = Counter(r["condition"] for r in records if r["route"] in ("memory", "refused"))
    peaks = sorted(((max(c.get("peak_mb") or 0 for c in (r["first"], r["reuse"]) if c), r["key"]) for r in records),
                   reverse=True)
    requests = sorted(((p["s"], r["key"], p["name"]) for r in records for p in r["first"].get("probes", [])),
                      reverse=True)
    return {
        "tables": len(records),
        "routes": dict(routes),
        "ineligible_by_condition": dict(conditions),
        "highest_peaks_mb": [{"key": k, "peak_mb": v} for v, k in peaks[:5]],
        "slowest_requests": [{"key": k, "request": n, "s": s} for s, k, n in requests[:5]],
        "over_targets": {r["key"]: r["over"] for r in records if r["over"]},
        "failed": sorted(r["key"] for r in records if not ok(r)),
    }


def print_summary(summary: dict, args: argparse.Namespace) -> None:
    print(f"\n{summary['tables']} tables measured")
    print("routes: " + ", ".join(f"{k} {v}" for k, v in sorted(summary["routes"].items())))
    if summary["ineligible_by_condition"]:
        by_count = sorted(summary["ineligible_by_condition"].items(), key=lambda kv: (-kv[1], kv[0]))
        print("not file-backed, by condition: " + ", ".join(f"{k} {v}" for k, v in by_count))
    print("highest peaks: " + ", ".join(f"{p['key']} {p['peak_mb']} MB" for p in summary["highest_peaks_mb"]))
    print("slowest requests:")
    for p in summary["slowest_requests"]:
        print(f"  {p['s']:.2f} s  {p['key']}  {p['request']}")
    over = summary["over_targets"]
    print(f"over the targets ({args.max_gb:g} GB, {args.max_seconds:g} s): {len(over) or 'none'}")
    for key, items in over.items():
        print(f"  {key}: " + ", ".join(f"{o['what']} {o['value']} {o['unit']}" for o in items))
    if summary["failed"]:
        print(f"not measured to the end or refused: {', '.join(summary['failed'])}")


def write_out(path: Path, report: dict) -> None:
    """Write through a temporary file and rename it over --out; retried, as OneDrive or a virus scanner may hold the
    file for a moment on Windows."""
    text = json.dumps(report, ensure_ascii=False, indent=1)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(text, encoding="utf-8")
    for attempt in range(10):
        try:
            os.replace(temp, path)
            return
        except PermissionError:
            time.sleep(0.5 * (attempt + 1))
    path.write_text(text, encoding="utf-8")
    temp.unlink(missing_ok=True)


def parse(argv: list[str] | None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Measure peak memory and seconds per read of the big tables.")
    ap.add_argument("--data", help="the data folder (tables.csv, values/)")
    ap.add_argument("--out", help="the JSON file to write (UTF-8)")
    ap.add_argument("--min-rows", type=int, default=500_000, help="the footer row count a table needs (default 500000)")
    ap.add_argument("--tables", nargs="+", metavar="KEY", help="measure these keys and merge them into --out")
    ap.add_argument("--index-dir", help="where the saved scans go (default: <data>/index, as the chatbot)")
    ap.add_argument("--fresh", action="store_true",
                    help="delete each table's saved scan (<key>.json, <key>.descriptors.parquet, <key>.merge.parquet) "
                         "and saved index (<key>.table_index.json) first")
    ap.add_argument("--timeout", type=float, default=1800, help="seconds one child may take (default 1800)")
    ap.add_argument("--max-gb", type=float, default=3.0, help="peak memory target in GiB (default 3)")
    ap.add_argument("--max-seconds", type=float, default=10.0, help="seconds target per read (default 10)")
    ap.add_argument("--child", choices=["measure", "reload"], help=argparse.SUPPRESS)
    ap.add_argument("--path", help=argparse.SUPPRESS)
    ap.add_argument("--result", help=argparse.SUPPRESS)
    ap.add_argument("--big-rows", type=int, help=argparse.SUPPRESS)  # tests: small tables down the big-table path
    args = ap.parse_args(argv)
    if not args.child and (not args.data or not args.out):
        ap.error("--data and --out are required")
    if args.child and (not args.path or not args.result):
        ap.error("--child needs --path and --result")
    if args.min_rows < 0 or args.timeout <= 0 or args.max_gb <= 0 or args.max_seconds <= 0:
        ap.error("--min-rows, --timeout, --max-gb and --max-seconds must be positive")
    return args


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")  # cmd's cp949 console
    args = parse(argv)
    if args.child:
        return child_main(args)
    data = Path(args.data)
    if not (data / "tables.csv").is_file():
        print("no tables.csv in the --data folder", file=sys.stderr)
        return 2
    with (data / "tables.csv").open(encoding="utf-8-sig", newline="") as f:
        known = {r.get("key") for r in csv.DictReader(f)}
    unknown = [k for k in args.tables or [] if k not in known]
    if unknown:
        print(f"not in tables.csv: {', '.join(unknown)}", file=sys.stderr)
        return 2
    out = Path(args.out)
    report = {"format": FORMAT, "started": _now(), "python": platform.python_version(), "platform": sys.platform,
              "settings": {"min_rows": args.min_rows, "max_gb": args.max_gb, "max_seconds": args.max_seconds,
                           "timeout": args.timeout, "fresh": args.fresh, "index_dir_given": bool(args.index_dir)},
              "tables": {}}
    if args.tables and out.is_file():
        try:
            old = json.loads(out.read_text(encoding="utf-8"))
            if old.get("format") == FORMAT:
                report["tables"] = old.get("tables", {})
                report["started"] = old.get("started", report["started"])
        except (OSError, ValueError):
            pass

    from kosis_chat.adapters import parquet
    from kosis_chat.prepare.convert import ROW_LIMIT

    big = parquet.BIG_TABLE_ROWS if args.big_rows is None else args.big_rows
    report["settings"] |= {"big_table_rows": big, "row_limit": ROW_LIMIT}
    tables, missing, unreadable, catalog_rows = list_tables(data, args.min_rows, args.tables)
    which = f"{len(args.tables)} keys asked" if args.tables else f"at least {args.min_rows:,} rows"
    print(f"{len(tables)} of {catalog_rows} tables to measure ({which}), biggest last", flush=True)
    for name, keys in (("missing", missing), ("not readable as Parquet", unreadable)):
        if keys:
            more = f" and {len(keys) - 5} more" if len(keys) > 5 else ""
            print(f"{len(keys)} files {name}: {', '.join(keys[:5])}{more}", flush=True)
    report["missing"], report["unreadable"] = missing, unreadable
    for done, table in enumerate(tables, 1):
        record = measure(table, args)
        report["tables"][table["key"]] = record
        report["updated"] = _now()
        report["summary"] = summarize(report["tables"])
        write_out(out, report)
        print(line(record, done, len(tables)), flush=True)
    report["summary"] = summarize(report["tables"])
    report["updated"] = _now()
    write_out(out, report)
    print_summary(report["summary"], args)
    return 0 if report["tables"] and all(r["ok"] for r in report["tables"].values()) and not unreadable else 1


if __name__ == "__main__":
    sys.exit(main())
