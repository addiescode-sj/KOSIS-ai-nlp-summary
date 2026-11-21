"""The saved index of a file-backed table (T1-31 S3): prepare.index.index_of builds the TableIndex of a file-backed
table once, its store saves it beside the saved scan, and a later process loads it instead of looping over every
descriptor. The loaded index equals the built one in every field and dict order; it is used only for the same file
bytes (sha256) and the same code (prepare.index.INDEX_CODE and the store's own code); a corrupt file or one that
cannot be written is harmless; a memory table never saves anything.

Synthetic tables only (made-up values), from the generators of tests/test_parquet_store.py."""

from __future__ import annotations

import json
import logging
import random
import sys
from pathlib import Path

import pytest

from kosis_chat.adapters import parquet, parquet_store
from kosis_chat.prepare import index as index_mod
from kosis_chat.prepare import store as store_mod
from kosis_chat.prepare.index import index_of
from kosis_chat.prepare.schemas import SchemaError, TableInput
from kosis_chat.schemas.resources import load_resources
from tests.test_parquet_store import (_fuzz_rows, _relabel, _replace_keeping_mtime, _row, compare_stores, cycle_rows,
                                      index_view, rich_rows, write)

KEY = "101_FAKE_RICH"


@pytest.fixture(autouse=True)
def index_dir(tmp_path, monkeypatch):
    folder = tmp_path / "saved_index"
    monkeypatch.setattr(parquet_store, "INDEX_DIR", folder)
    return folder


@pytest.fixture
def counted(monkeypatch):
    """Calls of ParquetStore.index_cells (the loop over every descriptor), counted."""
    calls = []
    real = parquet_store.ParquetStore.index_cells

    def index_cells(self):
        calls.append(self)
        return real(self)

    monkeypatch.setattr(parquet_store.ParquetStore, "index_cells", index_cells)
    return calls


def saved_file(folder: Path, key: str = KEY) -> Path:
    return folder / f"{key}.table_index.json"


def full_view(index) -> tuple:
    """index_view plus what it leaves out: the sets in measures as sets of tuples, and the Axis objects' own names."""
    return (index_view(index), {key: found for key, found in index.measures.items()},
            [(key, axis.name, type(axis).__name__) for key, axis in index.axes.items()])


def built_and_loaded(path: Path, counted: list) -> tuple:
    """(the index built by a first load, the index loaded by a second load with a new store)."""
    first = parquet.load_table(path, big_rows=0)
    built = index_of(first)
    assert first.store.index_state == "saved" and len(counted) == 1
    again = parquet.load_table(path, big_rows=0)
    assert again.store is not first.store
    loaded = index_of(again)
    assert again.store.index_state == "loaded" and len(counted) == 1  # not built again
    assert loaded.table is again and built.table is first
    return built, loaded


def tables() -> list[tuple[str, list[dict], int]]:
    return [("rich0", rich_rows(0), 4), ("rich1", rich_rows(1), 1),
            ("rich_no_scenario", rich_rows(2, scenario=False), 1000), ("cycles", cycle_rows(True), 3),
            ("cycles_apart", cycle_rows(False), 2)]


@pytest.mark.parametrize("name, rows, group_rows", tables(), ids=[t[0] for t in tables()])
def test_the_loaded_index_equals_the_built_one_in_every_field_and_order(tmp_path, counted, name, rows, group_rows):
    path = write(tmp_path / "data", "FAKE_RICH", rows, group_rows)
    built, loaded = built_and_loaded(path, counted)
    assert full_view(loaded) == full_view(built)
    assert full_view(loaded) == full_view(index_of(parquet.load_table(path, big_rows=10**9)))  # the memory route's
    for key, found in loaded.measures.items():
        assert isinstance(found, set) and all(isinstance(m, tuple) and len(m) == 2 for m in found), key
    assert [list(a.labels) for a in loaded.axes.values()] == [list(a.labels) for a in built.axes.values()]
    assert [list(a.parents) for a in loaded.axes.values()] == [list(a.parents) for a in built.axes.values()]


def test_the_loaded_index_equals_the_built_one_on_fuzzed_tables(tmp_path_factory, counted):
    same = 0
    for seed in range(40):
        rng = random.Random(seed)
        path = write(tmp_path_factory.mktemp("fuzz"), "FAKE_FUZZ", _fuzz_rows(rng), rng.choice([1, 2, 5, 100]))
        try:
            first = parquet.load_table(path, big_rows=0)
        except SchemaError:
            continue
        if not first.file_backed:
            continue
        built = index_of(first)
        loaded = index_of(parquet.load_table(path, big_rows=0))
        assert full_view(loaded) == full_view(built), seed
        same += 1
    assert same >= 10


def test_the_pipeline_step_loads_the_saved_index_in_a_new_store(tmp_path, counted):
    path = write(tmp_path / "data", "FAKE_RICH", rich_rows(), 4)
    first = load_resources(parquet.load_table(path, big_rows=0))
    assert len(counted) == 1
    second = load_resources(parquet.load_table(path, big_rows=0))
    assert len(counted) == 1 and second.table.store.index_state == "loaded"
    assert full_view(second.index) == full_view(first.index) and second.index.table is second.table
    index_of(second.table)  # kept with the table: not loaded again either
    assert len(counted) == 1


def test_the_saved_file_is_beside_the_scan_and_names_nothing_local(tmp_path, index_dir):
    path = write(tmp_path / "data", "FAKE_RICH", rich_rows(), 4)
    index_of(parquet.load_table(path, big_rows=0))
    assert sorted(p.name for p in index_dir.iterdir()) == [f"{KEY}.descriptors.parquet", f"{KEY}.json",
                                                           f"{KEY}.table_index.json"]
    text = saved_file(index_dir).read_text(encoding="utf-8")
    saved = json.loads(text)
    assert str(tmp_path) not in text and saved["sha256"] == parquet.sha256_of(path)
    assert saved["head"] == {"scan": parquet_store.SCAN_CODE, "adapters": parquet_store.ADAPTERS_CODE,
                             "index": index_mod.INDEX_CODE}


# ---------------------------------------------------------------- validity


def test_other_file_bytes_rebuild_the_index(tmp_path, index_dir, counted):
    path = write(tmp_path / "data", "FAKE_RICH", rich_rows(), 4)
    index_of(parquet.load_table(path, big_rows=0))
    old = saved_file(index_dir).read_bytes()
    _replace_keeping_mtime(path, _relabel(rich_rows(), "남성"))  # other bytes, same date
    changed = parquet.load_table(path, big_rows=0)
    assert index_of(changed).axes["성별"].labels["1"] == "남성" and len(counted) == 2
    memory = full_view(index_of(parquet.load_table(path, big_rows=10**9)))
    assert full_view(index_of(changed)) == memory
    saved_file(index_dir).write_bytes(old)  # the other file's index beside this file's scan
    table = parquet.load_table(path, big_rows=0)
    assert index_of(table).axes["성별"].labels["1"] == "남성" and full_view(index_of(table)) == memory
    assert len(counted) == 3 and table.store.index_state == "saved"


@pytest.mark.parametrize("where", ["index", "adapters", "scan"])
def test_another_code_rebuilds_the_index(tmp_path, monkeypatch, counted, where):
    path = write(tmp_path / "data", "FAKE_RICH", rich_rows(), 4)
    built = index_of(parquet.load_table(path, big_rows=0))
    module, name = {"index": (index_mod, "INDEX_CODE"), "adapters": (parquet_store, "ADAPTERS_CODE"),
                    "scan": (parquet_store, "SCAN_CODE")}[where]
    monkeypatch.setattr(module, name, "another version of the code")
    table = parquet.load_table(path, big_rows=0)
    assert full_view(index_of(table)) == full_view(built)
    assert len(counted) == 2 and table.store.index_state == "saved"
    assert full_view(index_of(parquet.load_table(path, big_rows=0))) == full_view(built)  # saved for the new code
    assert len(counted) == 2


def test_the_code_hash_covers_the_folders_the_index_depends_on(tmp_path):
    chat = Path(index_mod.__file__).resolve().parents[1]
    folders = index_mod._code_folders()
    assert folders[:2] == [chat / "prepare", chat / "shared"] and (folders[2] / "units.py").exists()
    assert (folders[2] / "schemas.py").exists() and Path(index_mod.__file__).resolve().parent in folders
    copy = tmp_path / "prepare"
    (copy / "sub").mkdir(parents=True)
    (copy / "a.py").write_bytes(b"x = 1\r\ny = 2\r\n")
    (copy / "sub" / "b.py").write_bytes(b"z = 3\n")
    (copy / "notes.txt").write_bytes(b"not code")
    first = index_mod._source_hash([copy])
    (copy / "a.py").write_bytes(b"x = 1\ny = 2\n")  # line endings only: the same code
    (copy / "notes.txt").write_bytes(b"other")
    assert index_mod._source_hash([copy]) == first
    (copy / "sub" / "b.py").write_bytes(b"z = 4\n")  # a module in a subfolder changed
    assert index_mod._source_hash([copy]) != first
    assert index_mod._source_hash(folders) == index_mod.INDEX_CODE
    adapters = Path(parquet_store.__file__).resolve().parent
    assert {p.name for p in adapters.glob("*.py")} >= {"parquet.py", "parquet_store.py"}


def test_the_code_hash_reads_file_names_and_every_adapter_module(tmp_path):
    for name in ("one", "two"):
        (tmp_path / name).mkdir()
    (tmp_path / "one" / "a.py").write_bytes(b"x = 1\n")
    (tmp_path / "two" / "b.py").write_bytes(b"x = 1\n")
    (tmp_path / "two" / "one").mkdir()
    assert index_mod._source_hash([tmp_path / "one"]) != index_mod._source_hash([tmp_path / "two"])  # names count
    adapters = Path(parquet_store.__file__).resolve().parent
    assert [name for name, _ in parquet_store._adapter_sources()] == sorted(p.name for p in adapters.glob("*.py"))
    assert parquet_store.ADAPTERS_CODE == store_mod.source_hash(parquet_store._adapter_sources())
    assert parquet_store.SCAN_CODE == parquet_store._code_hash() is not None


@pytest.fixture
def untrusted(monkeypatch, caplog):
    """source_hash's warning, as if not yet logged in this process."""
    monkeypatch.setattr(store_mod, "_untrusted_warned", False)
    caplog.set_level(logging.WARNING, logger=store_mod.__name__)
    return lambda: [r for r in caplog.records if r.name == store_mod.__name__]


def test_a_folder_named_like_a_module_is_skipped(tmp_path, untrusted):
    folder = tmp_path / "prepare"
    folder.mkdir()
    (folder / "a.py").write_bytes(b"x = 1\n")
    first = index_mod._source_hash([folder])
    (folder / "b.py").mkdir()
    (folder / "b.py" / "c.txt").write_bytes(b"not code")
    assert index_mod._source_hash([folder]) == first is not None and not untrusted()


@pytest.mark.skipif(sys.platform == "win32", reason="symbolic links need a privilege on Windows")
def test_a_dangling_link_is_skipped(tmp_path, untrusted):
    folder = tmp_path / "prepare"
    folder.mkdir()
    (folder / "a.py").write_bytes(b"x = 1\n")
    first = index_mod._source_hash([folder])
    (folder / "gone.py").symlink_to(tmp_path / "nowhere.py")
    assert index_mod._source_hash([folder]) == first is not None and not untrusted()


def test_a_source_that_cannot_be_read_gives_no_code_key_and_one_warning(tmp_path, monkeypatch, untrusted):
    folder = tmp_path / "prepare"
    folder.mkdir()
    for name in ("a.py", "b.py"):
        (folder / name).write_bytes(b"x = 1\n")
    real = Path.read_bytes

    def read_bytes(self):
        if self.name == "b.py":
            raise PermissionError(13, "locked by another process", str(self))
        return real(self)

    monkeypatch.setattr(Path, "read_bytes", read_bytes)
    assert index_mod._source_hash([folder]) is None
    assert index_mod._source_hash([folder]) is None and parquet_store._adapters_code() is not None
    assert len(untrusted()) == 1 and "locked" in untrusted()[0].getMessage()


def test_no_source_at_all_gives_no_code_key(tmp_path, monkeypatch, untrusted):
    empty = tmp_path / "prepare"
    empty.mkdir()
    (empty / "a.pyc").write_bytes(b"\x00compiled")
    assert index_mod._source_hash([empty]) is None and index_mod._source_hash([tmp_path / "missing"]) is None
    assert store_mod.source_hash([("a.py", empty / "a.pyc")], required=2) is None
    monkeypatch.setattr(parquet_store, "__file__", str(tmp_path / "app.zip" / "parquet_store.py"))
    assert parquet_store._code_hash() is None and parquet_store._adapters_code() is None
    assert len(untrusted()) == 1


@pytest.mark.parametrize("where", ["index", "adapters", "scan"])
def test_without_a_code_key_nothing_is_loaded_or_saved(tmp_path, index_dir, monkeypatch, counted, where):
    path = write(tmp_path / "data", "FAKE_RICH", rich_rows(), 4)
    expected = full_view(index_of(parquet.load_table(path, big_rows=10**9)))
    index_of(parquet.load_table(path, big_rows=0))  # saved with the real keys
    before = {p.name: p.read_bytes() for p in index_dir.iterdir()}
    module, name = {"index": (index_mod, "INDEX_CODE"), "adapters": (parquet_store, "ADAPTERS_CODE"),
                    "scan": (parquet_store, "SCAN_CODE")}[where]
    monkeypatch.setattr(module, name, None)
    monkeypatch.setattr(parquet_store, "_write_table_index", lambda *a: pytest.fail("index saved"))
    real = parquet_store.scan
    scans = []
    monkeypatch.setattr(parquet_store, "scan", lambda *a: scans.append(1) or real(*a))
    if where == "scan":
        monkeypatch.setattr(parquet_store, "_write_saved", lambda *a: pytest.fail("scan saved"))
    if where == "index":  # not even asked of the store
        monkeypatch.setattr(parquet_store.ParquetStore, "load_index", lambda *a: pytest.fail("index asked for"))
    for n in range(2):
        table = parquet.load_table(path, big_rows=0)
        assert full_view(index_of(table)) == expected and table.store.index_state is None
        assert len(counted) == n + 2  # built every time
        assert len(scans) == (n + 1 if where == "scan" else 0)  # the saved scan is used only with its code key
        compare_stores(load_resources(parquet.load_table(path, big_rows=10**9)), load_resources(table), summaries=5)
    assert {p.name: p.read_bytes() for p in index_dir.iterdir()} == before


# ---------------------------------------------------------------- the size of the saved index


def test_an_index_over_the_entry_limit_is_not_saved(tmp_path, index_dir, monkeypatch, counted):
    path = write(tmp_path / "data", "FAKE_RICH", rich_rows(), 4)
    built = index_of(parquet.load_table(path, big_rows=0))
    entries = index_mod._entries(built)
    assert entries == sum(len(a.labels) for a in built.axes.values()) + len(built.measures) + len(built.periods) + len(
        built.units)
    saved_file(index_dir).unlink()
    monkeypatch.setattr(index_mod, "MAX_SAVED_ENTRIES", entries - 1)
    table = parquet.load_table(path, big_rows=0)
    assert full_view(index_of(table)) == full_view(built) and table.store.index_state is None
    assert not saved_file(index_dir).exists()
    monkeypatch.setattr(index_mod, "MAX_SAVED_ENTRIES", entries)
    index_of(parquet.load_table(path, big_rows=0))
    assert saved_file(index_dir).exists() and len(counted) == 3


def test_an_index_file_over_the_byte_limit_is_neither_written_nor_read(tmp_path, index_dir, monkeypatch, counted):
    path = write(tmp_path / "data", "FAKE_RICH", rich_rows(), 4)
    built = index_of(parquet.load_table(path, big_rows=0))
    size = saved_file(index_dir).stat().st_size
    monkeypatch.setattr(parquet_store, "MAX_INDEX_BYTES", size - 1)
    table = parquet.load_table(path, big_rows=0)
    assert full_view(index_of(table)) == full_view(built) and table.store.index_state == "not saved"
    assert len(counted) == 2 and saved_file(index_dir).stat().st_size == size
    assert not [p for p in index_dir.iterdir() if p.name.endswith(".tmp")]
    monkeypatch.setattr(parquet_store, "MAX_INDEX_BYTES", size)
    table = parquet.load_table(path, big_rows=0)
    assert full_view(index_of(table)) == full_view(built) and table.store.index_state == "loaded"


def test_the_index_file_is_written_as_it_is_dumped(tmp_path, index_dir, monkeypatch):
    path = write(tmp_path / "data", "FAKE_RICH", rich_rows(), 4)
    table = parquet.load_table(path, big_rows=0)  # the scan's facts are a small text: json.dumps
    monkeypatch.setattr(json, "dumps", lambda *a, **k: pytest.fail("the whole text made in memory"))
    index_of(table)
    saved = json.loads(saved_file(index_dir).read_bytes().decode("utf-8"))
    assert saved["sha256"] == parquet.sha256_of(path) and saved["index"]["tbl_id"] == "FAKE_RICH"


# ---------------------------------------------------------------- a file held for a moment (Windows)


def test_a_rename_refused_twice_is_tried_again(tmp_path, index_dir, monkeypatch, counted):
    path = write(tmp_path / "data", "FAKE_RICH", rich_rows(), 4)
    real = parquet_store.os.replace
    refused, waits = [], []

    def replace(source, target):
        if len(refused) < 2:
            refused.append(target)
            raise PermissionError(13, "held by another process", str(target))
        return real(source, target)

    monkeypatch.setattr(parquet_store.os, "replace", replace)
    monkeypatch.setattr(parquet_store.time, "sleep", waits.append)
    index_of(parquet.load_table(path, big_rows=0))
    assert len(refused) == 2 and waits == list(parquet_store.REPLACE_RETRIES[:2])
    assert 0.5 <= sum(parquet_store.REPLACE_RETRIES) <= 2
    assert sorted(p.name for p in index_dir.iterdir()) == [f"{KEY}.descriptors.parquet", f"{KEY}.json",
                                                           f"{KEY}.table_index.json"]
    assert parquet.load_table(path, big_rows=0).store.facts.descriptors is None  # the scan was saved


def test_a_rename_refused_every_time_warns_and_leaves_no_temporary_file(tmp_path, index_dir, monkeypatch, caplog):
    path = write(tmp_path / "data", "FAKE_RICH", rich_rows(), 4)
    tries = []

    def replace(source, target):
        tries.append(target)
        raise PermissionError(13, "held by another process", str(target))

    monkeypatch.setattr(parquet_store.os, "replace", replace)
    monkeypatch.setattr(parquet_store.time, "sleep", lambda s: None)
    monkeypatch.setattr(parquet_store, "_warned", set())
    with caplog.at_level(logging.WARNING, logger=parquet_store.__name__):
        table = parquet.load_table(path, big_rows=0)
    assert len(tries) == len(parquet_store.REPLACE_RETRIES) + 1 and table.store.facts.descriptors is not None
    assert len([r for r in caplog.records if r.name == parquet_store.__name__]) == 1
    assert not [p for p in index_dir.iterdir() if p.name.endswith(".tmp")]


# ---------------------------------------------------------------- failures are harmless


def _payload_cases(saved: dict) -> list[tuple[str, dict]]:
    def changed(**kw) -> dict:
        return saved | {"index": saved["index"] | kw}

    index = saved["index"]
    axes = [list(a) for a in index["axes"]]
    axes[1] = [axes[1][0], axes[1][1], [[code, 7] for code, _ in axes[1][2]], axes[1][3]]
    return [("format", changed(format=index["format"] + 1)), ("tbl_id", changed(tbl_id="OTHER")),
            ("label type", changed(axes=axes)), ("axis shape", changed(axes=[a[:3] for a in index["axes"]])),
            ("measures", changed(measures=[[a, c] for a, c, _ in index["measures"]])),
            ("item_named", changed(item_named="yes")), ("units", changed(units=[1])),
            ("missing field", saved | {"index": {k: v for k, v in index.items() if k != "periods"}}),
            ("no index", {k: v for k, v in saved.items() if k != "index"}), ("not a dict", saved | {"index": [1]}),
            ("sha256", saved | {"sha256": "0" * 64})]


def test_a_corrupt_or_foreign_saved_index_is_built_again(tmp_path, index_dir, counted):
    path = write(tmp_path / "data", "FAKE_RICH", rich_rows(), 4)
    expected = full_view(index_of(parquet.load_table(path, big_rows=10**9)))
    index_of(parquet.load_table(path, big_rows=0))
    good = saved_file(index_dir).read_bytes()
    saved = json.loads(good)
    garbage = [("empty", b""), ("truncated", good[: len(good) // 2]), ("not json", b"{oops"),
               ("binary", b"\xff\xfe\x00"), ("a list", b"[]")] + [(name, json.dumps(case).encode("utf-8")) for name, case in _payload_cases(saved)]
    for name, data in garbage:
        saved_file(index_dir).write_bytes(data)
        before = len(counted)
        table = parquet.load_table(path, big_rows=0)
        assert full_view(index_of(table)) == expected, name
        assert len(counted) == before + 1 and table.store.index_state == "saved", name
        assert saved_file(index_dir).read_bytes() == good, name  # saved again, as it was
    table = parquet.load_table(path, big_rows=0)
    assert full_view(index_of(table)) == expected and table.store.index_state == "loaded"


def test_an_index_file_that_cannot_be_written_is_logged_once(tmp_path, index_dir, caplog, counted):
    path = write(tmp_path / "data", "FAKE_RICH", rich_rows(), 4)
    expected = full_view(index_of(parquet.load_table(path, big_rows=10**9)))
    parquet.load_table(path, big_rows=0)  # the scan is saved
    saved_file(index_dir).mkdir()  # a folder where the index file goes: it cannot be written (on every OS)
    with caplog.at_level(logging.WARNING, logger=parquet_store.__name__):
        for n in range(3):
            table = parquet.load_table(path, big_rows=0)
            assert full_view(index_of(table)) == expected and table.store.index_state == "not saved"
            assert len(counted) == n + 1
    warnings = [r for r in caplog.records if r.name == parquet_store.__name__]
    assert len(warnings) == 1 and str(index_dir) in warnings[0].getMessage()
    assert not [p for p in index_dir.iterdir() if p.name.endswith(".tmp")]  # no temporary file left


def test_an_unwritable_index_folder_saves_nothing_and_warns_once(tmp_path, monkeypatch, caplog, counted):
    path = write(tmp_path / "data", "FAKE_RICH", rich_rows(), 4)
    blocker = tmp_path / "a_file"
    blocker.write_text("x")
    monkeypatch.setattr(parquet_store, "INDEX_DIR", blocker / "index")
    expected = full_view(index_of(parquet.load_table(path, big_rows=10**9)))
    with caplog.at_level(logging.WARNING, logger=parquet_store.__name__):
        for n in range(2):
            table = parquet.load_table(path, big_rows=0)
            assert full_view(index_of(table)) == expected and len(counted) == n + 1
            assert table.store.index_state is None  # the scan was not saved: nowhere to save the index
    assert len([r for r in caplog.records if r.name == parquet_store.__name__]) == 1


def test_no_index_folder_saves_nothing(tmp_path, monkeypatch, counted):
    path = write(tmp_path / "data", "FAKE_RICH", rich_rows(), 4)
    monkeypatch.setattr(parquet_store, "INDEX_DIR", None)
    monkeypatch.setattr(parquet, "index_folder", lambda p: None)
    before = sorted(p.relative_to(tmp_path) for p in tmp_path.rglob("*"))
    for n in range(2):
        table = parquet.load_table(path, big_rows=0)
        index_of(table)
        assert table.store.index_state is None and len(counted) == n + 1
    assert sorted(p.relative_to(tmp_path) for p in tmp_path.rglob("*")) == before


def test_a_store_without_saving_still_gets_its_index(tmp_path, monkeypatch, counted):
    path = write(tmp_path / "data", "FAKE_RICH", rich_rows(), 4)
    table = parquet.load_table(path, big_rows=0)
    expected = full_view(index_of(parquet.load_table(path, big_rows=10**9)))

    def broken(*args):
        raise RuntimeError("no saving here")

    monkeypatch.setattr(parquet_store.ParquetStore, "load_index", broken)
    monkeypatch.setattr(parquet_store.ParquetStore, "save_index", broken)
    assert full_view(index_of(table)) == expected and len(counted) == 1


def test_a_memory_table_never_saves_or_loads_an_index(tmp_path, index_dir, monkeypatch):
    path = write(tmp_path / "data", "FAKE_RICH", rich_rows(), 4)
    monkeypatch.setattr(index_mod, "_saved_index", lambda table: pytest.fail("loaded"))
    monkeypatch.setattr(index_mod, "_save_index", lambda table, index: pytest.fail("saved"))
    table = parquet.load_table(path, big_rows=10**9)
    assert not table.file_backed and isinstance(table, TableInput)
    resources = load_resources(table)
    assert resources.index.periods and not index_dir.exists()


def test_a_period_added_to_the_file_is_in_the_rebuilt_index(tmp_path, counted):
    path = write(tmp_path / "data", "FAKE_RICH", rich_rows(), 4)
    index_of(parquet.load_table(path, big_rows=0))
    path.unlink()
    write(tmp_path / "data", "FAKE_RICH", rich_rows() + [_row("2025", "T1", [("행정구역별", "00", "전국"), ("성별", "0", "계")],
                                                             "5")], 4)
    assert "2025" in index_of(parquet.load_table(path, big_rows=0)).periods and len(counted) == 2


# ---------------------------------------------------------------- scripts/measure_big_tables.py


def test_the_measurement_records_a_built_then_a_loaded_index(tmp_path):
    from tests.test_measure_big_tables import _tables, measure

    data, index, out = tmp_path / "data", tmp_path / "index", tmp_path / "out.json"
    _tables(data)
    args = ["--data", str(data), "--out", str(out), "--index-dir", str(index), "--min-rows", "0", "--big-rows", "0",
            "--tables", "101_FAKE_GOOD"]
    for _ in range(2):  # --fresh deletes the saved index too: the first child builds it every time
        assert measure.main(args + ["--fresh"]) == 0
        good = json.loads(out.read_text(encoding="utf-8"))["tables"]["101_FAKE_GOOD"]
        assert good["first"]["index_from"] == "built" and good["first"]["index_state"] == "saved"
        assert good["reuse"]["index_from"] == "loaded" and good["reuse"]["saved_index_bytes"] > 0
        assert good["reuse"]["index_state"] == "loaded" and "index_saved" not in good["reuse"]
    assert measure.main(args) == 0  # without --fresh the first child finds the saved index
    good = json.loads(out.read_text(encoding="utf-8"))["tables"]["101_FAKE_GOOD"]
    assert good["first"]["index_from"] == "loaded" and good["first"]["scan"] == "saved"


def test_fresh_deletes_every_saved_file_of_the_table(tmp_path, monkeypatch):
    from tests.test_measure_big_tables import measure

    path = write(tmp_path / "data", "FAKE_RICH", rich_rows(), 4)
    monkeypatch.setattr(parquet_store, "INDEX_DIR", tmp_path / "index")
    files = measure.saved_scan_files(path) + [measure.saved_index_file(path)]
    assert sorted(f.name for f in files) == [f"{KEY}.descriptors.parquet", f"{KEY}.json", f"{KEY}.merge.parquet",
                                             f"{KEY}.table_index.json"]
    assert {f.parent for f in files} == {tmp_path / "index"}
    assert all(name in measure.__doc__ for name in ("<key>.merge.parquet", "<key>.table_index.json"))
