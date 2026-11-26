"""The table index: the item column and each category as code -> label trees, the periods, and each code's measures.

Moved from kosis_chat/judge/question.py in T1-28 S2 with what it needs (is_total, cell_measure, the axis patterns);
judge/question.py re-exports every name. index_of(table) keeps one index per TableInput (TableInput.caches), so the
pipeline, the judge and the compute checks share it instead of building it again (compute/semantics.py had its own
module-global cache keyed by id(table)).
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from kosis_chat.prepare.schemas import Cell, TableInput
from kosis_chat.prepare.store import ITEM_AXIS, axis_code, source_hash

try:
    from kosis_ai import units
except ImportError:  # same fallback as judge/question.py: src/ is not on the path for `python -m ...` (S6 removes it)
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
    from kosis_ai import units

from kosis_chat.shared.patterns import UNIT_PAREN  # noqa: E402  (after the block above: it imports kosis_ai)

_UNIT_PAREN = UNIT_PAREN  # old name (T1-28 S2b: one trailing '(...)' pattern, kosis_ai.units.UNIT_PAREN)
_REGION_AXIS = re.compile(r"행정구역|시도|시군구|지역")
# Categories that are alternative scenarios (장래추계 고위/중위/저위), not parts of a whole.
SCENARIO_AXIS = re.compile(r"가정|시나리오|추계")
TOTAL_LABELS = ("합계", "계", "총계", "전체")

_INDEX_CACHE = "prepare.index"  # LookupCaches name of the index


# Loop 6: a total word joined to the stem of a name by a mark (an underscore, a hyphen or a space): '일반가구_계',
# '임금근로자-계', '남자 합계', '농가-전체'. Group 1 is the stem (the group the total is of). Was '_계' and ' 계' only.
TOTAL_MARK = re.compile(r"^(.*[^\s_\-])\s*(?:[_\-]|\s)\s*(?:"
                        + "|".join(sorted((*TOTAL_LABELS, "소계"), key=len, reverse=True)) + r")$")


def total_stem(label: str) -> str | None:
    """The stem of a name made of a stem and a total mark ('임금근로자-계' -> '임금근로자'), None for any other name."""
    core = _UNIT_PAREN.sub("", label).strip().lstrip("-").strip()
    m = TOTAL_MARK.match(core)
    return m.group(1).strip() if m else None


def is_total(label: str) -> bool:
    core = _UNIT_PAREN.sub("", label).strip()
    return core in TOTAL_LABELS or core == "전국" or core.endswith(("_계", " 계")) or total_stem(label) is not None


@dataclass
class Axis:
    """The item column or one category: code -> label, code -> parent code (KOSIS codes nest by prefix)."""

    name: str
    labels: dict[str, str] = field(default_factory=dict)
    parents: dict[str, str | None] = field(default_factory=dict)

    @property
    def regional(self) -> bool:
        return bool(_REGION_AXIS.search(self.name))

    @property
    def has_total(self) -> bool:
        return any(is_total(label) for label in self.labels.values())

    def depth(self, code: str) -> int:
        n, parent = 0, self.parents.get(code)
        while parent is not None:
            n, parent = n + 1, self.parents.get(parent)
        return n

    def ancestors(self, code: str) -> list[str]:
        out, parent = [], self.parents.get(code)
        while parent is not None:
            out.append(parent)
            parent = self.parents.get(parent)
        return out


def _link_parents(axis: Axis) -> None:
    codes = set(axis.labels)
    for code in axis.labels:
        axis.parents[code] = next((code[:k] for k in range(len(code) - 1, 0, -1) if code[:k] in codes), None)
    _link_sections(axis)


_TITLE_PAREN = re.compile(r"\([^)]*\)")
_TITLE_TAIL = re.compile(r"[_\s]*(?:시도|시군구|지역|전국)$")
# Loop 12: a tail that names the regional unit of the breakdown, joined to the end of a title by a dash, an underscore
# or a space ('경제활동인구 - 시도', '취업자_시군구', '취업자 (시도)'), repeated ('- 시도/시군구').
_REGION_UNIT = r"(?:시도|시군구|읍면동|시군|지역|전국|광역시도)(?:별)?"
_REGION_TAIL = re.compile(rf"(?:\s*[-–_/·]\s*|\s+|\s*\(\s*){_REGION_UNIT}\s*\)?\s*$")
_NAME_WORDS = re.compile(r"[0-9A-Za-z가-힣]+")


def _title_body(title: str) -> str:
    """The title without its regional tails (_REGION_TAIL) and the dashes, underscores and spaces left at its end."""
    text = (title or "").strip()
    while True:
        cut = _REGION_TAIL.sub("", text).rstrip(" -–_/·")
        if cut == text:
            return text
        text = cut


def title_noun(title: str) -> str:
    """Loop 11: the noun a table's title ends with, squashed: the last word once the brackets and the words of a
    breakdown ('시도별', '산업별') are cut off ('가상 시도별 산업별 취업자' -> '취업자', '취업자(시도별)' -> '취업자');
    '' when the title ends in no such word. Loop 12: also once a tail naming the regional unit is cut off
    (_REGION_TAIL: '산업별 취업자 - 시도', '취업자_시군구')."""
    words = [w for w in re.split(r"[\s,·/\-–_]+", _TITLE_PAREN.sub(" ", _title_body(title))) if w and not w.endswith("별")]
    return _TITLE_TAIL.sub("", words[-1]) if words else ""


def _name_words(label: str) -> list[str]:
    """The words of an item name, brackets' words included ('15세 이상 취업자(전체)' -> '15세', '이상', '취업자', '전체')."""
    return _NAME_WORDS.findall(_UNIT_PAREN.sub("", label))


def _unmarked_whole(axis: Axis, title: str, measures: dict[tuple[str, str], set]) -> bool:
    """Loop 11: the item category's first name is the whole of the others though it carries no total mark: no code
    has a parent, the first name (its brackets cut off) is the noun the table's title ends with (the group the table
    counts: '취업자' in '시도별 산업별 취업자'), and every item is measured as that one is (one unit: not a table of
    different measures whose first one the title happens to name). '취업자' then holds '농림어업', '제조업', ... .

    Loop 12: the first name is compared with the title word by word, not letter by letter: it holds the title's noun
    (title_noun) and each of its words is in the title (spaces left out, words in brackets included: '15세 이상 취업자'
    beside '취업자(15세 이상)', '경제활동인구' beside '경제 활동 인구'), and no other item's words are all in the title
    too (a title naming two measures side by side: '출생아 및 사망자')."""
    codes = list(axis.labels)
    if len(codes) < 3 or any(axis.parents.get(c) is not None for c in codes) or is_total(axis.labels[codes[0]]):
        return False
    noun = re.sub(r"\s+", "", title_noun(title))
    flat = re.sub(r"[^0-9A-Za-z가-힣]", "", _title_body(title))

    def in_title(label: str) -> bool:
        words = _name_words(label)
        return bool(words) and all(w in flat for w in words)

    first = re.sub(r"\s+", "", _TITLE_PAREN.sub("", _UNIT_PAREN.sub("", axis.labels[codes[0]])))
    if len(noun) < 2 or not (first == noun or (noun in first and in_title(axis.labels[codes[0]]))):
        return False
    if first != noun and any(in_title(axis.labels[c]) for c in codes[1:]):
        return False
    units = [measures.get((axis.name, c), set()) for c in codes]
    return len(units[0]) == 1 and all(u == units[0] for u in units)


def _link_whole(axis: Axis) -> None:
    """Loop 11: the parents under an unmarked whole (_unmarked_whole): the sections _link_sections reads when marked
    subtotals follow it, else every other name directly under the whole."""
    if _link_sections(axis, unmarked=True):
        return
    whole = next(iter(axis.labels))
    axis.parents.update({c: whole for c in axis.labels if c != whole})


def _link_sections(axis: Axis, unmarked: bool = False) -> bool:
    """Loop 6: the parents of a category whose codes do not nest by prefix but whose names do, in table order: the
    whole first ('취업자-계', or '계'), then subtotals that each carry a total mark ('임금근로자-계') and are followed by
    their details ('상용근로자', '임시근로자'), up to the next subtotal. A subtotal's parent is the whole; a detail's
    is the subtotal before it (the whole when it comes before every subtotal). Nothing is linked when a code already
    has a prefix parent, when the first name is not a total, when the whole is a marked name not followed right away
    by a subtotal (that is a list of subtotals: '남자_계', '남자_20대', '여자_계'), or when a subtotal has no detail.
    A code that has a prefix parent keeps it (and counts as a detail of its subtotal: 'B' -> 'B1'); only the codes of
    the top level are linked so. Loop 11: unmarked, the whole is an unmarked first name (_unmarked_whole) and one
    marked subtotal is enough. True when the parents were linked."""
    codes = list(axis.labels)
    top = [c for c in codes if axis.parents.get(c) is None]
    if len(codes) < 3 or not top or top[0] != codes[0]:
        return False
    marked = {c for c in top if is_total(axis.labels[c])}
    whole = codes[0]
    if unmarked:
        if not marked or whole in marked:
            return False
    elif len(marked) < 2 or whole not in marked:
        return False
    elif total_stem(axis.labels[whole]) is not None and top[1] not in marked:
        return False
    parents: dict[str, str] = {}
    current, details, linked = None, 0, 0
    for code in codes[1:]:
        if axis.parents.get(code) is not None:  # a prefix child: its parent stays
            details += current is not None and axis.parents[code] == current
            continue
        if code in marked:
            if current is not None and not details:
                return False
            linked += current is not None
            current, details = code, 0
            parents[code] = whole
        else:
            parents[code] = current or whole
            details += current is not None
    # loop 12: under an unmarked whole, a last subtotal with no detail of its own (its details are not in the table)
    # is a subtotal under the whole all the same; the subtotals before it keep their details
    if current is None or not (details or (unmarked and linked)):
        return False
    axis.parents.update(parents)
    return True


# ponytail: S2 keeps this cache: a pure function of its value arguments (not of a table object), bounded, and
# shared by tables that repeat the same item / labels; moving it into LookupCaches would only lose that sharing.
@lru_cache(maxsize=4096)
def _measure(tbl_id: str, unit: str | None, item: str, dim_labels: tuple[str, ...]) -> tuple[str | None, str | None]:
    """(unit, measure_kind) by the T1-05 rules (src/kosis_ai/units.py, re-exported by kosis_ai.normalize)."""
    resolved, _ = units.resolve_unit(unit, item, list(dim_labels))
    return resolved, units.measure_kind(tbl_id, resolved, item, list(dim_labels))


def cell_measure(table: TableInput, cell: Cell) -> tuple[str | None, str | None]:
    given = cell.unit if cell.unit_source == "given" else None
    # Scenario labels name every assumption ('출산율-중위 / 기대수명-중위'), which the keyword rules would read
    # as a rate in 년 for every cell; the measure is in the other labels.
    labels = tuple(label for name, label in cell.dims.items() if not _scenario(name))
    return _measure(table.meta.tbl_id, given, cell.item, labels)


# ponytail: a pure function of a category name, bounded (a table has a few names); not per table.
@lru_cache(maxsize=1024)
def _scenario(name: str) -> bool:
    """SCENARIO_AXIS.search(name), once per category name (cell_measure runs per cell; a table has a few names)."""
    return bool(SCENARIO_AXIS.search(name))


@dataclass
class TableIndex:
    table: TableInput
    axes: dict[str, Axis]
    periods: list[str]
    # (axis, code) -> every (unit, measure_kind) of its cells; a measure label has exactly one.
    measures: dict[tuple[str, str], set[tuple[str | None, str | None]]]
    # False when one item only wraps a '기본항목별' category that holds the real items (DT_1B8000F): facts then
    # leave the item out (compute/engine.py). Here, not in the cell store, because it needs the whole table.
    item_named: bool
    units: list[str]  # sorted distinct non-empty units of the cells (the multi-table prompt shows them)

    def axis_of(self, cell: Cell, name: str) -> str:
        return axis_code(cell, name)

    def year_ok(self, year: str) -> bool:
        return any(p.startswith(year) for p in self.periods)

    def period_ok(self, period: str) -> bool:
        return period in self.periods if len(period) > 4 else self.year_ok(period)

    def measure_of(self, axis: str, code: str) -> tuple[str | None, str | None] | None:
        found = self.measures.get((axis, code), set())
        return next(iter(found)) if len(found) == 1 else None


def build_index(table: TableInput) -> TableIndex:
    """The index of a table: from its cells, or, for a file-backed table, from its store's descriptor cells (one per
    distinct cell but for period and value, in first-seen order) and its periods. Both give the same index for one
    file (T1-31 S3): the loop below reads nothing a descriptor cell does not carry, and a later cell with the same
    descriptor changes nothing (setdefault, sets). A file-backed table's index is saved by its store once built and
    loaded from there by the next process (the saved index, below)."""
    if table.file_backed:
        saved = _saved_index(table)
        if saved is not None:
            return saved
        index = index_from(table, table.store.index_cells(), table.store.periods)
        _save_index(table, index)
        return index
    return index_from(table, table.cells)


def index_from(table: TableInput, cells: Iterable[Cell], periods: Iterable[str] | None = None) -> TableIndex:
    """The index over `cells` in table order (read once); periods: the table's periods, else those of the cells."""
    axes: dict[str, Axis] = {ITEM_AXIS: Axis(ITEM_AXIS)}
    measures: dict[tuple[str, str], set] = {}
    found: set[str] = set() if periods is None else set(periods)
    units: set[str] = set()
    item_codes: set[str] = set()  # not axes[ITEM_AXIS]: a category named '항목' is filed under that axis too
    first = None
    for c in cells:
        if first is None:
            first = c
        if periods is None:
            found.add(c.period_raw)
        item_codes.add(c.item_code)
        if c.unit:
            units.add(c.unit)
        axes[ITEM_AXIS].labels.setdefault(c.item_code, c.item)
        measure = cell_measure(table, c)
        measures.setdefault((ITEM_AXIS, c.item_code), set()).add(measure)
        for name, code in c.dim_codes.items():
            axis = axes.get(name)  # not setdefault(name, Axis(name)): no Axis made and dropped per cell and category
            if axis is None:
                axis = axes[name] = Axis(name)
            axis.labels.setdefault(code, c.dims.get(name, ""))
            measures.setdefault((name, code), set()).add(measure)
    for axis in axes.values():
        _link_parents(axis)
    if _unmarked_whole(axes[ITEM_AXIS], getattr(table.meta, "tbl_nm", "") or "", measures):  # loop 11
        _link_whole(axes[ITEM_AXIS])
    item_named = len(item_codes) > 1 or first is None or not any("항목" in name for name in first.dims)
    return TableIndex(table=table, axes=axes, periods=sorted(found), measures=measures, item_named=item_named,
                      units=sorted(units))


# ---------------------------------------------------------------- the saved index of a file-backed table (T1-31 S3)
#
# Building the index of a file-backed table loops over every descriptor (up to millions, 25-45 us each), in every
# process. So the built index is saved by the table's store (FileTable.save_index: beside the saved scan, keyed by the
# file's sha256 and the store's own code) as plain JSON data, and the next index_of loads it instead. The payload is
# decided here: every field of the TableIndex but the table, in dict order, with the tbl_id of the meta (the measure
# rules read it, and it comes from tables.csv, not from the file). INDEX_CODE is the code key: the source text of
# every module the index content depends on (below), so an edit of any of them rebuilds the index by itself.
# A payload that does not read back exactly (another format, another table, a wrong type) is built again; nothing
# here raises, and a memory table never saves anything. Nothing is saved or loaded when INDEX_CODE is None (a source
# could not be read: prepare.store.source_hash), nor for an index of more than MAX_SAVED_ENTRIES entries (its JSON
# would cost about as much to load as the index costs to build; the store also refuses to write or read a file over
# its own byte limit, parquet_store.MAX_INDEX_BYTES).

SAVED_FORMAT = 1
# Entries of a saved index (_entries: codes of every axis, measures, periods, units). The largest provided tables
# have about 8,600 (a 700 KB file, 60 to 80 bytes an entry). Measured on made-up tables: 720,000 entries, a 33 MB
# file, loaded in 2.3 s (built in 5.7 s), the load raising the peak by about 0.5 GB, as the build does; 3,000,000
# codes on one category (about 6 million entries), a 288 MB file whose load peaked above its build.
MAX_SAVED_ENTRIES = 1_000_000


def _code_folders() -> list[Path]:
    """The folders whose .py sources the index content depends on: kosis_chat/prepare (this module, the cells'
    conversion in convert.py and clean.py, the store protocol, the schemas), kosis_chat/shared (the label patterns,
    imported through kosis_ai too) and the kosis_ai package (units: the measure and unit rules; schemas, which
    shared imports). The file side (kosis_chat/adapters: the scan and the descriptor cells) is the store's part of
    the key."""
    chat = Path(__file__).resolve().parents[1]
    return [chat / "prepare", chat / "shared", Path(units.__file__).resolve().parent]


def _sources(folders: Iterable[Path]) -> Iterator[tuple[str, Path]]:
    """(folder name / relative path, path) of each folder's .py files, sorted within each folder."""
    for folder in folders:
        for source in sorted(p for p in folder.rglob("*.py") if "__pycache__" not in p.parts):
            yield f"{folder.name}/{source.relative_to(folder).as_posix()}", source


def _source_hash(folders: Iterable[Path]) -> str | None:
    """The sha256 over each folder's .py files (relative path and text, CRLF read as LF: one hash on every OS); None
    when a file cannot be read or there is none (prepare.store.source_hash: the saved index is then not used)."""
    return source_hash(_sources(folders))


INDEX_CODE = _source_hash(_code_folders())


def _payload(index: TableIndex) -> dict:
    return {
        "format": SAVED_FORMAT,
        "tbl_id": index.table.meta.tbl_id,
        "axes": [[key, axis.name, list(axis.labels.items()), list(axis.parents.items())]
                 for key, axis in index.axes.items()],
        "periods": list(index.periods),
        "measures": [[axis, code, [list(m) for m in found]] for (axis, code), found in index.measures.items()],
        "item_named": index.item_named,
        "units": list(index.units),
    }


def _text(value) -> str:
    if type(value) is not str:
        raise TypeError("not a string")
    return value


def _maybe(value) -> str | None:
    return None if value is None else _text(value)


def _from_payload(table: TableInput, payload: dict) -> TableIndex:
    """The TableIndex of a saved payload; an error when it is not exactly one of this table's (format, tbl_id,
    types)."""
    if payload["format"] != SAVED_FORMAT or payload["tbl_id"] != table.meta.tbl_id:
        raise ValueError("another format or table")
    axes: dict[str, Axis] = {}
    for key, name, labels, parents in payload["axes"]:
        axes[_text(key)] = Axis(_text(name), {_text(c): _text(label) for c, label in labels},
                                {_text(c): _maybe(parent) for c, parent in parents})
    measures: dict[tuple[str, str], set] = {}
    for axis, code, found in payload["measures"]:
        measures[_text(axis), _text(code)] = {(_maybe(unit), _maybe(kind)) for unit, kind in found}
    item_named = payload["item_named"]
    if type(item_named) is not bool:
        raise TypeError("item_named")
    return TableIndex(table=table, axes=axes, periods=[_text(p) for p in payload["periods"]], measures=measures,
                      item_named=item_named, units=[_text(u) for u in payload["units"]])


def _entries(index: TableIndex) -> int:
    return sum(len(axis.labels) for axis in index.axes.values()) + len(index.measures) + len(index.periods) + len(
        index.units)


def _saved_index(table: TableInput) -> TableIndex | None:
    if INDEX_CODE is None:
        return None
    try:
        payload = table.store.load_index(INDEX_CODE)
        return None if payload is None else _from_payload(table, payload)
    except Exception:  # a store without saving, or a payload that does not read back: build it
        return None


def _save_index(table: TableInput, index: TableIndex) -> None:
    if INDEX_CODE is None or _entries(index) > MAX_SAVED_ENTRIES:
        return
    try:
        table.store.save_index(INDEX_CODE, _payload(index))
    except Exception:  # the store logs what it cannot write; a store without saving saves nothing
        return


def index_of(table: TableInput) -> TableIndex:
    """The index of this TableInput object, built once (TableInput.caches). Callers must not change it."""
    return table.caches.get(_INDEX_CACHE, lambda: build_index(table))
