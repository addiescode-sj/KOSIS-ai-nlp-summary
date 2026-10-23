"""Golden set contract for automatic grading: GoldenItem (Pydantic) and checks of golden facts against Parquet cells.

The file format is the one scripts/golden_to_parquet.py writes: `table` (Parquet path), `table_sha256`, `scope`.
`scope` belongs to the B0 runner (kosis_chat/data/scope.py); grading does not read it.
"""

from __future__ import annotations

import csv
import hashlib
import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from kosis_ai.schemas import Cell, Fact, QuestionKind, QuestionType, RefusalReason, Source, StatResult

ROOT = Path(__file__).resolve().parents[1]


class GoldenError(ValueError):
    pass


class GoldenItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    kind: QuestionKind
    type: QuestionType | None = None
    question: str = Field(min_length=1)
    table: str  # data/values/<org>_<tbl>.parquet
    table_sha256: str
    scope: dict[str, Any] | None = None  # runner-only (oracle table part)
    difficulty: str | None = None
    refusal_reason: RefusalReason | None = None  # representative reason; allowed_reasons is what grading uses
    allowed_reasons: list[RefusalReason] = Field(default_factory=list)
    required_facts: list[Fact] = Field(default_factory=list)
    aux_facts: list[Fact] = Field(default_factory=list)
    default_applied: dict[str, str] = Field(default_factory=dict)
    clarify_options: list[str] = Field(default_factory=list)
    eda_traps: list[str] = Field(default_factory=list)
    notes: str | None = None

    @model_validator(mode="after")
    def _kind_needs(self) -> GoldenItem:
        if self.kind == "not_answerable":
            if not self.allowed_reasons:
                raise ValueError("not_answerable needs allowed_reasons")
            if self.refusal_reason and self.refusal_reason not in self.allowed_reasons:
                raise ValueError(f"refusal_reason {self.refusal_reason!r} is not in allowed_reasons")
        elif self.refusal_reason or self.allowed_reasons:
            raise ValueError(f"{self.kind} must not carry refusal reasons")
        if self.kind in ("answerable", "default_allowed") and not self.required_facts:
            raise ValueError(f"{self.kind} needs required_facts")
        if self.kind == "default_allowed" and not self.default_applied:
            raise ValueError("default_allowed needs default_applied")
        if self.kind == "clarify" and len(self.clarify_options) < 2:
            raise ValueError("clarify needs at least two clarify_options")
        StatResult(facts=self.required_facts + self.aux_facts)  # unique ids, derivations reference known facts
        return self

    @property
    def path(self) -> Path:
        p = Path(self.table)
        return p if p.is_absolute() else ROOT / p


@lru_cache(maxsize=16)
def read_cells(path: Path) -> dict[str, Cell]:
    """All cells of one Parquet table keyed by Cell.ref() (the source_ref format). Cached per file."""
    import pyarrow.parquet as pq

    cells = (Cell.from_row(row) for row in pq.read_table(path).to_pylist())
    return {c.ref(): c for c in cells}


def read_source(path: Path) -> Source:
    """Citation for a Parquet table from the tables.csv row whose key is the file stem (data/values/x -> data/tables.csv)."""
    with (path.parent.parent / "tables.csv").open(encoding="utf-8-sig", newline="") as f:
        row = next(r for r in csv.DictReader(f) if r["key"] == path.stem)
    return Source(org_id=row["org_id"], tbl_id=row["tbl_id"], survey_nm=row["survey_name"], tbl_nm=row["table_name"],
                  url=row.get("url") or None)


def check_facts(item: GoldenItem, cells: dict[str, Cell]) -> list[str]:
    """Problems where a golden cell fact disagrees with the Parquet data."""
    problems = []
    for f in item.required_facts + item.aux_facts:
        if not f.source_ref:
            continue
        where = f"{item.id}/{f.id} {f.source_ref}"
        cell = cells.get(f.source_ref)
        if cell is None:
            problems.append(f"{where}: no such cell in the Parquet table")
        elif f.value is None and (cell.value is not None or cell.value_raw.strip() != f.raw):
            problems.append(f"{where}: golden raw {f.raw!r}, data {cell.value_raw.strip()!r}")
        elif f.value is not None and (cell.value is None or abs(cell.value - f.value) > 1e-9 * max(1.0, abs(f.value))):
            problems.append(f"{where}: golden value {f.value}, data {cell.value_raw.strip()!r}")
    return problems


def load_items(path: Path) -> tuple[str, list[GoldenItem], str]:
    """Read a golden set (.jsonl file or directory of them) as GoldenItems -> (version, items, sha256).

    Raises GoldenError listing every problem (schema, missing table, sha256 mismatch, fact vs data), so grading
    never runs on a golden set that disagrees with the data.
    """
    files = sorted(path.glob("*.jsonl")) if path.is_dir() else [path]
    if not files:
        raise GoldenError(f"no .jsonl files in {path}")
    items: list[GoldenItem] = []
    problems: list[str] = []
    hashes: dict[Path, str] = {}
    digest = hashlib.sha256()
    for f in files:
        digest.update(f.read_bytes())
        for line_no, line in enumerate(f.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            try:
                item = GoldenItem.model_validate(json.loads(line))
            except (ValidationError, json.JSONDecodeError) as exc:
                problems.append(f"{f.name}:{line_no}: {exc}")
                continue
            if not item.path.exists():
                problems.append(f"{f.name}:{line_no}: table {item.table} not found")
                continue
            if item.path not in hashes:
                hashes[item.path] = hashlib.sha256(item.path.read_bytes()).hexdigest()
            if hashes[item.path] != item.table_sha256:
                problems.append(f"{item.id}: {item.table} sha256 {hashes[item.path][:12]} != golden {item.table_sha256[:12]}")
                continue
            problems += check_facts(item, read_cells(item.path))
            items.append(item)
    ids = [i.id for i in items]
    problems += [f"duplicate question id {i!r}" for i in sorted({i for i in ids if ids.count(i) > 1})]
    if problems:
        raise GoldenError("golden set problems:\n  " + "\n  ".join(problems))
    version = path.name if path.is_dir() else path.parent.name
    return version, items, digest.hexdigest()
