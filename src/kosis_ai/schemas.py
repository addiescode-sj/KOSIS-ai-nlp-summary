"""Shared I/O contract for summarize_stat (stage 1), table search (stage 2) and the agent tools (stage 3).

Numbers live only in Fact. The LLM picks AnswerParts (fact ids + connecting text); code renders numbers.
Field names follow the golden set (eval/golden/stat_golden.jsonl) so golden facts load into the same Fact model.
"""

from __future__ import annotations

import re
from typing import Annotated, Literal, get_args

from pydantic import BaseModel, Field, model_validator

QuestionType = Literal["value", "trend", "compare", "share", "extreme"]
QuestionKind = Literal["answerable", "not_answerable", "clarify", "default_allowed"]
RefusalReason = Literal[
    "no_period",  # requested period is not in the table
    "no_item",  # requested item is not in the table
    "class_mismatch",  # requested category does not exist at that level or time
    "unit_mismatch",  # values in different units cannot be combined
    "not_aggregatable",  # summing rates, averages or stocks across time
    "causal",  # asks why; the table cannot support a cause
    "missing_value",  # the cell exists but holds a symbol such as '-' or 'X'
    "out_of_scope",  # outside the provided tables
]
MeasureKind = Literal["count_stock", "count_flow", "ratio", "average", "index"]
Outcome = Literal["answered", "refused", "clarify"]
# value_kind in data/values/*.parquet: "ok" has a number, every other kind is a symbol kept in value_raw.
ValueKind = Literal["ok", "dash", "x_secret", "blank", "ellipsis", "asterisk", "other_text"]

QUESTION_TYPES = get_args(QuestionType)
QUESTION_KINDS = get_args(QuestionKind)
REFUSAL_REASONS = get_args(RefusalReason)
VALUE_KINDS = get_args(ValueKind)
MAX_DIMS = 8  # dim1..dim8 columns in the Parquet files
# Trailing "(unit)" in an item name or dim label, e.g. "출생건수 (명)", "출생아수(명)".
# ponytail: short no-space/no-digit token only, so "(출생시 기대여명)" or "(2012~)" are not units; a unit table if this misses too many.
_UNIT_SUFFIX = re.compile(r"\(([^()\s\d~]{1,5})\)\s*$")
_IDENT = re.compile(r"[A-Za-z_]\w*")


class Source(BaseModel):
    org_id: str
    tbl_id: str
    survey_nm: str
    tbl_nm: str
    url: str | None = None

    def citation(self) -> str:
        """'KOSIS / survey / table (url)' — built by code, never by the LLM."""
        text = f"KOSIS / {self.survey_nm} / {self.tbl_nm}"
        return f"{text} ({self.url})" if self.url else text


class TableInfo(BaseModel):
    """One row of data/tables.csv: a table in the provided scope and where its values are."""

    key: str  # "<org_id>_<tbl_id>"
    org_id: str
    tbl_id: str
    survey_name: str
    organization: str
    table_name: str
    period: str  # listed period, e.g. "년 (2015 ~ 2025)"
    source: Literal["api", "zip"]
    coverage: str  # e.g. "A_정상", "B_수록시점차이", "C_수집누락"
    missing_periods: str = ""  # empty when nothing is missing
    rows: int
    numeric_rows: int
    symbol_rows: int
    parquet: str  # path relative to data/, e.g. "values/101_DT_1B8000H.parquet"
    url: str
    menu_path: str

    def to_source(self) -> Source:
        return Source(org_id=self.org_id, tbl_id=self.tbl_id, survey_nm=self.survey_name, tbl_nm=self.table_name, url=self.url)


class Dim(BaseModel):
    name: str  # e.g. "행정구역별"
    code: str  # e.g. "31"
    label: str  # e.g. "경기도"


class Cell(BaseModel):
    """One row of data/values/<org>_<tbl>.parquet. value_raw is never changed."""

    org_id: str
    tbl_id: str
    source: Literal["api", "zip"]
    prd_se: str  # period type: Y, M, Q, F (multi-year), IR (irregular)
    period: str
    item_id: str
    item_name: str
    unit: str | None = None  # missing for many tables (docs/eda-full.md)
    dims: list[Dim] = Field(default_factory=list, max_length=MAX_DIMS)
    value_raw: str
    value: float | None = None
    value_kind: ValueKind

    @model_validator(mode="after")
    def _value_matches_kind(self) -> Cell:
        if (self.value is not None) != (self.value_kind == "ok"):
            raise ValueError(f"{self.ref()}: value must be set exactly when value_kind is 'ok'")
        return self

    @classmethod
    def from_row(cls, row: dict) -> Cell:
        """Build from a flat Parquet row (dim1_name, dim1_code, dim1_label, ... dim8_*)."""
        dims = [
            Dim(name=row[f"dim{i}_name"], code=row[f"dim{i}_code"], label=row[f"dim{i}_label"])
            for i in range(1, MAX_DIMS + 1)
            if row.get(f"dim{i}_code") is not None
        ]
        flat = {k: v for k, v in row.items() if not k.startswith("dim")}
        return cls(**flat, dims=dims)

    def ref(self) -> str:
        """source_ref used by Fact and the golden set: 'tbl_id|item_id|dim codes...|period'."""
        return "|".join([self.tbl_id, self.item_id, *(d.code for d in self.dims), self.period])

    def unit_or_suffix(self) -> str | None:
        """Unit column, else a trailing '(unit)' in the item name, else in a dim label. None when unknown."""
        if self.unit:
            return self.unit
        for text in [self.item_name, *(d.label for d in self.dims)]:
            if m := _UNIT_SUFFIX.search(text):
                return m.group(1)
        return None

    def to_fact(self, fact_id: str) -> Fact:
        return Fact(
            id=fact_id,
            value=self.value,
            unit=self.unit_or_suffix(),
            period=self.period,
            dims={d.name: d.label for d in self.dims},
            source_ref=self.ref(),
            raw=None if self.value_kind == "ok" else self.value_raw.strip(),
        )


class StatQuery(BaseModel):
    """Question analysis result (T1-06)."""

    question: str
    type: QuestionType | None = None  # None for causal or out-of-scope questions
    item: str | None = None
    periods: list[str] = Field(default_factory=list)
    filters: dict[str, str] = Field(default_factory=dict)
    ambiguity: str | None = None  # what is unclear; None when the question is fully specified


class Fact(BaseModel):
    """One number: a source cell, a value computed from other facts, or a note."""

    id: str
    value: int | float | None = None
    unit: str | None = None
    period: str | None = None
    dims: dict[str, str] | None = None
    source_ref: str | None = None  # Cell.ref() of the source cell, e.g. "DT_1B8000H|T10|31|2025"
    raw: str | None = None  # original symbol when value is None, e.g. "-" or "X"
    derivation: str | None = None  # formula over other fact ids, e.g. "(f2 / f1 - 1) * 100"
    round: int | None = None
    measure_kind: MeasureKind | None = None
    note: str | None = None

    @model_validator(mode="after")
    def _traceable(self) -> Fact:
        if self.value is not None and not (self.source_ref or self.derivation):
            raise ValueError(f"fact {self.id}: a value needs a source_ref or a derivation")
        if not (self.source_ref or self.derivation or self.note):
            raise ValueError(f"fact {self.id}: needs a source_ref, a derivation or a note")
        if self.derivation and self.round is None:
            raise ValueError(f"fact {self.id}: a derivation needs round")
        return self


class StatResult(BaseModel):
    """Facts computed by code for one question (T1-07)."""

    facts: list[Fact] = Field(default_factory=list)

    @model_validator(mode="after")
    def _unique_ids(self) -> StatResult:
        ids = [f.id for f in self.facts]
        dupes = sorted({i for i in ids if ids.count(i) > 1})
        if dupes:
            raise ValueError(f"duplicate fact ids: {dupes}")
        unknown = {f.id: sorted(set(_IDENT.findall(f.derivation)) - set(ids)) for f in self.facts if f.derivation}
        unknown = {k: v for k, v in unknown.items() if v}
        if unknown:
            raise ValueError(f"derivations reference unknown facts: {unknown}")
        return self

    def get(self, fact_id: str) -> Fact | None:
        return next((f for f in self.facts if f.id == fact_id), None)


class Refusal(BaseModel):
    """Why the question cannot be answered (T1-08). Graded correct when reason is in the golden allowed_reasons."""

    reason: RefusalReason
    message: str = Field(min_length=1)
    available: str | None = None  # what can be answered instead, e.g. the latest published year


class FactPart(BaseModel):
    kind: Literal["fact"] = "fact"
    fact_id: str


class TextPart(BaseModel):
    kind: Literal["text"] = "text"
    text: str  # connecting text; numeric claims here are rejected by the verifier (V4, T1-10)


AnswerPart = Annotated[FactPart | TextPart, Field(discriminator="kind")]


class StatAnswer(BaseModel):
    """Return value of summarize_stat (T1-11) and of the stage 3 tool."""

    outcome: Outcome
    query: StatQuery | None = None
    result: StatResult = Field(default_factory=StatResult)
    parts: list[AnswerPart] = Field(default_factory=list)
    text: str = ""  # final rendered answer
    refusal: Refusal | None = None
    clarify_options: list[str] = Field(default_factory=list)
    default_applied: dict[str, str] = Field(default_factory=dict)  # e.g. {"period": "latest (2025)"}
    source: Source | None = None

    @model_validator(mode="after")
    def _contract(self) -> StatAnswer:
        missing = [p.fact_id for p in self.parts if isinstance(p, FactPart) and self.result.get(p.fact_id) is None]
        if missing:
            raise ValueError(f"parts reference unknown facts: {missing}")  # V1
        if (self.outcome == "refused") != (self.refusal is not None):
            raise ValueError("refusal is required exactly when outcome is 'refused'")
        if self.outcome == "clarify" and len(self.clarify_options) < 2:
            raise ValueError("outcome 'clarify' needs at least two clarify_options")
        out_of_scope = self.refusal is not None and self.refusal.reason == "out_of_scope"
        if self.source is None and not out_of_scope:
            raise ValueError("source is required unless the question is out of scope")
        return self
