"""Shared I/O contract for summarize_stat (stage 1), table search (stage 2) and the agent tools (stage 3).

Numbers live only in Fact. The LLM picks AnswerParts (fact ids + connecting text); code renders numbers.
Field names follow the golden set (eval/golden/stat_golden.jsonl) so golden facts load into the same Fact model.

v2 (2026-10-26, plans/T0-03.md): every model rejects unknown fields (a typo cannot pass silently), Fact checks its
id, value, raw, source_ref, derivation and round, and StatAnswer records who wrote the numbers (numbers_origin) and
whether code verified them (numbers_verified). Records written before v2 still load: the new fields default to
"not recorded" (None / False) and are only cross-checked when set.
Also v2: an operation over periods (Op: an average, a sum, two periods compared) that StatQuery records and the
engine computes as derived facts, and the chart drawn from the facts (Chart) carried inside StatAnswer.
"""

from __future__ import annotations

import ast
import keyword
import math
import re
from typing import Annotated, Literal, get_args

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

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
# An operation over periods, on top of the question type (v2, T1-25 complex questions):
# avg / sum: one period group aggregated to one number; pavg / psum: two period groups, each aggregated, compared;
# vsa: one period against the average of a period group; shr: a share at two periods and its change in %p.
Op = Literal["avg", "sum", "pavg", "psum", "vsa", "shr"]
# How a chart is drawn (kosis_chat/answer/chart.py): bars by name, bars in period order, a line over periods, a pie.
ChartKind = Literal["bar", "timeline", "line", "pie"]
Outcome = Literal["answered", "refused", "clarify"]
# Who wrote the numbers the answer shows (v2). code: every number is a Fact rendered by code (FactPart);
# model: the model wrote the sentence, so its numbers are unverified; none: a sentence code wrote without citing
# facts (a rule refusal or ask-back). None on a record written before v2.
NumbersOrigin = Literal["code", "model", "none"]
# value_kind in data/values/*.parquet: "ok" has a number, every other kind is a symbol kept in value_raw.
ValueKind = Literal["ok", "dash", "x_secret", "blank", "ellipsis", "asterisk", "other_text"]

QUESTION_TYPES = get_args(QuestionType)
QUESTION_KINDS = get_args(QuestionKind)
REFUSAL_REASONS = get_args(RefusalReason)
NUMBERS_ORIGINS = get_args(NumbersOrigin)
OPS = get_args(Op)
CHART_KINDS = get_args(ChartKind)
VALUE_KINDS = get_args(ValueKind)
MAX_DIMS = 8  # dim1..dim8 columns in the Parquet files
MAX_ROUND = 10  # decimals a derived fact may keep (the golden set uses 0..5)
# Trailing "(unit)" in an item name or dim label, e.g. "출생건수 (명)", "출생아수(명)".
# ponytail: short no-space/no-digit token only, so "(출생시 기대여명)" or "(2012~)" are not units; a unit table if this misses too many.
_UNIT_SUFFIX = re.compile(r"\(([^()\s\d~]{1,5})\)\s*$")
_IDENT = re.compile(r"[A-Za-z_]\w*")
_ORG_ID = re.compile(r"\d{3}")
# A derivation is arithmetic over fact ids: + - * / and parentheses (what eval/grade.py recomputes).
_DERIVATION_NODES = (ast.Expression, ast.BinOp, ast.UnaryOp, ast.Constant, ast.Name, ast.Load)
_DERIVATION_OPS = (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.USub, ast.UAdd)


class Contract(BaseModel):
    """Base of every shared model: unknown fields are an error, never dropped."""

    model_config = ConfigDict(extra="forbid")


def derivation_names(expr: str) -> list[str]:
    """Fact ids a derivation refers to, in order of first appearance."""
    return list(dict.fromkeys(_IDENT.findall(expr)))


def check_derivation(expr: str) -> list[str]:
    """Validate a derivation's syntax and return the fact ids it uses (ValueError when it is not plain arithmetic)."""
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"derivation {expr!r} is not an expression: {exc.msg}") from None
    for node in ast.walk(tree):
        if not isinstance(node, _DERIVATION_NODES + _DERIVATION_OPS):
            raise ValueError(f"derivation {expr!r} may only use fact ids, numbers and + - * / ( )")
        if isinstance(node, ast.Constant) and (isinstance(node.value, bool) or not isinstance(node.value, (int, float))):
            raise ValueError(f"derivation {expr!r} may only use numeric constants")
    names = [node.id for node in ast.walk(tree) if isinstance(node, ast.Name)]
    if not names:
        raise ValueError(f"derivation {expr!r} refers to no fact")
    return list(dict.fromkeys(names))


def _fact_id_ok(fact_id: str) -> bool:
    """A fact id is used as a name inside derivations, so it must be a plain identifier ('f1', 'a2')."""
    return fact_id.isidentifier() and not keyword.iskeyword(fact_id)


class Source(Contract):
    org_id: str = Field(pattern=r"^\d{3}$")  # KOSIS agency code, e.g. "101"
    tbl_id: str = Field(min_length=1)
    survey_nm: str = Field(min_length=1)
    tbl_nm: str = Field(min_length=1)
    url: str | None = None

    def citation(self) -> str:
        """'KOSIS / survey / table (url)' — built by code, never by the LLM."""
        text = f"KOSIS / {self.survey_nm} / {self.tbl_nm}"
        return f"{text} ({self.url})" if self.url else text


class TableInfo(Contract):
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


class Dim(Contract):
    name: str  # e.g. "행정구역별"
    code: str  # e.g. "31"
    label: str  # e.g. "경기도"


class Cell(Contract):
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
        """Build from a flat Parquet row (dim1_name, dim1_code, dim1_label, ... dim8_*).

        Only the contract's own columns are read: an extra column in a Parquet file is not a contract violation,
        a wrong value in a known column is."""
        dims = [
            Dim(name=row[f"dim{i}_name"], code=row[f"dim{i}_code"], label=row[f"dim{i}_label"])
            for i in range(1, MAX_DIMS + 1)
            if row.get(f"dim{i}_code") is not None
        ]
        flat = {k: v for k, v in row.items() if k in cls.model_fields and k != "dims"}
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


class StatQuery(Contract):
    """Question analysis result (T1-06)."""

    question: str
    type: QuestionType | None = None  # None for causal or out-of-scope questions
    item: str | None = None
    periods: list[str] = Field(default_factory=list)
    filters: dict[str, str] = Field(default_factory=dict)
    ambiguity: str | None = None  # what is unclear; None when the question is fully specified
    # Optional, for questions one item and one label per category cannot state (C pipeline, 2026-10-24):
    items: list[str] = Field(default_factory=list)  # items asked together, e.g. 1인·2인·3인 for "3인 이하 가구"
    targets: list[dict[str, str]] = Field(default_factory=list)  # compare: the two sides, e.g. [{"행정구역별": "부산광역시"}, ...]
    reference: int | None = None  # compare: index in targets of the side named as the reference ("전국보다")
    over: dict[str, list[str]] = Field(default_factory=dict)  # extreme: category ranked over -> its labels
    pick: Literal["max", "min"] | None = None  # extreme: highest or lowest
    # v2: the operation over periods, and the period groups it works on (table periods, in question order),
    # e.g. pavg over [["2010", ..., "2014"], ["2015", ..., "2019"]]; periods holds every period of every group.
    op: Op | None = None
    period_groups: list[list[str]] = Field(default_factory=list)

    @model_validator(mode="after")
    def _groups_in_periods(self) -> StatQuery:
        if self.period_groups and not self.op:
            raise ValueError("period_groups need an op")
        if any(not g for g in self.period_groups):
            raise ValueError("a period group cannot be empty")
        outside = [p for g in self.period_groups for p in g if p not in self.periods]
        if outside:
            raise ValueError(f"period_groups name periods that are not in periods: {outside}")
        return self


class Fact(Contract):
    """One number: a source cell, a value computed from other facts, or a note."""

    id: str
    value: int | float | None = None
    unit: str | None = None
    period: str | None = None
    dims: dict[str, str] | None = None
    source_ref: str | None = None  # Cell.ref() of the source cell, e.g. "DT_1B8000H|T10|31|2025"
    raw: str | None = None  # original symbol when value is None, e.g. "-" or "X"
    derivation: str | None = None  # formula over other fact ids, e.g. "(f2 / f1 - 1) * 100"
    round: int | None = Field(default=None, ge=0, le=MAX_ROUND)
    measure_kind: MeasureKind | None = None
    note: str | None = None

    @field_validator("value", mode="before")
    @classmethod
    def _not_a_bool(cls, value):
        if isinstance(value, bool):  # bool is an int to Python; pydantic would quietly read True as 1
            raise ValueError("value must be a number, not a bool")
        return value

    @model_validator(mode="after")
    def _traceable(self) -> Fact:
        if not _fact_id_ok(self.id):
            raise ValueError(f"fact id {self.id!r} must be a plain identifier such as 'f1'")
        if self.value is not None and not math.isfinite(self.value):
            raise ValueError(f"fact {self.id}: value must be a finite number, got {self.value!r}")
        if self.value is not None and not (self.source_ref or self.derivation):
            raise ValueError(f"fact {self.id}: a value needs a source_ref or a derivation")
        if not (self.source_ref or self.derivation or self.note):
            raise ValueError(f"fact {self.id}: needs a source_ref, a derivation or a note")
        if self.raw is not None and self.value is not None:
            raise ValueError(f"fact {self.id}: raw is the symbol of a cell without a value, so value must be None")
        if self.unit is not None and not self.unit.strip():
            raise ValueError(f"fact {self.id}: unit must be None when unknown, not blank")
        if self.source_ref is not None:
            parts = self.source_ref.split("|")
            if len(parts) < 3 or not (parts[0] and parts[1] and parts[-1]):
                raise ValueError(f"fact {self.id}: source_ref {self.source_ref!r} is not 'tbl_id|item_id|dim codes...|period'")
        if self.derivation is not None:
            if self.round is None:
                raise ValueError(f"fact {self.id}: a derivation needs round")
            if self.id in check_derivation(self.derivation):
                raise ValueError(f"fact {self.id}: a derivation cannot refer to itself")
        return self


class StatResult(Contract):
    """Facts computed by code for one question (T1-07)."""

    facts: list[Fact] = Field(default_factory=list)

    @model_validator(mode="after")
    def _unique_ids(self) -> StatResult:
        ids = [f.id for f in self.facts]
        dupes = sorted({i for i in ids if ids.count(i) > 1})
        if dupes:
            raise ValueError(f"duplicate fact ids: {dupes}")
        unknown = {f.id: sorted(set(derivation_names(f.derivation)) - set(ids)) for f in self.facts if f.derivation}
        unknown = {k: v for k, v in unknown.items() if v}
        if unknown:
            raise ValueError(f"derivations reference unknown facts: {unknown}")
        return self

    def get(self, fact_id: str) -> Fact | None:
        return next((f for f in self.facts if f.id == fact_id), None)


class Refusal(Contract):
    """Why the question cannot be answered (T1-08). Graded correct when reason is in the golden allowed_reasons."""

    reason: RefusalReason
    message: str = Field(min_length=1)
    available: str | None = None  # what can be answered instead, e.g. the latest published year


class FactPart(Contract):
    kind: Literal["fact"] = "fact"
    fact_id: str


class TextPart(Contract):
    kind: Literal["text"] = "text"
    text: str = Field(min_length=1)  # connecting text; numeric claims here are rejected by the verifier (V4, T1-10)


AnswerPart = Annotated[FactPart | TextPart, Field(discriminator="kind")]


class Chart(Contract):
    """A chart code drew from the answer's facts (v2). spec is Vega-Lite v5 JSON; fact_ids names the facts whose
    values the chart shows (StatAnswer checks they exist). Rows drawn from the table for context only (the other
    periods of a series, the other candidates of a ranking) are not facts and are not listed."""

    kind: ChartKind
    spec: dict
    fact_ids: list[str] = Field(min_length=1)

    @model_validator(mode="after")
    def _spec_has_data(self) -> Chart:
        if not isinstance(self.spec.get("data"), dict) or not isinstance(self.spec["data"].get("values"), list):
            raise ValueError("chart spec needs inline data: spec['data']['values'] must be a list")
        if len(self.fact_ids) != len(set(self.fact_ids)):
            raise ValueError("fact_ids must be unique")
        return self


class StatAnswer(Contract):
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
    # v2 record of trust: see NumbersOrigin. numbers_verified is True only when code checked every number in the
    # parts against result (kosis_chat/answer/verify.py); defaults_verified only when the calculation itself used
    # default_applied (a chatbot answer records the rules' plan without proof the model followed it).
    numbers_origin: NumbersOrigin | None = None
    numbers_verified: bool = False
    defaults_verified: bool = False
    chart: Chart | None = None  # v2: drawn from result by code; only an answered answer with code numbers has one

    @model_validator(mode="after")
    def _contract(self) -> StatAnswer:
        cited = [p.fact_id for p in self.parts if isinstance(p, FactPart)]
        missing = [i for i in cited if self.result.get(i) is None]
        if missing:
            raise ValueError(f"parts reference unknown facts: {missing}")  # V1
        if (self.outcome == "refused") != (self.refusal is not None):
            raise ValueError("refusal is required exactly when outcome is 'refused'")
        if self.outcome == "clarify" and len(self.clarify_options) < 2:
            raise ValueError("outcome 'clarify' needs at least two clarify_options")
        if self.outcome == "answered" and not self.parts:
            raise ValueError("an answered answer needs parts")
        if self.outcome != "answered" and cited:
            raise ValueError(f"a {self.outcome} answer cannot cite facts: {cited}")
        out_of_scope = self.refusal is not None and self.refusal.reason == "out_of_scope"
        if self.source is None and not out_of_scope:
            raise ValueError("source is required unless the question is out of scope")
        blank = [k for k, v in self.default_applied.items() if not k.strip() or not v.strip()]
        if blank:
            raise ValueError(f"default_applied needs a non-blank key and value: {blank!r}")
        if self.numbers_origin == "code" and (self.outcome != "answered" or not cited):
            raise ValueError("numbers_origin 'code' means an answered answer that cites at least one fact")
        if self.numbers_origin in ("model", "none") and cited:
            raise ValueError(f"numbers_origin {self.numbers_origin!r} cannot cite facts (numbers would come from code)")
        if self.numbers_verified and self.numbers_origin != "code":
            raise ValueError("only numbers code wrote can be verified (numbers_origin 'code')")
        if self.defaults_verified and self.numbers_origin != "code":
            raise ValueError("default_applied is verified only when the calculation used it (numbers_origin 'code')")
        if self.chart is not None:
            if self.outcome != "answered" or self.numbers_origin != "code":
                raise ValueError("a chart belongs to an answered answer whose numbers code computed")
            unknown = [i for i in self.chart.fact_ids if self.result.get(i) is None]
            if unknown:
                raise ValueError(f"chart.fact_ids reference unknown facts: {unknown}")
        return self
