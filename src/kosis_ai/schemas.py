"""Shared I/O contract for summarize_stat (stage 1), table search (stage 2) and the agent tools (stage 3).

Numbers live only in Fact. The LLM picks AnswerParts (fact ids + connecting text); code renders numbers.
Field names follow the golden set (eval/golden/stat_golden.jsonl) so golden facts load into the same Fact model.

v2 (2026-10-26, plans/T0-03.md): every model rejects unknown fields (a typo cannot pass silently), Fact checks its
id, value, raw, source_ref, derivation and round, and StatAnswer records who wrote the numbers (numbers_origin) and
whether code verified them (numbers_verified). Records written before v2 still load: the new fields default to
"not recorded" (None / False) and are only cross-checked when set.
Also v2: the chart drawn from the facts (Chart) carried inside StatAnswer.
v3 (T0-04): the calculation as blocks (PlanV3), recorded in StatQuery.plan; it replaced v2's operation over periods
(StatQuery.op / period_groups), which records written before v3 still carry and which are dropped on load.
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
# How a chart is drawn (kosis_chat/answer/chart.py): bars by name, bars in period order, a line over periods, a pie,
# (S6) one line per series over periods, and bars of several targets side by side at each period.
ChartKind = Literal["bar", "timeline", "line", "pie", "multi_line", "grouped_bar"]
# What a caveat warns about (S6, kosis_chat/answer/caveats.py): asked periods the table lacks, a condition the question
# left out and code filled in, a simple average of rates, a provisional value, a symbol cell shown, a series that
# changed its basis, an event start the model gave, and an alternative (another scenario, a look-alike label).
CaveatKind = Literal["missing_years", "default_applied", "simple_average", "provisional", "symbol_cell", "series_break",
                     "event_source", "alternative"]
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
CHART_KINDS = get_args(ChartKind)
CAVEAT_KINDS = get_args(CaveatKind)
VALUE_KINDS = get_args(ValueKind)
MAX_DIMS = 8  # dim1..dim8 columns in the Parquet files
MAX_ROUND = 10  # decimals a derived fact may keep (the golden set uses 0..5)
# Trailing "(unit)" in an item name or dim label, e.g. "출생건수 (명)", "출생아수(명)".
# ponytail: short no-space/no-digit token only, so "(출생시 기대여명)" or "(2012~)" are not units; a unit table if this misses too many.
_UNIT_SUFFIX = re.compile(r"\(([^()\s\d~]{1,5})\)\s*$")
_IDENT = re.compile(r"[A-Za-z_]\w*")
_ORG_ID = re.compile(r"\d{3}")
# A derivation is arithmetic over fact ids: + - * / and parentheses (what eval/grade.py recomputes), and ** with a
# constant exponent (v3, for an average annual growth rate: '((f2 / f1) ** (1 / 9) - 1) * 100').
_DERIVATION_NODES = (ast.Expression, ast.BinOp, ast.UnaryOp, ast.Constant, ast.Name, ast.Load)
_DERIVATION_OPS = (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.USub, ast.UAdd, ast.Pow)


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
            raise ValueError(f"derivation {expr!r} may only use fact ids, numbers and + - * / ** ( )")
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Pow) and any(
            isinstance(n, ast.Name) for n in ast.walk(node.right)
        ):
            raise ValueError(f"derivation {expr!r}: an exponent must be a number, not a fact")
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


# ---------------------------------------------------------------- v3 calculation plan (T0-04)
# A plan is a list of steps, each a small block whose inputs are earlier steps: select cells, aggregate a series,
# combine two values, rank candidates, find a period or a category by a condition, or repeat a sub-plan per target.
# New question shapes are new combinations of the same blocks, so this schema does not grow with them.
# present says how the answer is shown (the question type); the steps alone say what is computed.

AggFn = Literal["sum", "avg", "min", "max", "first", "last"]
# diff: a - b · ratio: a / b · pct_change: (a / b - 1) * 100 · pp_diff: a - b of two percentages, in %p ·
# share: a (summed when several cells) / b * 100 · per: a / b * scale ("per 1,000 people") ·
# cagr: ((a / b) ** (1 / years) - 1) * 100 · index: a / b * 100 (b as 100)
CombineFn = Literal["diff", "ratio", "pct_change", "pp_diff", "share", "per", "cagr", "index"]
Compare = Literal[">", ">=", "<", "<=", "=="]
PeriodKind = Literal["list", "range", "recent", "anchor", "span"]
SpanPart = Literal["year", "H1", "H2", "Q1", "Q2", "Q3", "Q4"]
STEP_SHAPES = ("cells", "scalar", "ranking", "found", "mapped")


class PeriodSpec(Contract):
    """Periods as the question names them; code resolves them to table periods (kosis_chat/compute/resolve.py).

    list: periods · range: start..end · recent: the last n periods that hold a number · anchor: an event or a period
    moved by offset (or every period from it when to_latest), e.g. 코로나 -1 · span: a part of a year on a monthly or
    quarterly table (a half, a quarter, the whole year)."""

    kind: PeriodKind
    periods: list[str] = Field(default_factory=list)
    start: str | None = None
    end: str | None = None
    n: int | None = Field(default=None, ge=1)
    event: str | None = None  # a name in kosis_chat/data/events.json
    period: str | None = None  # anchor on a period instead of an event
    offset: int = 0
    # anchor: what offset counts ("코로나 직전 해" = year, "직전 달" = month); None: the table's period unit
    unit: Literal["year", "month"] | None = None
    to_latest: bool = False
    year: str | None = None
    part: SpanPart | None = None

    @model_validator(mode="after")
    def _fields_of_kind(self) -> PeriodSpec:
        need = {"list": bool(self.periods), "range": bool(self.start and self.end), "recent": self.n is not None,
                "anchor": bool(self.event) != bool(self.period), "span": bool(self.year and self.part)}
        if not need[self.kind]:
            raise ValueError(f"period kind {self.kind!r} is missing its fields")
        return self


class TargetSpec(Contract):
    """Codes of one category (or of the items): listed, every child of a code, the total, or a label path
    ('서울특별시 > 중구') that code resolves to one code or to clarify options."""

    codes: list[str] = Field(default_factory=list)
    children_of: str | None = None
    total: bool = False
    path: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _one_way(self) -> TargetSpec:
        ways = [bool(self.codes), self.children_of is not None, self.total, bool(self.path)]
        if sum(ways) != 1:
            raise ValueError("a target names codes, children_of, total or a path: exactly one")
        return self


class _Step(Contract):
    id: str

    @field_validator("id")
    @classmethod
    def _identifier(cls, value: str) -> str:
        if not _fact_id_ok(value):
            raise ValueError(f"step id {value!r} must be a plain identifier")
        return value

    def inputs(self) -> list[str]:
        return []


class SelectStep(_Step):
    """Cells of one series or one cross-section: by source_ref (cells), or by item, categories and periods that
    code resolves first (table defaults to the question's table)."""

    op: Literal["select"] = "select"
    cells: list[str] = Field(default_factory=list)
    table: str | None = None
    item: TargetSpec | None = None
    dims: dict[str, TargetSpec] = Field(default_factory=dict)
    periods: PeriodSpec | None = None

    @model_validator(mode="after")
    def _cells_or_spec(self) -> SelectStep:
        spec = self.item is not None or bool(self.dims) or self.periods is not None
        if bool(self.cells) == spec:
            raise ValueError(f"select {self.id}: name cells by source_ref or by item/dims/periods, not both or neither")
        if len(self.cells) != len(set(self.cells)):
            raise ValueError(f"select {self.id}: a cell is listed twice")
        return self


class AggregateStep(_Step):
    op: Literal["aggregate"] = "aggregate"
    of: str
    fn: AggFn

    def inputs(self) -> list[str]:
        return [self.of]


class CombineStep(_Step):
    op: Literal["combine"] = "combine"
    a: str
    b: str
    fn: CombineFn
    # larger_first: a and b are swapped when b is larger (a gap with no named reference: larger minus smaller);
    # smaller_first: the other way (which of two declines is larger: the more negative change first)
    orient: Literal["fixed", "larger_first", "smaller_first"] = "fixed"
    optional: bool = False  # skip (no fact) when the value would mean nothing: a growth rate of a %, a zero base
    scale: float | None = Field(default=None, gt=0)  # per: e.g. 1000
    years: int | None = Field(default=None, ge=1)  # cagr: periods between b and a
    note: str | None = None

    @model_validator(mode="after")
    def _fn_fields(self) -> CombineStep:
        if (self.fn == "per") != (self.scale is not None):
            raise ValueError(f"combine {self.id}: scale goes with per, and per needs it")
        if (self.fn == "cagr") != (self.years is not None):
            raise ValueError(f"combine {self.id}: years goes with cagr, and cagr needs it")
        if self.a == self.b:
            raise ValueError(f"combine {self.id}: a and b are the same step")
        return self

    def inputs(self) -> list[str]:
        return [self.a, self.b]


class RankStep(_Step):
    """Candidates ranked; the first keep are reported. Candidates are cells (one cell each, or a share of each
    candidate's total) or the values of a map step (of), e.g. each province's growth rate."""

    op: Literal["rank"] = "rank"
    candidates: list[str] = Field(default_factory=list)  # source_refs
    totals: list[str] = Field(default_factory=list)  # share ranking: the total of each candidate, same order
    of: str | None = None  # a map step
    pick: Literal["max", "min"]
    keep: int = Field(default=2, ge=1)

    @model_validator(mode="after")
    def _totals_match(self) -> RankStep:
        if (len(self.candidates) >= 2) == (self.of is not None):
            raise ValueError(f"rank {self.id}: rank two or more candidates or a map step (of), exactly one")
        if self.candidates and len(self.candidates) == 1:
            raise ValueError(f"rank {self.id}: needs two or more candidates")
        if self.totals and len(self.totals) != len(self.candidates):
            raise ValueError(f"rank {self.id}: totals must match candidates")
        return self

    def inputs(self) -> list[str]:
        return [self.of] if self.of else []


class FindStep(_Step):
    """The periods (or categories) of a series whose value meets a condition against a number or a scalar step."""

    op: Literal["find"] = "find"
    of: str
    cmp: Compare
    value: float | None = None
    ref: str | None = None
    which: Literal["first", "last", "all"] = "first"

    @model_validator(mode="after")
    def _one_threshold(self) -> FindStep:
        if (self.value is None) == (self.ref is None):
            raise ValueError(f"find {self.id}: compare with value or with ref, exactly one")
        return self

    def inputs(self) -> list[str]:
        return [self.of, *([self.ref] if self.ref else [])]


class MapStep(_Step):
    """The same sub-plan once per code of one category; body selects with the placeholder code '*' on that axis."""

    op: Literal["map"] = "map"
    axis: str
    codes: list[str] = Field(min_length=2)
    body: list[SelectStep | AggregateStep | CombineStep] = Field(min_length=1)
    result: str  # the body step whose value is kept per code

    @model_validator(mode="after")
    def _result_in_body(self) -> MapStep:
        ids = [s.id for s in self.body]
        if self.result not in ids:
            raise ValueError(f"map {self.id}: result {self.result!r} is not a body step")
        starred = [s for s in self.body if isinstance(s, SelectStep) and (t := s.dims.get(self.axis)) and t.codes == ["*"]]
        if not starred:
            raise ValueError(f"map {self.id}: a body select must name the mapped axis {self.axis!r} as codes ['*']")
        _check_order(self.body, f"map {self.id}")
        return self


Step = Annotated[SelectStep | AggregateStep | CombineStep | RankStep | FindStep | MapStep, Field(discriminator="op")]


def _check_order(steps: list, where: str) -> dict[str, str]:
    """Unique ids, every input defined earlier; returns step id -> shape."""
    shapes: dict[str, str] = {}
    for s in steps:
        if s.id in shapes:
            raise ValueError(f"{where}: step id {s.id!r} is used twice")
        missing = [i for i in s.inputs() if i not in shapes]
        if missing:
            raise ValueError(f"{where}: step {s.id} uses {missing}, which are not earlier steps")
        shapes[s.id] = _shape(s, shapes)
    return shapes


def _shape(step, shapes: dict[str, str]) -> str:
    """What a step yields, checking what its inputs yield: cells (one or more), a scalar, a ranking, found
    periods, or one value per mapped code."""
    if isinstance(step, SelectStep):
        return "cells"
    if isinstance(step, AggregateStep):
        if shapes[step.of] not in ("cells", "mapped"):
            raise ValueError(f"aggregate {step.id}: needs cells or a map, got {shapes[step.of]}")
        return "scalar"
    if isinstance(step, CombineStep):
        bad = [i for i in (step.a, step.b) if shapes[i] not in ("cells", "scalar")]
        if bad:
            raise ValueError(f"combine {step.id}: inputs must be cells or scalars, not {[shapes[i] for i in bad]}")
        return "scalar"
    if isinstance(step, FindStep):
        if shapes[step.of] != "cells" or (step.ref and shapes[step.ref] not in ("cells", "scalar")):
            raise ValueError(f"find {step.id}: needs cells to search and a scalar threshold")
        return "found"
    if isinstance(step, RankStep):
        if step.of and shapes[step.of] != "mapped":
            raise ValueError(f"rank {step.id}: of must be a map step, got {shapes[step.of]}")
        return "ranking"
    return "mapped"


class PlanV3(Contract):
    """What code computes for one question (v3). show lists the steps the answer cites, in order."""

    plan_version: Literal["3"] = "3"
    steps: list[Step] = Field(min_length=1)
    show: list[str] = Field(min_length=1)
    present: QuestionType | Literal["table"]
    note: str | None = None  # what the answer should say about the choice (missing years, a default period)
    based_on: str | None = None  # a follow-up question's earlier plan; not supported yet

    @model_validator(mode="after")
    def _wired(self) -> PlanV3:
        shapes = _check_order(self.steps, "plan")
        unknown = [i for i in self.show if i not in shapes]
        if unknown:
            raise ValueError(f"show names steps that do not exist: {unknown}")
        if len(self.show) != len(set(self.show)):
            raise ValueError("show lists a step twice")
        if self.based_on is not None:
            raise ValueError("based_on (follow-up questions) is not supported yet")
        inner = [b.id for s in self.steps if isinstance(s, MapStep) for b in s.body]
        clash = sorted(set(inner) & set(shapes) | {i for i in inner if inner.count(i) > 1})
        if clash:
            raise ValueError(f"map body step ids must differ from every other step id: {clash}")
        if self.present == "extreme" and not any(shapes[i] == "ranking" for i in self.show):
            raise ValueError("an extreme answer shows a rank step")
        return self

    def step(self, step_id: str):
        return next(s for s in self.steps if s.id == step_id)


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
    plan: PlanV3 | None = None  # v3: the calculation as blocks (when code computed the answer)

    @model_validator(mode="before")
    @classmethod
    def _drop_v2_operation(cls, data: object) -> object:
        """Records written before v3 carry v2's op / period_groups; the plan replaced them, so they are dropped."""
        if isinstance(data, dict) and ("op" in data or "period_groups" in data):
            return {k: v for k, v in data.items() if k not in ("op", "period_groups")}
        return data


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


class Caveat(Contract):
    """A word of care shown with the answer (S6). text is code's sentence; it names periods but no other numbers
    (a number belongs in a fact); fact_ids are the facts it is about."""

    kind: CaveatKind
    text: str = Field(min_length=1)
    fact_ids: list[str] = Field(default_factory=list)


class ClarifyChoice(Contract):
    """One option of an ask-back with the code it stands for (S6); clarify_options keeps the labels."""

    code: str = Field(min_length=1)
    label: str = Field(min_length=1)
    axis: str = Field(min_length=1)


class TableRow(Contract):
    label: str = Field(min_length=1)
    fact_ids: list[str | None] = Field(min_length=1)  # one per column; None where the row has no fact


class TablePart(Contract):
    """The answer's facts as a table (S6): rows and columns hold fact ids, the numbers stay in the facts."""

    kind: Literal["table"] = "table"
    columns: list[str] = Field(min_length=1)
    rows: list[TableRow] = Field(min_length=1)

    @model_validator(mode="after")
    def _shape(self) -> TablePart:
        bad = [r.label for r in self.rows if len(r.fact_ids) != len(self.columns)]
        if bad:
            raise ValueError(f"table rows need one cell per column: {bad}")
        return self

    def fact_ids(self) -> list[str]:
        return [i for r in self.rows for i in r.fact_ids if i is not None]


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


_NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?")
_PERIOD_AFTER = re.compile(r"\s*(?:년|월|분기|반기)")


def _caveat_numbers(text: str, result: StatResult) -> list[str]:
    """Numbers in a caveat that are neither a period ('2018년', '3월', '1분기'), nor part of a name ('코로나19'),
    nor in a fact's labels ('65세')."""
    labels = {n for f in result.facts for label in (f.dims or {}).values() for n in _NUMBER.findall(label)}
    return [m.group(0) for m in _NUMBER.finditer(text)
            if not _PERIOD_AFTER.match(text, m.end()) and not (m.start() and text[m.start() - 1].isalpha())
            and m.group(0) not in labels]


class StatAnswer(Contract):
    """Return value of summarize_stat (T1-11) and of the stage 3 tool."""

    outcome: Outcome
    query: StatQuery | None = None
    result: StatResult = Field(default_factory=StatResult)
    parts: list[AnswerPart] = Field(default_factory=list)
    text: str = ""  # final rendered answer
    refusal: Refusal | None = None
    clarify_options: list[str] = Field(default_factory=list)
    clarify_choices: list[ClarifyChoice] = Field(default_factory=list)  # S6: the options with their codes
    default_applied: dict[str, str] = Field(default_factory=dict)  # e.g. {"period": "latest (2025)"}
    source: Source | None = None
    # v2 record of trust: see NumbersOrigin. numbers_verified is True only when code checked every number in the
    # parts against result (kosis_chat/answer/verify.py); defaults_verified only when the calculation itself used
    # default_applied (a chatbot answer records the rules' plan without proof the model followed it).
    numbers_origin: NumbersOrigin | None = None
    numbers_verified: bool = False
    defaults_verified: bool = False
    chart: Chart | None = None  # v2: drawn from result by code; only an answered answer with code numbers has one
    # S6: words of care, the facts shown beside the answer (not asked for, never computed with), and a table of facts.
    caveats: list[Caveat] = Field(default_factory=list)
    context_fact_ids: list[str] = Field(default_factory=list)
    table: TablePart | None = None

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
        self._s6_contract()
        return self

    def _s6_contract(self) -> None:
        if self.clarify_choices and [c.label for c in self.clarify_choices] != self.clarify_options:
            raise ValueError("clarify_choices labels must equal clarify_options")
        named = {"caveats": [i for c in self.caveats for i in c.fact_ids], "context_fact_ids": self.context_fact_ids,
                 "table": self.table.fact_ids() if self.table else []}
        for where, ids in named.items():
            unknown = [i for i in ids if self.result.get(i) is None]
            if unknown:
                raise ValueError(f"{where} reference unknown facts: {unknown}")
        stray = [c.text for c in self.caveats if _caveat_numbers(c.text, self.result)]
        if stray:
            raise ValueError(f"caveat text names numbers that are not periods: {stray}")
        if (self.context_fact_ids or self.table) and self.outcome != "answered":
            raise ValueError("context facts and a table belong to an answered answer")
