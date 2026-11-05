"""Reading records saved under an older contract (T1-28 S5): the one place that knows their shape.

StatAnswer lost its `chart` field in T1-28 S5 (the chart is the module's StatOutput.chart, a ChartSpec). A results.jsonl
written before that has `stat_answer.chart`: None, or the old Chart {kind, spec (Vega-Lite), fact_ids}. Readers of saved
runs (eval/grade_run.py, eval/recheck_contract.py, eval/harness.py, Phase 2 evals/runners/stat.py) load the answer with
load_stat_answer; the old chart, when they want it, is legacy_chart. StatAnswer itself still drops that one key on
input (its before-validator calls without_legacy_chart) so Phase 2 readers that call StatAnswer.model_validate keep
working until they move here; any other unknown key is still an error (extra="forbid").
"""

from __future__ import annotations

from typing import Any

from kosis_ai.schemas import StatAnswer

LEGACY_CHART_KEYS = frozenset({"kind", "spec", "fact_ids"})  # the v2 Chart model's fields


def _is_legacy_chart(value: Any) -> bool:
    """None, or a dict the v2 Chart model accepted: exactly kind (str), spec (dict with inline data.values, a list) and
    fact_ids (a non-empty list of distinct str)."""
    if value is None:
        return True
    if not (isinstance(value, dict) and set(value) == LEGACY_CHART_KEYS and isinstance(value["kind"], str)
            and isinstance(value["spec"], dict) and isinstance(value["fact_ids"], list)):
        return False
    data, ids = value["spec"].get("data"), value["fact_ids"]
    return (isinstance(data, dict) and isinstance(data.get("values"), list) and bool(ids)
            and all(isinstance(i, str) for i in ids) and len(ids) == len(set(ids)))


def _old_contract_ok(data: dict, chart: dict | None) -> bool:
    """The rule the v2 contract held for StatAnswer.chart: only an answered answer whose numbers code computed has one,
    and its fact_ids name facts of the result. A record that broke it failed to load before S5 and still does."""
    if chart is None:
        return True
    if data.get("outcome") != "answered" or data.get("numbers_origin") != "code":
        return False
    result = data.get("result") or {}
    ids = {f.get("id") for f in (result.get("facts") or []) if isinstance(f, dict)} if isinstance(result, dict) else set()
    return all(i in ids for i in chart["fact_ids"])


def without_legacy_chart(data: Any) -> Any:
    """The record without its pre-S5 `chart` key (None or an old Chart dict that the old contract accepted); anything
    else is returned unchanged, so a ChartSpec, a malformed chart or a chart on a refused answer still fails as an
    unknown field."""
    if isinstance(data, dict) and "chart" in data and _is_legacy_chart(data["chart"]) and _old_contract_ok(data, data["chart"]):
        return {k: v for k, v in data.items() if k != "chart"}
    return data


def legacy_chart(data: Any) -> dict | None:
    """The old Chart dict {kind, spec, fact_ids} of a pre-S5 stat_answer record, or None."""
    if isinstance(data, dict) and isinstance(data.get("chart"), dict) and _is_legacy_chart(data["chart"]):
        return data["chart"]
    return None


def load_stat_answer(data: dict) -> StatAnswer:
    """A saved stat_answer (any contract version since v2) as today's StatAnswer."""
    return StatAnswer.model_validate(without_legacy_chart(data))


__all__ = ["LEGACY_CHART_KEYS", "legacy_chart", "load_stat_answer", "without_legacy_chart"]
