"""The import surface other code relies on (docs/compat-surface.md), so the T1-28 file moves cannot drop a name.

Each entry is a module path, the names read from it (private ones included: `import *` would not re-export them)
and, for callables, the parameters callers pass. Moving a file keeps a re-export at the old path that names each
symbol explicitly. Only importing and inspecting: no data, model or Phase 2 checkout needed.
"""

from __future__ import annotations

import importlib
import importlib.util
import inspect
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

# (module, names) -- who uses them is listed in docs/compat-surface.md.
SURFACE: tuple[tuple[str, tuple[str, ...]], ...] = (
    # Phase 2 (KOSIS-ai-RAG)
    ("kosis_chat.api", ("summarize_stat", "table_path", "_build", "_cached")),
    ("kosis_chat.config", ("Settings", "load_settings", "load_env", "get_secret", "resolve_path")),
    ("kosis_chat.contract", ("to_stat_answer", "org_id_from_table_path")),
    ("kosis_chat.pipeline", ("VERSION", "StatPipeline")),
    ("kosis_chat.llm.model", ("LLMClient", "LLMResult", "MockClient", "HCXClient", "build_client",
                              "STATUS_OK", "STATUS_API_ERROR")),
    ("kosis_chat.data.loader", ("sha256_of", "load_table", "infer_unit", "UNKNOWN_UNIT", "KNOWN_UNITS",
                                "UNIT_OVERRIDES")),
    ("kosis_ai.schemas", ("TableInfo", "MAX_DIMS", "Source", "Cell", "Fact", "StatAnswer", "StatResult", "StatQuery",
                          "TextPart", "FactPart", "Refusal", "QuestionKind", "QuestionType", "RefusalReason")),
    ("kosis_ai.normalize", ("_meta_unit",)),
    ("kosis_ai.paths", ("data_dir", "workspace_root", "ENV_VAR")),
    ("eval.golden", ("load_items",)),
    ("eval.grade_run", ("main",)),
    ("eval.scorer", ("read_jsonl",)),
    # Phase 1 itself (code and tests) reading private names across modules
    ("kosis_chat.compute.engine", ("_read", "_Read", "CalcRefused", "Computed", "TermResult")),
    ("kosis_chat.compute.semantics", ("_whole_label", "index_of")),
    ("kosis_chat.data.events", ("event_key", "load_events", "EVENTS_PATH")),
    # T1-28 S2: data/ split into prepare/ (convert, clean, schemas, index), adapters/parquet.py and shared/events.py;
    # the old modules re-export them by name
    ("kosis_chat.data.schema", ("TableInput", "TableMeta", "Cell", "Problem", "SchemaError", "validate",
                                "validate_meta_dict", "REQUIRED_META_FIELDS", "VALUE_FLAGS")),
    ("kosis_chat.data.preprocessing", ("clean", "clean_cell", "parse_value", "KIND_FLAGS", "MISSING_MARKS")),
    ("kosis_chat.compute.resolve", ("load_events",)),
    ("kosis_chat.judge.question", ("build_index", "index_of", "cell_measure", "is_total", "ITEM_AXIS", "COUNT_KINDS",
                                   "SCENARIO_AXIS", "CAUSAL_WORDS", "asked_periods", "match_labels", "squash")),
    # T1-28 S1b: node-boundary models moved to <stage>/schemas.py; the old modules re-export them by name
    ("kosis_chat.judge.answerability", ("Verdict", "Resolved", "refuse", "judge", "resolve")),
    ("kosis_chat.judge.question", ("Periods", "Match", "Parsed", "in_range", "TableIndex", "Axis")),
    ("kosis_chat.compute", ("Built", "Computed", "CalcRefused", "build", "compute")),
    ("kosis_chat.compute.analyze", ("Plan", "Unresolved", "PERIOD_AXIS")),
    ("kosis_chat.compute.build", ("Built", "TermSteps", "Pattern", "Agg")),
    ("kosis_chat.compute.context", ("Alternatives",)),
    ("kosis_chat.compute.resolve", ("Resolved", "NeedsClarify")),
    ("kosis_chat.llm.analyze_plan", ("Placed",)),
    ("kosis_chat.llm.compose", ("ComposeOutput",)),
    # T1-28 S2b: unit rules moved to kosis_ai/units.py, shared helpers to kosis_chat/shared/; old names re-exported
    ("kosis_ai.units", ("infer_unit", "unit_in_parens", "UNKNOWN_UNIT", "INFER_KNOWN_UNITS", "INFER_UNIT_OVERRIDES",
                        "resolve_unit", "measure_kind", "known_unit", "paren_unit", "override_unit", "keyword_kind",
                        "RESOLVE_KNOWN_UNITS", "RESOLVE_UNIT_OVERRIDES", "MEASURE_OVERRIDES", "unit_or_suffix",
                        "SUFFIX_UNIT", "PAREN", "UNIT_PAREN")),
    ("kosis_ai.normalize", ("meta_unit", "resolve_unit", "measure_kind", "KNOWN_UNITS", "UNIT_OVERRIDES",
                            "MEASURE_OVERRIDES", "VALUE_LABELS", "LABEL_COLUMNS", "_known", "_paren_unit",
                            "_override_unit", "_keyword_kind", "_PAREN", "_NOT_COUNT", "_KIND_KEYWORDS",
                            "_RATIO_UNITS", "_STOCK_UNITS", "_FLOW_UNITS", "_UNIT_SUFFIX")),
    ("kosis_ai.schemas", ("NUMBER", "_NUMBER", "_UNIT_SUFFIX")),
    ("kosis_chat.prepare.convert", ("infer_unit", "_unit_in_parens", "_PAREN", "UNKNOWN_UNIT", "KNOWN_UNITS",
                                    "UNIT_OVERRIDES")),
    ("kosis_chat.answer.render", ("SYMBOL_MEANING", "format_number", "format_fact", "is_change", "direction",
                                  "derived_kind", "period_text", "span_text", "item_label", "fact_label", "josa",
                                  "render_parts", "default_sentence", "final_text", "_number", "_UNIT_SUFFIX",
                                  "_CHANGE", "_KIND_WORDS", "_SUM", "_AVG", "_HOUSEHOLD")),
    ("kosis_chat.judge.messages", ("REFUSAL_MESSAGES", "refusal_message", "josa", "with_josa", "default_note",
                                   "_default_text", "_DEFAULT_KEYS", "_LATEST", "_RECENT", "_PREVIOUS", "_READ_LAST",
                                   "_PAIRS", "_PLACEHOLDER_JOSA", "_FORM", "_Fields")),
    ("kosis_chat.judge.answerability", ("_UNIT_SUFFIX", "_period_text", "_range_text", "defaults")),
    ("kosis_chat.judge.question", ("LABEL_ALIASES", "_YEAR", "_RANGE", "_SINCE", "_RECENT_N", "_COHORT",
                                   "_REGION_SUFFIX")),
    ("kosis_chat.data.scope", ("LABEL_ALIASES", "_squash", "_YEAR", "_YEAR_RANGE", "_YEAR_SINCE", "_RECENT_N",
                               "_UNIT_PAREN", "_REGION_SUFFIX", "question_periods", "auto_scope")),
    ("kosis_chat.data.serializer", ("display_value", "_TRAILING_ZEROS")),
    ("kosis_chat.answer.verify", ("NUMBER",)),
    ("kosis_chat.answer.template", ("_DERIVED_IDS",)),
    ("kosis_chat.answer.caveats", ("_EVENT_MODEL", "_PREFIX", "_from_caution")),
    ("kosis_chat.compute.analyze", ("_LATEST", "_RECENT", "_PREVIOUS")),
    ("kosis_chat.compute.roles", ("_PROVINCE", "is_province")),
    ("kosis_chat.contract", ("to_source", "render_text", "strip_model_source")),
)

# Callables whose parameters callers pass by name or position: (module, name, parameter names in order).
SIGNATURES: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("kosis_chat.api", "summarize_stat", ("question", "table_id", "settings", "client", "retry_interval")),
    ("kosis_chat.api", "table_path", ("table_id",)),
    ("kosis_chat.api", "_build", ("path", "settings", "client", "retry_interval")),  # Phase 2 evals/trace.py:61
    ("kosis_chat.contract", "to_stat_answer", ("question", "answer", "org_id")),
    ("kosis_chat.config", "load_settings", ("path", "overrides")),
    ("kosis_chat.config", "load_env", ("path",)),
    ("kosis_chat.config", "get_secret", ("name",)),
    ("kosis_chat.llm.model", "build_client", ("settings",)),
    ("kosis_chat.data.loader", "sha256_of", ("path",)),
    ("kosis_chat.data.loader", "load_table", ("path",)),
    ("kosis_chat.data.loader", "infer_unit", ("unit", "item", "dim_labels")),  # tests/test_data.py
    ("kosis_chat.data.preprocessing", "clean", ("table",)),
    ("kosis_chat.data.schema", "validate", ("table",)),
    ("kosis_chat.judge.question", "build_index", ("table",)),
    ("kosis_chat.pipeline", "StatPipeline", ("table", "client", "settings", "org_id", "retry_interval")),
    ("kosis_ai.normalize", "_meta_unit", ("table_id", "data_dir")),
    ("kosis_ai.normalize", "meta_unit", ("table_id", "data_dir")),  # T1-28 S2b public alias (Phase 2 metadata.py)
    ("kosis_ai.normalize", "resolve_unit", ("unit", "item_name", "dim_labels", "meta_unit")),
    ("kosis_ai.normalize", "measure_kind", ("tbl_id", "unit", "item_name", "dim_labels")),
    ("kosis_ai.units", "infer_unit", ("unit", "item", "dim_labels")),
    ("kosis_ai.units", "unit_or_suffix", ("unit", "texts")),
    ("kosis_chat.answer.render", "default_sentence", ("applied",)),
    ("kosis_chat.answer.render", "final_text", ("body", "source", "applied", "notes")),
    ("eval.golden", "load_items", ("path",)),
)
KEYWORD_ONLY = {("kosis_chat.api", "summarize_stat"): {"settings", "client", "retry_interval"},
                ("kosis_chat.contract", "to_stat_answer"): {"org_id"}}

# Attributes callers read from returned objects.
SETTINGS_FIELDS = ("backend", "serialization", "model_name", "temperature", "max_tokens", "timeout_seconds",
                   "base_url", "api_key_env", "thinking_level", "max_input_tokens")
PIPELINE_MEMBERS = ("ask", "client", "org_id", "table", "prompt_version", "system_prompt", "answer_model",
                    "decision_metadata", "settings", "retry_interval")
CONFTEST_NAMES = ("write_table", "household_rows", "pop_rows", "PARQUET_COLUMNS", "CATALOG_COLUMNS")


@pytest.mark.parametrize("module,names", SURFACE, ids=[m for m, _ in SURFACE])
def test_names_import(module: str, names: tuple[str, ...]):
    mod = importlib.import_module(module)
    missing = [n for n in names if not hasattr(mod, n)]
    assert not missing, f"{module} lost {missing}"


@pytest.mark.parametrize("module,name,params", SIGNATURES, ids=[f"{m}.{n}" for m, n, _ in SIGNATURES])
def test_signatures(module: str, name: str, params: tuple[str, ...]):
    func = getattr(importlib.import_module(module), name)
    func = getattr(func, "__wrapped__", func)
    sig = inspect.signature(func).parameters
    assert tuple(sig)[: len(params)] == params, f"{module}.{name}{inspect.signature(func)}"
    for kw in KEYWORD_ONLY.get((module, name), ()):
        assert sig[kw].kind is inspect.Parameter.KEYWORD_ONLY, f"{module}.{name}: {kw} must stay keyword-only"


def test_cached_is_an_lru_cache_with_the_build_arguments():
    from kosis_chat import api

    assert callable(api._cached.cache_clear) and callable(api._cached.cache_info)  # Phase 2 tests/test_stat_baseline.py:53
    params = tuple(inspect.signature(api._cached.__wrapped__).parameters)
    assert params == ("path", "settings", "retry_interval")


def test_settings_fields_and_model_params():
    from kosis_chat.config import Settings

    settings = Settings()
    missing = [f for f in SETTINGS_FIELDS if not hasattr(settings, f)]
    assert not missing, f"Settings lost {missing}"
    assert {"backend", "model_name"} <= set(settings.model_params())  # Phase 2 scripts/stat_baseline.py:82


def test_pipeline_members():
    from kosis_chat.pipeline import StatPipeline

    assert callable(StatPipeline.ask)
    assert tuple(inspect.signature(StatPipeline.ask).parameters)[:2] == ("self", "question")
    client = inspect.getattr_static(StatPipeline, "client")
    assert isinstance(client, property) and client.fset is not None  # eval/run_eval.py:229 sets bot.client
    source = inspect.getsource(StatPipeline.__init__)
    lost = [m for m in PIPELINE_MEMBERS if m not in ("ask", "client") and f"self.{m} =" not in source]
    assert not lost, f"StatPipeline.__init__ no longer sets {lost}"


def test_llm_client_interface():
    from kosis_chat.llm.model import LLMClient, LLMResult

    assert "generate" in LLMClient.__abstractmethods__
    assert tuple(inspect.signature(LLMClient.generate).parameters) == ("self", "system", "user")
    assert {"status", "text"} <= set(inspect.signature(LLMResult).parameters)


def test_conftest_helpers_by_path():
    """Phase 2 tests/test_stat_baseline.py:29 loads this file by path and calls these helpers."""
    spec = importlib.util.spec_from_file_location("phase1_conftest", ROOT / "tests" / "conftest.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    missing = [n for n in CONFTEST_NAMES if not hasattr(module, n)]
    assert not missing, f"tests/conftest.py lost {missing}"
