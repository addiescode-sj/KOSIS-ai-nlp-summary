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
    # T1-28 S2c: the model is injected as an LLMConfig; providers table; Settings stays as the Phase 2 shim
    ("kosis_chat.schemas.resources", ("LLMConfig", "LLMKeyMissing", "DEFAULT_MODEL", "DEFAULT_BASE_URL",
                                      "MOCK_PROVIDER", "MOCK_RESPONSE")),
    ("kosis_chat.llm.providers", ("PROVIDERS", "register_provider", "build_client", "LLMKeyMissing")),
    ("kosis_chat.llm.model", ("BACKENDS", "TraceWriteError")),
    ("kosis_chat.config", ("PROJECT_ROOT", "settings_from_llm", "env_llm_config")),
    # T1-28 S3a: the pure stage steps (node.py) with their In/Out models, and the modules moved into the stage folders
    ("kosis_chat.judge.node", ("step_judge",)),
    ("kosis_chat.judge.schemas", ("JudgeIn", "JudgeOut")),
    ("kosis_chat.analyze.node", ("step_plan_rules",)),
    ("kosis_chat.analyze.schemas", ("PlanRulesIn", "PlanRulesOut")),
    ("kosis_chat.compute.node", ("step_compute",)),
    ("kosis_chat.compute.schemas", ("ComputeIn", "ComputeOut")),
    ("kosis_chat.compose.node", ("step_caveats", "step_verify", "step_template")),
    ("kosis_chat.compose.schemas", ("CaveatsIn", "CaveatsOut", "VerifyIn", "VerifyOut", "TemplateIn", "TemplateOut",
                                    "COMPOSE_MAX_ATTEMPTS", "ComposeOutput")),
    ("kosis_chat.chart.node", ("step_chart",)),
    ("kosis_chat.chart.schemas", ("ChartIn", "ChartOut")),
    ("kosis_chat.render.node", ("step_render", "step_answer_final", "causal_note")),
    ("kosis_chat.render.schemas", ("RenderIn", "AnswerFinalIn", "RenderOut")),
    ("kosis_chat.judge.rules", ("judge", "resolve", "refuse", "Verdict", "Resolved", "defaults", "_period_text",
                                "_range_text", "_UNIT_SUFFIX", "RECENT_POINTS")),
    ("kosis_chat.judge.texts", ("REFUSAL_MESSAGES", "refusal_message", "default_note", "josa", "with_josa", "STOCK_OVER_TIME",
                                "CLARIFY_LABEL", "CLARIFY_ITEM", "CLARIFY_COMPARE", "COMPARE_OPTIONS", "_default_text")),
    ("kosis_chat.analyze.rules", ("rule_plan", "build_plan", "to_query", "Plan", "Unresolved", "PERIOD_AXIS", "_LATEST",
                                  "_RECENT", "_PREVIOUS", "_lookup", "_Lookup")),
    ("kosis_chat.analyze.blocks", ("Built", "TermSteps", "Pattern", "Agg", "AGG_WORDS", "value", "trend", "compare", "share",
                                   "extreme", "over_periods", "written")),
    ("kosis_chat.analyze.roles", ("is_province", "pick_of", "reference_index", "_PROVINCE")),
    ("kosis_chat.compose.caveats", ("caveats_of", "_EVENT_MODEL", "_PREFIX", "_from_caution")),
    ("kosis_chat.compose.verify", ("verify", "label_digits", "allowed_numbers", "NUMBER")),
    ("kosis_chat.compose.template", ("template_parts", "_DERIVED_IDS", "_TEMPLATES")),
    ("kosis_chat.chart.builders", ("chart_spec", "GRID_PATTERNS", "GROUPED_MAX_PERIODS", "_CHARTS", "_grid")),
    ("kosis_chat.chart.vega", ("SCHEMA", "HIGHLIGHT", "OTHER", "WIDTH", "_base", "_bars", "_timeline", "_row")),
    ("kosis_chat.chart.grid", ("Grid", "grid")),
    # T1-28 S5: the module entry point, the chart as data, the old-record boundary
    ("kosis_chat.api", ("StatModule", "summarize_stat_run", "_cached_llm")),
    ("kosis_chat.schemas.io", ("StatInput", "ParquetTable", "StatOutput", "StatOptions")),
    ("kosis_chat.chart.builders", ("BUILDERS", "register_builder")),
    ("kosis_chat.chart.vega", ("to_vega", "with_vega", "DRAW", "register_kind")),
    ("kosis_chat.adapters.parquet", ("read_catalog", "file_stamp", "CHECK_CATALOG_TBL_ID", "catalog_tbl_id_problems")),
    ("kosis_ai.schemas", ("ChartSpec", "ChartSeries", "ChartPoint", "CHART_KINDS")),
    ("kosis_ai.compat", ("load_stat_answer", "legacy_chart", "without_legacy_chart", "LEGACY_CHART_KEYS")),
    ("kosis_chat.render.text", ("render_parts", "final_text", "default_sentence")),
    ("kosis_chat.render.table", ("table_part",)),
    ("kosis_chat.render.source", ("to_source", "org_id_from_table_path", "_ORG_FILE", "_kosis_host")),
    # ... and the old paths keep every name (tests, eval and Phase 2 import them there)
    ("kosis_chat.answer.chart", ("chart_spec", "GRID_PATTERNS", "GROUPED_MAX_PERIODS", "SCHEMA", "HIGHLIGHT", "OTHER",
                                 "WIDTH", "_CHARTS", "_grid", "_bars", "_base", "_row", "_timeline", "_common", "_differing")),
    ("kosis_chat.answer.table", ("Grid", "grid", "table_part")),
    ("kosis_chat.answer.caveats", ("caveats_of",)),
    ("kosis_chat.answer.template", ("template_parts", "_TEMPLATES")),
    ("kosis_chat.answer.verify", ("verify", "label_digits", "allowed_numbers")),
    ("kosis_chat.compute.analyze", ("rule_plan", "build_plan", "to_query", "_lookup", "_Lookup")),
    ("kosis_chat.compute.build", ("value", "trend", "compare", "share", "extreme", "over_periods", "written", "AGG_WORDS")),
    ("kosis_chat.compute.roles", ("pick_of", "reference_index")),
    ("kosis_chat.judge.messages", ("STOCK_OVER_TIME", "CLARIFY_LABEL", "CLARIFY_ITEM", "CLARIFY_COMPARE", "COMPARE_OPTIONS")),
    ("kosis_chat.pipeline", ("_causal_note",)),
    # T1-28 S3b: the model steps (node.py, legacy/fallback.py), their records, spans and legacy Answer; the LLM step
    # modules moved into analyze/ and compose/ with their prompts as .md files
    ("kosis_chat.analyze.node", ("step_plan_fill_llm", "step_plan_write_llm", "step_event_lookup")),
    ("kosis_chat.analyze.schemas", ("PlanFillIn", "PlanFillOut", "PlanWriteIn", "PlanWriteOut", "EventLookupIn",
                                    "EventLookupOut")),
    ("kosis_chat.compose.node", ("step_compose_llm",)),
    ("kosis_chat.compose.schemas", ("ComposeLLMIn", "ComposeLLMOut")),
    ("kosis_chat.legacy.fallback", ("step_fallback_chatbot", "FallbackIn", "FallbackOut", "FALLBACK_PROMPT")),
    ("kosis_chat.schemas.state", ("LLMCall", "StepSpan", "Record", "called_before")),
    ("kosis_chat.llm.calls", ("ModelAccess",)),
    ("kosis_chat.graph.spans", ("spans_of", "span_of", "step_span", "llm_attributes", "SPAN_NAMES")),
    ("kosis_chat.graph.legacy", ("to_answer", "fallback_answer", "totals_of", "add_usage", "DECISION_KEYS")),
    ("kosis_chat.analyze.prompts", ("load", "version", "FILL", "PLAN", "EVENT_LOOKUP")),
    ("kosis_chat.compose.prompts", ("load", "version", "COMPOSE")),
    ("kosis_chat.analyze.fill", ("AnalyzeFill", "apply", "user_prompt", "retry_prompt", "resolve_code", "menu",
                                 "asked_axes", "MENU_LIMIT", "VERSION", "SYSTEM_PROMPT", "RETRY", "_move", "_menu_codes")),
    ("kosis_chat.analyze.write", ("place", "normalized", "query_plan", "user_prompt", "retry_prompt", "menu", "Placed",
                                  "VERSION", "SYSTEM_PROMPT", "EXAMPLES", "RETRY", "FN_WORDS", "PERIODS_SHOWN",
                                  "_periods_line", "_missing_periods", "_select", "_codes", "_read")),
    ("kosis_chat.analyze.event_lookup", ("EventStart", "user_prompt", "VERSION", "SYSTEM_PROMPT", "FIRST_YEAR", "_START")),
    ("kosis_chat.compose.compose", ("ComposeOutput", "user_prompt", "retry_prompt", "to_parts", "fact_line", "VERSION",
                                    "SYSTEM_PROMPT", "EXAMPLE", "RETRY", "_PLACEHOLDER")),
    # ... and the old paths keep every name
    ("kosis_chat.llm.compose", ("VERSION", "SYSTEM_PROMPT", "EXAMPLE", "RETRY", "user_prompt", "retry_prompt", "to_parts",
                                "fact_line", "_PLACEHOLDER")),
    ("kosis_chat.llm.analyze_fill", ("AnalyzeFill", "apply", "user_prompt", "retry_prompt", "resolve_code", "MENU_LIMIT",
                                     "VERSION", "SYSTEM_PROMPT", "RETRY")),
    ("kosis_chat.llm.analyze_plan", ("place", "user_prompt", "retry_prompt", "VERSION", "SYSTEM_PROMPT", "EXAMPLES",
                                     "RETRY", "build", "_read")),
    ("kosis_chat.llm.event_lookup", ("EventStart", "user_prompt", "VERSION", "SYSTEM_PROMPT")),
    ("kosis_chat.pipeline", ("add_usage", "FALLBACK_PROMPT", "_llm_attributes")),
    # T1-28 S6: the B0 modules moved to kosis_chat/legacy/; the old paths (above and eval/run_eval.py) keep every name
    ("kosis_chat.chatbot", ("Answer", "Chatbot", "JUDGE_MODES", "judge_record", "verdict_answer", "estimate_tokens")),
    ("kosis_chat.raw", ("RawRunner", "DECISION_METADATA")),
    ("kosis_chat.data.scope", ("MODES", "Scope", "select", "apply", "fit", "keep", "AUTO_MAX_CHARS")),
    ("kosis_chat.data.serializer", ("serialize", "meta_header", "STYLES", "to_markdown", "to_wide_markdown")),
    ("kosis_chat.llm.prompts", ("SYSTEM_PROMPTS", "NUMBER_RULE", "STRUCTURED_PROMPT_VERSIONS", "build_user_prompt",
                                "get_system_prompt", "resolve_answer_prompt")),
    ("kosis_chat.llm.raw_prompt", ("VERSION", "SYSTEM_PROMPT", "USER_TEMPLATE", "user_prompt")),
    ("kosis_chat.llm.answer_schema", ("LLMAnswer", "CompactAnswer", "answer_model", "PLAIN_TEXT")),
    ("kosis_chat.llm.compact_v2", ("CompactAnswerV2", "VERSION", "repair_hint", "strip_absent")),
    ("kosis_chat.contract", ("to_stat_answer", "render_text", "strip_model_source", "_SOURCE_LINE", "_is_kosis_link_line")),
    ("kosis_chat.legacy.chatbot", ("Answer", "Chatbot", "JUDGE_MODES", "judge_record", "verdict_answer")),
    ("kosis_chat.legacy.contract_text", ("to_stat_answer", "render_text", "strip_model_source")),
    ("kosis_chat.legacy.cli", ("main", "build_parser", "QUIT_WORDS")),  # the B0 CLI (kosis_chat/cli.py: StatModule)
    ("kosis_chat.cli", ("main", "build_parser", "llm_config", "QUIT_WORDS")),
    ("kosis_chat.graph.legacy", ("Answer", "judge_record", "FallbackIn", "step_fallback_chatbot")),
)

# T1-28 S3a/S3b moves: (old module, new module); every public and private name of the new module the old one lists is the
# same object at both paths (the old module is a name-by-name re-export).
MOVED = (
    ("kosis_chat.judge.answerability", "kosis_chat.judge.rules"),
    ("kosis_chat.judge.messages", "kosis_chat.judge.texts"),
    ("kosis_chat.compute.analyze", "kosis_chat.analyze.rules"),
    ("kosis_chat.compute.build", "kosis_chat.analyze.blocks"),
    ("kosis_chat.compute.roles", "kosis_chat.analyze.roles"),
    ("kosis_chat.answer.caveats", "kosis_chat.compose.caveats"),
    ("kosis_chat.answer.verify", "kosis_chat.compose.verify"),
    ("kosis_chat.answer.template", "kosis_chat.compose.template"),
    ("kosis_chat.answer.render", "kosis_chat.render.text"),
    ("kosis_chat.answer.chart", "kosis_chat.chart.builders"),
    # T1-28 S3b
    ("kosis_chat.llm.analyze_fill", "kosis_chat.analyze.fill"),
    ("kosis_chat.llm.analyze_plan", "kosis_chat.analyze.write"),
    ("kosis_chat.llm.event_lookup", "kosis_chat.analyze.event_lookup"),
    ("kosis_chat.llm.compose", "kosis_chat.compose.compose"),
    # T1-28 S6: the B0 modules moved to kosis_chat/legacy/
    ("kosis_chat.chatbot", "kosis_chat.legacy.chatbot"),
    ("kosis_chat.raw", "kosis_chat.legacy.raw"),
    ("kosis_chat.data.scope", "kosis_chat.legacy.scope"),
    ("kosis_chat.data.serializer", "kosis_chat.legacy.serializer"),
    ("kosis_chat.llm.prompts", "kosis_chat.legacy.prompts"),
    ("kosis_chat.llm.raw_prompt", "kosis_chat.legacy.raw_prompt"),
    ("kosis_chat.llm.answer_schema", "kosis_chat.legacy.answer_schema"),
    ("kosis_chat.llm.compact_v2", "kosis_chat.legacy.compact_v2"),
)

# Callables whose parameters callers pass by name or position: (module, name, parameter names in order).
SIGNATURES: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("kosis_chat.api", "summarize_stat", ("question", "table_id", "settings", "client", "retry_interval", "llm")),
    ("kosis_chat.api", "table_path", ("table_id",)),
    ("kosis_chat.api", "_build", ("path", "settings", "client", "retry_interval", "llm")),  # Phase 2 evals/trace.py:61
    ("kosis_chat.api", "summarize_stat_run", ("question", "table_id", "settings", "client", "retry_interval", "llm")),
    ("kosis_chat.api", "StatModule", ("llm", "options", "client", "cache_size")),
    ("kosis_ai.compat", "load_stat_answer", ("data",)),
    ("kosis_chat.contract", "to_stat_answer", ("question", "answer", "org_id")),
    ("kosis_chat.config", "load_settings", ("path", "overrides")),
    ("kosis_chat.config", "load_env", ("path",)),
    ("kosis_chat.config", "get_secret", ("name",)),
    ("kosis_chat.llm.model", "build_client", ("settings",)),
    ("kosis_chat.llm.model", "HCXClient", ("config", "sleep", "observer", "settings")),  # S2c: config | Settings
    ("kosis_chat.llm.providers", "build_client", ("config",)),
    ("kosis_chat.llm.providers", "register_provider", ("name", "cls")),
    ("kosis_chat.schemas.resources", "LLMConfig.from_settings", ("settings", "api_key")),
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
    # T1-28 S3a: step_<name>(In, StatResources) -> Out
    ("kosis_chat.judge.node", "step_judge", ("inp", "res")),
    ("kosis_chat.analyze.node", "step_plan_rules", ("inp", "res", "build")),
    ("kosis_chat.compute.node", "step_compute", ("inp", "res")),
    ("kosis_chat.compose.node", "step_caveats", ("inp", "res")),
    ("kosis_chat.compose.node", "step_verify", ("inp", "res")),
    ("kosis_chat.compose.node", "step_template", ("inp", "res")),
    ("kosis_chat.chart.node", "step_chart", ("inp", "res")),
    ("kosis_chat.render.node", "step_render", ("inp", "res")),
    ("kosis_chat.render.node", "step_answer_final", ("inp", "res")),
    # T1-28 S3b: the model steps take the model as a keyword (llm=ModelAccess)
    ("kosis_chat.analyze.node", "step_plan_fill_llm", ("inp", "res", "llm", "build")),
    ("kosis_chat.analyze.node", "step_plan_write_llm", ("inp", "res", "llm")),
    ("kosis_chat.analyze.node", "step_event_lookup", ("inp", "res", "llm")),
    ("kosis_chat.compose.node", "step_compose_llm", ("inp", "res", "llm")),
    ("kosis_chat.legacy.fallback", "step_fallback_chatbot", ("inp", "res", "chatbot", "llm")),
    ("kosis_chat.llm.analyze_plan", "place", ("question", "plan", "table", "index", "lookup")),
    ("kosis_chat.llm.analyze_plan", "user_prompt", ("question", "index", "rule")),
    ("kosis_chat.llm.analyze_fill", "apply", ("plan", "fill", "index")),
    ("kosis_chat.llm.analyze_fill", "user_prompt", ("question", "index", "plan")),
    ("kosis_chat.llm.compose", "user_prompt", ("question", "computed", "cautions", "context_ids")),
    ("kosis_chat.llm.compose", "to_parts", ("answer", "computed")),
    ("kosis_chat.judge.answerability", "judge", ("question", "table", "index")),
    ("kosis_chat.compute.analyze", "rule_plan", ("question", "index", "default_applied")),
    ("kosis_chat.compute.analyze", "build_plan", ("plan", "index", "table")),
    ("kosis_chat.compute.analyze", "to_query", ("plan", "index", "built")),
    ("kosis_chat.answer.caveats", "caveats_of", ("computed", "table", "alternatives")),
    ("kosis_chat.answer.chart", "chart_spec", ("kind", "computed")),
    ("kosis_chat.answer.template", "template_parts", ("kind", "computed")),
    ("kosis_chat.answer.verify", "verify", ("parts", "computed")),
    ("kosis_chat.contract", "to_source", ("meta", "org_id")),
    # T1-28 S6: the B0 classes run_eval and tests build through the old paths
    ("kosis_chat.chatbot", "Chatbot", ("table", "client", "settings", "prompt_version", "form", "scope_mode",
                                       "answer_format", "retry_interval", "judge_mode")),
    ("kosis_chat.raw", "RawRunner", ("table", "client", "settings", "scope_mode")),
    ("kosis_chat.data.scope", "select", ("table", "mode", "question", "golden_scope", "style")),
    ("kosis_chat.data.serializer", "serialize", ("table", "style")),
    ("kosis_chat.cli", "main", ("argv",)),
    ("kosis_chat.legacy.cli", "main", ("argv",)),
)
KEYWORD_ONLY = {("kosis_chat.api", "summarize_stat"): {"settings", "client", "retry_interval", "llm"},
                ("kosis_chat.analyze.node", "step_plan_rules"): {"build"},
                ("kosis_chat.analyze.node", "step_plan_fill_llm"): {"llm", "build"},
                ("kosis_chat.analyze.node", "step_plan_write_llm"): {"llm"},
                ("kosis_chat.analyze.node", "step_event_lookup"): {"llm"},
                ("kosis_chat.compose.node", "step_compose_llm"): {"llm"},
                ("kosis_chat.legacy.fallback", "step_fallback_chatbot"): {"chatbot", "llm"},
                ("kosis_chat.contract", "to_stat_answer"): {"org_id"},
                ("kosis_chat.llm.model", "HCXClient"): {"settings"}}

# Attributes callers read from returned objects.
SETTINGS_FIELDS = ("backend", "serialization", "model_name", "temperature", "max_tokens", "timeout_seconds",
                   "base_url", "api_key_env", "thinking_level", "max_input_tokens")
LLM_CONFIG_FIELDS = ("api_key", "model", "provider", "temperature", "max_tokens", "max_input_tokens", "timeout_seconds",
                     "base_url", "thinking_effort")  # plans/T1-28.md 입출력 계약
PIPELINE_MEMBERS = ("ask", "client", "org_id", "table", "prompt_version", "system_prompt", "answer_model",
                    "decision_metadata", "settings", "retry_interval")
CONFTEST_NAMES = ("write_table", "household_rows", "pop_rows", "PARQUET_COLUMNS", "CATALOG_COLUMNS")


@pytest.mark.parametrize("module,names", SURFACE, ids=[m for m, _ in SURFACE])
def test_names_import(module: str, names: tuple[str, ...]):
    mod = importlib.import_module(module)
    missing = [n for n in names if not hasattr(mod, n)]
    assert not missing, f"{module} lost {missing}"


@pytest.mark.parametrize("old,new", MOVED, ids=[o for o, _ in MOVED])
def test_moved_names_are_the_same_objects(old: str, new: str):
    old_mod, new_mod = importlib.import_module(old), importlib.import_module(new)
    names = [n for n in vars(old_mod) if not (n.startswith("__") and n.endswith("__"))]
    assert names, old
    differ = [n for n in names if getattr(old_mod, n) is not getattr(new_mod, n, object())]
    assert not differ, f"{old} names that are not {new}'s: {differ}"
    # and the other way for the S6 B0 moves (bodies moved whole, so the shim keeps every name of the new module; earlier
    # stages' new modules gained names the old paths never had)
    if not new.startswith("kosis_chat.legacy."):
        return
    dropped = [n for n in vars(new_mod) if n.isidentifier() and not (n.startswith("__") and n.endswith("__"))
               and n != "annotations" and n not in vars(old_mod)]
    assert not dropped, f"{old} no longer re-exports {dropped}"


# Split moves: a name each (old module, name, new module).
SPLIT_MOVES = (
    ("kosis_chat.answer.table", "Grid", "kosis_chat.chart.grid"),
    ("kosis_chat.answer.table", "grid", "kosis_chat.chart.grid"),
    ("kosis_chat.answer.table", "table_part", "kosis_chat.render.table"),
    ("kosis_chat.answer.chart", "_bars", "kosis_chat.chart.vega"),
    ("kosis_chat.contract", "to_source", "kosis_chat.render.source"),
    ("kosis_chat.contract", "org_id_from_table_path", "kosis_chat.render.source"),
    ("kosis_chat.contract", "_ORG_FILE", "kosis_chat.render.source"),
    ("kosis_chat.contract", "_kosis_host", "kosis_chat.render.source"),
    # T1-28 S6: the B0 text handling of contract.py
    ("kosis_chat.contract", "to_stat_answer", "kosis_chat.legacy.contract_text"),
    ("kosis_chat.contract", "render_text", "kosis_chat.legacy.contract_text"),
    ("kosis_chat.contract", "strip_model_source", "kosis_chat.legacy.contract_text"),
    ("kosis_chat.contract", "_SOURCE_LINE", "kosis_chat.legacy.contract_text"),
    ("kosis_chat.contract", "_SOURCE_LINK", "kosis_chat.legacy.contract_text"),
    ("kosis_chat.contract", "_SEPARATOR", "kosis_chat.legacy.contract_text"),
    ("kosis_chat.contract", "_is_kosis_link_line", "kosis_chat.legacy.contract_text"),
    ("kosis_chat.graph.legacy", "Answer", "kosis_chat.legacy.chatbot"),
    ("kosis_chat.graph.legacy", "step_fallback_chatbot", "kosis_chat.legacy.fallback"),
    ("kosis_chat.graph.legacy", "FallbackIn", "kosis_chat.legacy.fallback"),
)


@pytest.mark.parametrize("old,name,new", SPLIT_MOVES, ids=[f"{o}.{n}" for o, n, _ in SPLIT_MOVES])
def test_split_moved_names_are_the_same_objects(old: str, name: str, new: str):
    assert getattr(importlib.import_module(old), name) is getattr(importlib.import_module(new), name)


@pytest.mark.parametrize("module,name,params", SIGNATURES, ids=[f"{m}.{n}" for m, n, _ in SIGNATURES])
def test_signatures(module: str, name: str, params: tuple[str, ...]):
    func = importlib.import_module(module)
    for part in name.split("."):
        func = getattr(func, part)
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


def test_llm_config_fields_and_config_shims():
    import warnings

    from kosis_chat import config
    from kosis_chat.schemas.resources import LLMConfig, LLMKeyMissing

    assert tuple(LLMConfig.model_fields) == LLM_CONFIG_FIELDS
    assert issubclass(LLMKeyMissing, RuntimeError)  # summarize_stat: a failed model call raises RuntimeError
    for shim, args in ((config.load_env, ("does-not-exist.env",)), (config.get_secret, ("T1_28_UNSET",))):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            shim(*args)  # Phase 2 still calls them; they warn and keep working
        assert any(issubclass(w.category, DeprecationWarning) for w in caught), shim.__name__


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
