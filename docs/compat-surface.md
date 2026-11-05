# Phase 1 호환 목록 (T1-28)

리팩터로 파일을 옮겨도 아래 이름은 옛 경로에서 그대로 import 돼야 한다. `tests/test_compat_surface.py`가 이 목록을 검사한다(이름, 인자 이름·순서, 키워드 전용 인자, 반환 객체에서 읽는 속성).

규칙
- 파일을 옮기면 옛 경로에 재export 파일을 남긴다. `from x import *`는 밑줄 이름을 내보내지 않으므로 **이름을 하나씩 적는다**(`from kosis_chat.prepare.convert import load_table, sha256_of` 같은 식).
- 목록의 이름을 지우거나 인자를 바꾸려면 plan에 적고, Phase 1·Phase 2 호출처를 다시 grep한 결과를 PR에 남긴다.
- 확인 시점: 2026-11-05, Phase 2 `src evals scripts tests lab` + Phase 1 `kosis_chat src eval tests` grep.

## Phase 2 (KOSIS-ai-RAG)가 쓰는 것

| 모듈 | 이름 | 쓰는 곳 |
|---|---|---|
| `kosis_chat.api` | `summarize_stat(question, table_id, *, settings, client, retry_interval)` | `scripts/stat_baseline.py`, `lab/rag_baseline.py` |
| | `table_path(table_id)` | `evals/trace.py:61`, `scripts/stat_baseline.py` |
| | `_build(path, settings, client, retry_interval)` → `.ask(question)`, `.org_id`, `Answer.decision_metadata["spans"]` | `evals/trace.py:61` (S5에서 `summarize_stat_run`으로 대체) |
| | `_cached` (lru_cache, `cache_clear`) | `tests/test_stat_baseline.py:53` |
| `kosis_chat.config` | `Settings`(`backend serialization model_name temperature max_tokens timeout_seconds api_key_env …`, `model_params()`), `load_settings(path, **overrides)`, `load_env`, `get_secret`, `resolve_path` | `src/kosis_rag/{chain,clova,hcx}.py`, `evals/ops.py:64`, `scripts/stat_baseline.py`, `tests/test_stat_baseline.py`, `lab/*` |
| `kosis_chat.contract` | `to_stat_answer(question, answer, *, org_id)` | `evals/trace.py:58` |
| `kosis_chat.pipeline` | `VERSION` | `scripts/stat_baseline.py:53` |
| `kosis_chat.llm.model` | `LLMClient`(`generate(system, user)`), `LLMResult`, `MockClient`, `HCXClient`, `build_client(settings)`, `STATUS_OK`, `STATUS_API_ERROR` | `src/kosis_rag/{chain,graph,hcx,tools,hq/graph}.py`, `evals/{ops,runners/retrieval}.py`, tests 여러 곳 |
| `kosis_chat.data.loader` | `sha256_of(path)` | `tests/test_stat_baseline.py:16` |
| `kosis_ai.schemas` | `TableInfo MAX_DIMS Source Cell Fact StatAnswer StatResult StatQuery TextPart FactPart Refusal QuestionKind QuestionType RefusalReason` | `src/kosis_rag/*`, `evals/*`, tests 여러 곳 |
| `kosis_ai.normalize` | `_meta_unit(table_id, data_dir)` | `src/kosis_rag/metadata.py:22` (S2b에서 공개 별칭 `meta_unit` 추가) |
| `kosis_ai.paths` | `data_dir`, `workspace_root`, `ENV_VAR` | `tests/test_workspace_default.py` |
| `eval.golden` | `load_items(path)` | `scripts/stat_baseline.py:50` |
| `eval.grade_run`, `eval.scorer.read_jsonl` | | `tests/test_stat_baseline.py:12-13` |
| `tests/conftest.py` (파일 경로로 로드) | `write_table`, `household_rows`, `pop_rows` | `tests/test_stat_baseline.py:29` |

## Phase 1 안에서 모듈을 넘어 읽는 비공개 이름

| 모듈 | 이름 | 쓰는 곳 |
|---|---|---|
| `kosis_chat.compute.engine` | `_read`, `_Read` | `llm/analyze_plan.py:21`, `compute/semantics.py:10`, `tests/test_semantics.py:9` |
| `kosis_chat.compute.semantics` | `_whole_label` | `compute/resolve.py:19` |
| `kosis_chat.data.events` | `event_key` | `tests/test_answerability.py:9` (`compute/resolve.py`는 S2부터 `kosis_chat.shared.events`) |
| `kosis_chat.api` | `_build`, `_cached` (monkeypatch) | `tests/test_workspace.py:84-109` |

## 옮긴 노드 경계 타입 (T1-28 S1b)

Pydantic 모델로 바꿔 단계 `schemas.py`로 옮기고 옛 경로는 이름을 적어 재export한다(필드 이름·순서·기본값·frozen 그대로, `tests/test_schemas_boundary.py`).

| 새 위치 | 이름 | 옛 경로(재export) |
|---|---|---|
| `kosis_chat.judge.schemas` | `Verdict`, `Resolved` | `kosis_chat.judge.answerability` |
| | `Periods`, `Match`, `Parsed`, `in_range` | `kosis_chat.judge.question` |
| `kosis_chat.analyze.schemas` | `Plan` | `kosis_chat.compute.analyze` |
| | `Built`, `TermSteps`, `Pattern`, `Agg` | `kosis_chat.compute.build` (`Built`은 `kosis_chat.compute`도) |
| | `Placed` | `kosis_chat.llm.analyze_plan` |
| `kosis_chat.compute.schemas` | `Computed`, `TermResult` | `kosis_chat.compute.engine` (`Computed`는 `kosis_chat.compute`도) |
| | `Alternatives` | `kosis_chat.compute.context` |
| | `Resolved`(select의 칸, judge의 `Resolved`와 다름) | `kosis_chat.compute.resolve` |
| `kosis_chat.compose.schemas` | `ComposeOutput` (원래 Pydantic, 이동만) | `kosis_chat.llm.compose` |

- `dataclasses.replace(x, ...)` → `x.model_copy(update={...})`. 위치 인자 생성(`Match(axis, code, ...)`)은 `shared/boundary.py`가 필드 순서대로 받는다.
- `Verdict.refuse(...)`(classmethod) → `kosis_chat.judge.answerability.refuse(...)`: 거절 문장을 쓰는 `judge/messages.py`가 옛 경로라 schemas가 import하지 않는다. Phase 1·2 호출처는 answerability 안뿐(grep 11/05).

## 옮긴 data/ 모듈 (T1-28 S2)

`kosis_chat/data/`를 모듈 안(`prepare/`: 행→셀·단위·clean·색인)과 모듈 밖(`adapters/parquet.py`: 파일 열기)으로 나눴다. 옛 모듈은 이름을 적어 재export한다(옛 밑줄 이름은 같은 객체의 별칭). `data/scope.py`·`data/serializer.py`(B0)는 그대로.

| 새 위치 | 이름 | 옛 경로(재export) |
|---|---|---|
| `kosis_chat.prepare.schemas` | `TableInput`(+ `caches: LookupCaches`, init·비교 제외), `TableMeta`, `Cell`, `Problem`, `SchemaError`, `validate`, `validate_meta_dict`(둘 다 키워드 `allow_empty` 추가, 기본값이면 동작 같음), `REQUIRED_META_FIELDS PERIOD_TYPES INPUT_FORMATS VALUE_FLAGS` | `kosis_chat.data.schema` |
| `kosis_chat.prepare.clean` | `clean`, `clean_cell`, `parse_value`, `KIND_FLAGS MISSING_MARKS SUPPRESSED_MARKS FLAGGED_MARKS _PROVISIONAL` | `kosis_chat.data.preprocessing` |
| `kosis_chat.prepare.convert` | `infer_unit`, `_unit_in_parens`, `_PAREN`, `UNKNOWN_UNIT KNOWN_UNITS UNIT_OVERRIDES`, `PARQUET_COLUMNS`(옛 `_PARQUET_COLUMNS`), `PRD_SE`(`_PRD_SE`), `DIM_LEVELS`(`_DIM_LEVELS`), `build_meta`(`_build_meta`) | `kosis_chat.data.loader` |
| `kosis_chat.adapters.parquet` | `load_table(path)`, `sha256_of(path)`, `read_parquet`, `find_catalog`(`_find_catalog`), `parquet_meta`(`_parquet_meta`), `load_parquet`(`_load_parquet`), `catalog_row`, `fetched_at_of` | `kosis_chat.data.loader` |
| `kosis_chat.shared.events` (+ `events.json`) | `event_key`, `load_events`, `EVENTS_PATH` | `kosis_chat.data.events` |
| `kosis_chat.prepare.index` | `TableIndex`, `Axis`, `build_index`, `index_of`, `cell_measure`, `is_total`, `_measure`, `_link_parents`, `ITEM_AXIS SCENARIO_AXIS TOTAL_LABELS _UNIT_PAREN _REGION_AXIS` | `kosis_chat.judge.question` (`index_of`는 `kosis_chat.compute.semantics`도) |

- 새 이름(옛 경로 없음): `kosis_chat.schemas.io`(`ParquetTable`, `StatInput`), `kosis_chat.schemas.resources`(`StatResources`, `load_resources`, `LookupCaches`), `kosis_chat.prepare.node`(`prepare`, `table_input`, `content_hash`). `fetched_at`은 `ParquetTable.fetched_at`(선택, 기본 `None`)에 둔다: Phase 2가 `TableInfo`를 tables.jsonl에 쓰므로 `TableInfo` 필드는 늘리지 않는다.
- `api._build(path, settings, client, retry_interval)`·`api._cached`(lru_cache, 경로 키) 서명 그대로. `_build`는 `adapters.parquet.load_table` → `StatPipeline`(생성자 서명 그대로, 안에서 `load_resources`).
- 모듈 전역 `id(table)` 캐시 5곳(compute/analyze `_LOOKUPS`, compute/engine `_INDEX_CACHE`·`_SERIES_CACHE`, compute/resolve `_CELLS`, compute/semantics `_INDEX`)은 `TableInput.caches`로 옮겼다. 남긴 캐시(값 키, 표와 무관): `prepare.index._measure`, `judge.question._event_pattern`, `shared.events.load_events`, `api._cached`, `kosis_ai.normalize`의 lru_cache.

## 반환 객체에서 읽는 속성

- `StatPipeline`: `ask(question, golden_scope=None)`, `client` (setter: `eval/run_eval.py:229`), `org_id table prompt_version system_prompt answer_model decision_metadata settings retry_interval`. T1-28 S2부터 `resources`(StatResources)와 `index`(= `resources.index`)도 있음.
