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
| `kosis_chat.data.events` | `event_key` | `compute/resolve.py:20` |
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

## 반환 객체에서 읽는 속성

- `StatPipeline`: `ask(question, golden_scope=None)`, `client` (setter: `eval/run_eval.py:229`), `org_id table prompt_version system_prompt answer_model decision_metadata settings retry_interval`.
