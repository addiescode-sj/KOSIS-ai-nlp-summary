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
| `kosis_chat.config` | `Settings`(`backend serialization model_name temperature max_tokens timeout_seconds api_key_env …`, `model_params()`), `load_settings(path, **overrides)`, `load_env`, `get_secret`, `resolve_path` (S2c부터 shim: 아래 "LLM 설정 주입") | `src/kosis_rag/{chain,clova,hcx}.py`, `evals/ops.py:64`, `scripts/stat_baseline.py`, `tests/test_stat_baseline.py`, `lab/*` |
| `kosis_chat.contract` | `to_stat_answer(question, answer, *, org_id)` | `evals/trace.py:58` |
| `kosis_chat.pipeline` | `VERSION` | `scripts/stat_baseline.py:53` |
| `kosis_chat.llm.model` | `LLMClient`(`generate(system, user)`), `LLMResult`, `MockClient`, `HCXClient`(S2c: `HCXClient(config \| settings)`), `build_client(settings)`, `STATUS_OK`, `STATUS_API_ERROR` | `src/kosis_rag/{chain,graph,hcx,tools,hq/graph}.py`, `evals/{ops,runners/retrieval}.py`, tests 여러 곳 |
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

## 공용 추출 (T1-28 S2b)

둘 이상의 단계가 쓰던 함수·정규식·문구를 `kosis_ai/units.py`와 `kosis_chat/shared/`로 옮겼다. 순수 이동이고 옛 경로는 이름을 적어 재export한다(같은 객체; 옛 밑줄 이름은 별칭). 정규식은 패턴 글자·플래그가 같거나 쓰는 곳에서 같은 문자열에 같게 맞는 것만 한 객체로 합쳤다(D16 (d)). `tests/test_shared_{units,periods,labels,numbers,protocols}.py`, `tests/test_compute_texts.py`.

| 새 위치 | 이름 | 옛 경로(재export) |
|---|---|---|
| `kosis_ai.units` | 로더 규칙 `infer_unit(unit, item, dim_labels)`, `unit_in_parens`, `UNKNOWN_UNIT`, `INFER_KNOWN_UNITS`(16), `INFER_UNIT_OVERRIDES`, `PAREN` | `kosis_chat.prepare.convert`·`kosis_chat.data.loader`(`infer_unit _unit_in_parens _PAREN UNKNOWN_UNIT KNOWN_UNITS UNIT_OVERRIDES`) |
| | normalize 규칙 `resolve_unit(unit, item_name, dim_labels, meta_unit=None)`, `measure_kind`, `known_unit`, `paren_unit`, `override_unit`, `keyword_kind`, `RESOLVE_KNOWN_UNITS`(26), `RESOLVE_UNIT_OVERRIDES`, `NOT_COUNT`, `KIND_KEYWORDS`, `RATIO_UNITS STOCK_UNITS FLOW_UNITS MEASURE_OVERRIDES`, `UNIT_PAREN` | `kosis_ai.normalize`(`resolve_unit measure_kind KNOWN_UNITS UNIT_OVERRIDES MEASURE_OVERRIDES _known _paren_unit _override_unit _keyword_kind _PAREN _NOT_COUNT _KIND_KEYWORDS _RATIO_UNITS _STOCK_UNITS _FLOW_UNITS _UNIT_SUFFIX`) |
| | 계약 규칙 `unit_or_suffix(unit, texts)`, `SUFFIX_UNIT` | `kosis_ai.schemas.Cell.unit_or_suffix()`(메서드 유지, 본문이 함수 호출), `kosis_ai.schemas._UNIT_SUFFIX` |
| `kosis_ai.normalize` | `meta_unit` = `_meta_unit`(같은 lru_cache 함수, 공개 별칭) | — |
| `kosis_ai.schemas` | `NUMBER`(= `_NUMBER`) | — |
| `kosis_chat.shared.patterns` | `UNIT_PAREN PAREN REGION_SUFFIX YEAR YEAR_RANGE YEAR_SINCE RECENT_N COHORT NUMBER FACT_ID ORG_ID` | `judge.question`(`_UNIT_PAREN _REGION_SUFFIX _YEAR _RANGE _SINCE _RECENT_N _COHORT`), `data.scope`(`_YEAR _YEAR_RANGE _YEAR_SINCE _RECENT_N _UNIT_PAREN _REGION_SUFFIX`), `judge.answerability._UNIT_SUFFIX`, `answer.render._UNIT_SUFFIX`, `compute.roles._PROVINCE`, `answer.verify.NUMBER`, `answer.template._DERIVED_IDS` |
| `kosis_chat.shared.periods` | `period_text`, `span_text` | `answer.render`, `judge.answerability._period_text` |
| `kosis_chat.shared.labels` | `squash`, `LABEL_ALIASES`, `item_label`, `fact_label` | `judge.question.squash`·`LABEL_ALIASES`, `data.scope._squash`·`LABEL_ALIASES`, `answer.render` |
| `kosis_chat.shared.numbers` | `SYMBOL_MEANING format_number format_fact is_change direction derived_kind display_value` | `answer.render`, `data.serializer.display_value` |
| `kosis_chat.shared.texts` | `josa`, `with_josa` | `judge.messages`, `answer.render.josa` |
| `kosis_chat.shared.defaults` | `make_latest make_previous make_recent parse_period parse_period_codes PERIOD_KEY DEFAULT_KEYS LATEST RECENT PREVIOUS LATEST_CODES RECENT_CODES PREVIOUS_CODES` | `judge.messages._LATEST _RECENT _PREVIOUS _DEFAULT_KEYS`, `compute.analyze._LATEST _RECENT _PREVIOUS` |
| `kosis_chat.shared.cautions` | `make_caution event_source default_applied DEFAULT_LATEST missing_years recent_short simple_average parse_event_source parse_caution parse_default_applied parse_missing_years EVENT_MODEL PREFIX` | `answer.caveats._EVENT_MODEL _PREFIX` |
| `kosis_chat.compute.texts` | 계산 거절·되묻기 문장 10개(`event_unknown path_ambiguous pick_category span_not_supported span_on_yearly other_table star_outside_map pp_diff_units share_outside_total share_out_of_bounds`) | — (resolve.py·semantics.py의 f-string 자리) |

- 층 위반 제거: `compute.build`·`compute.plan_engine` → `shared.periods.span_text`(옛 `answer.render`), `judge.question` → `shared.labels.LABEL_ALIASES`(옛 `data.scope`), `compute.engine` → `shared.numbers.display_value`(옛 `data.serializer`), `contract` → `chatbot.Answer`는 주석용 `TYPE_CHECKING` import만(실행 시 import 없음; `contract.Answer` 속성은 없어짐, 호출처 0건 grep 11/05).
- `prepare.index`는 `kosis_ai.normalize` 대신 `kosis_ai.units`를 import한다(`_measure`의 규칙 함수는 같은 객체).

## LLM 설정 주입 (T1-28 S2c)

키·모델은 호출자가 `LLMConfig`로 넘긴다(D25). 모듈은 `.env`·환경변수·`configs/model.yaml`(삭제)을 읽지 않는다. 예외는 아래 Settings shim 경로와 `kosis_ai/paths.py`(`KOSIS_WORKSPACE`). `tests/test_llm_config.py`.

| 위치 | 이름 | 비고 |
|---|---|---|
| `kosis_chat.schemas.resources` | `LLMConfig(api_key: SecretStr \| None = None, model, provider="hcx", temperature=0.0, max_tokens=1024, max_input_tokens=None, timeout_seconds=60.0, base_url=DEFAULT_BASE_URL, thinking_effort="none")`, `LLMConfig.from_settings(settings, api_key=None)`, `require_key()` | 기본값 = 옛 model.yaml `model` 블록. frozen·extra 금지. 키는 repr·로그·`model_dump`에 안 나옴 |
| | `LLMKeyMissing` | `RuntimeError`(summarize_stat 계약) 겸 `ValueError`(옛 `HCXClient` 키 없음 오류) 하위 |
| | `DEFAULT_MODEL DEFAULT_BASE_URL MOCK_PROVIDER MOCK_RESPONSE` | 옛 model.yaml 값 |
| `kosis_chat.llm.providers` (신규) | `PROVIDERS`(`{"hcx": HCXClient, "mock": MockClient}`), `register_provider(name, cls)`, `build_client(config)` | provider가 mock이 아니고 키가 없으면 `build_client`에서 `LLMKeyMissing`. `cls.from_config(config)`로 만든다(`LLMClient` 기본 `cls(config)`, Mock은 `MOCK_RESPONSE`). `Settings`를 넘기면 `llm.model.build_client(settings)` |
| `kosis_chat.llm.model` | `HCXClient(config=None, sleep, observer, *, settings=None)` | 첫 인자 `LLMConfig` 또는 `Settings`(호환: 키는 그 `api_key_env` 셸 환경변수). 옛 키워드 `settings=`도 받음. `client.settings`(Settings)·`client.config`(LLMConfig) 둘 다 있음. `build_client(settings)`는 그대로(`LLMConfig`도 받음) |
| `kosis_chat.config` | `load_settings(path=None, **overrides)` | 파일 없이 같은 값(`backend="hcx"`, `model_name="HCX-DASH-002"`, `thinking_level="none"`, `mock_response`=`MOCK_RESPONSE` …). `path`를 주면 그 YAML을 예전처럼 읽음(`DeprecationWarning`) |
| | `load_env`, `get_secret` | Phase 2 전용, `DeprecationWarning`. Phase 1 코드는 호출하지 않음 |
| | `settings_from_llm(llm, **overrides)`, `env_llm_config(settings)` | 신규. Settings ↔ LLMConfig. 환경변수를 읽는 곳은 `env_llm_config`(Settings 호환 경로) 하나 |
| `kosis_chat.api` | `summarize_stat(..., llm=None)`, `_build(path, settings, client, retry_interval, llm=None)`, `_cached_llm` | `llm`은 키워드 끝에 추가(옛 호출 그대로). `llm`이 없으면 예전과 같은 인자로 `_build`·`_cached`를 부름 |
| `kosis_chat.pipeline` | `StatPipeline(..., llm=None)`, `.llm` | 클라이언트는 LLM 단계가 처음 쓸 때 만든다(`client` property). 규칙만으로 끝나는 거절·되묻기는 키 없이 동작. `Chatbot`·`RawRunner`의 `client`도 같은 방식(property + setter) |
| `eval.run_eval` | `llm_config(settings)` | `--backend hcx`면 시작 시 셸 환경변수 `HCX_API_KEY`를 읽고 없으면 exit 2. mock은 키 불필요. `--config`는 경고 후 무시. `eval.judge`도 같은 함수 |

## 단계 노드 — 순수 단계 (T1-28 S3a)

`StatPipeline.ask`의 순수 단계를 단계 폴더 `node.py`의 `step_<name>(inp, res: StatResources) -> Out`으로 옮겼다(LangGraph 없음, `ask`가 같은 순서로 부르는 수동 라우터). In/Out은 각 단계 `schemas.py`의 `BoundaryModel`(strict, extra 금지, 컨테이너는 같은 객체 보관). 표·색인은 In이 아니라 `StatResources`로 받는다. LLM 단계(`plan_fill_llm plan_write_llm event_lookup compose_llm fallback`)는 S3b에서 옮김(아래). `tests/test_nodes_pure.py`.

| 단계 함수 | In → Out | route |
|---|---|---|
| `kosis_chat.judge.node.step_judge` | `JudgeIn(question)` → `JudgeOut(verdict, route)` | `final` / `plan` |
| `kosis_chat.analyze.node.step_plan_rules(inp, res, *, build=build_plan)` | `PlanRulesIn(question, default_applied)` → `PlanRulesOut(rule, built, unresolved, route)` | `built` / `unresolved` |
| `kosis_chat.compute.node.step_compute` | `ComputeIn(plan, built, default_applied, computed=None)` → `ComputeOut(query, built, alternatives, computed, refusal, route)` | `ok` / `refused` |
| `kosis_chat.compose.node.step_caveats` | `CaveatsIn(computed, built, alternatives, default_applied)` → `CaveatsOut(context_ids, caveats, shown)` | — |
| `kosis_chat.compose.node.step_verify` | `VerifyIn(parts, computed, attempt)` → `VerifyOut(problems, route)` | `ok` / `retry` / `template` (`COMPOSE_MAX_ATTEMPTS = 2`) |
| `kosis_chat.compose.node.step_template` | `TemplateIn(type, computed)` → `TemplateOut(parts, problems, verified)` | — |
| `kosis_chat.chart.node.step_chart` | `ChartIn(type, computed)` → `ChartOut(chart)` (`kosis_ai.schemas.Chart`, S5에서 ChartSpec) | — |
| `kosis_chat.render.node.step_render` | `RenderIn(question, query, computed, parts, verified, source, default_applied, caveats, shown, context_ids, chart)` → `RenderOut(stat)` | — |
| `kosis_chat.render.node.step_answer_final` | `AnswerFinalIn(kind, question, verdict \| refusal+query+source \| clarify)` → `RenderOut(stat)` | kind `verdict` / `refused` / `clarify` |

- `step_plan_rules`의 키워드 전용 `build`: `StatPipeline`이 자기 모듈의 `build_plan`을 넘긴다(`tests/test_analyze_plan.py`가 `pipeline.build_plan`을 바꿔 모델 경로를 강제하므로). 그래프의 노드 교체(E4)도 같은 자리.
- span 이름·속성 키, `decision_metadata` 키, 재시도 횟수, 거절 문구는 그대로(mock 90문항 패리티 identical). `pipeline._causal_note`는 `render.node.causal_note`와 같은 함수.

옮긴 모듈(옛 경로는 옛 모듈의 모든 최상위 이름을 하나씩 재export, 같은 객체 — `test_moved_names_are_the_same_objects`):

| 새 위치 | 옛 경로(재export) |
|---|---|
| `kosis_chat.judge.rules` | `kosis_chat.judge.answerability` |
| `kosis_chat.judge.texts` | `kosis_chat.judge.messages` |
| `kosis_chat.analyze.rules` (rule_plan·build_plan·to_query 한 파일, rules/place 분할은 뒤로) | `kosis_chat.compute.analyze` |
| `kosis_chat.analyze.blocks` | `kosis_chat.compute.build` (`kosis_chat.compute.build` 모듈 객체 그대로: `from kosis_chat.compute import build`) |
| `kosis_chat.analyze.roles` | `kosis_chat.compute.roles` |
| `kosis_chat.compose.caveats` · `compose.verify` · `compose.template` | `kosis_chat.answer.caveats` · `answer.verify` · `answer.template` |
| `kosis_chat.chart.builders` (+ `chart.vega`: `SCHEMA HIGHLIGHT OTHER WIDTH _unit_title _base _color _tooltip _bars _row _timeline`) | `kosis_chat.answer.chart` |
| `kosis_chat.chart.grid` (`Grid`, `grid`) · `kosis_chat.render.table` (`table_part`) | `kosis_chat.answer.table` |
| `kosis_chat.render.text` (`render_parts default_sentence final_text` + 공용 재export) | `kosis_chat.answer.render` |
| `kosis_chat.render.source` (`to_source org_id_from_table_path _ORG_FILE _kosis_host`) | `kosis_chat.contract` (나머지 `strip_model_source render_text to_stat_answer`는 contract에 그대로) |

## 단계 노드 — LLM 단계 (T1-28 S3b)

모델을 부르는 단계도 `step_<name>(inp, res, *, llm: ModelAccess) -> Out`이다. 클라이언트는 `StatResources`에 없고(표마다 frozen, run_eval이 클라이언트를 바꿔 끼움) `kosis_chat.llm.calls.ModelAccess(client, model_name, retry_interval, max_input_tokens)`가 소유자(`StatPipeline`)의 `client` property를 **첫 호출 때** 읽는다(키 없는 hcx면 그때 `LLMKeyMissing`, S2c 그대로). 각 모델 단계는 한 일을 `LLMCall` 기록(`kosis_chat.schemas.state`)과 자기가 쓴 `decision_metadata` 항목(`meta`, 쓴 순서)으로 돌려주고, `ask`가 받은 순서대로 붙인다. In의 `called`는 "이 질문에 이미 모델 답이 왔음"(그러면 호출 전 `retry_interval` 대기, 옛 `_Calls.raw` 검사). `tests/test_nodes_llm.py`.

| 단계 함수 | In → Out | route · 호출 상한 |
|---|---|---|
| `kosis_chat.analyze.node.step_plan_fill_llm(inp, res, *, llm, build=build_plan)` | `PlanFillIn(rule, unresolved, called)` → `PlanFillOut(plan, built, calls, meta, route)` | `built` / `unplaced` · 2 (`analyze.llm_fill`, `analyze.llm_fill_retry`) |
| `kosis_chat.analyze.node.step_plan_write_llm(inp, res, *, llm)` | `PlanWriteIn(question, rule, called)` → `PlanWriteOut(placed, clarify, calls, meta, route)` | `placed` / `clarify` / `failed` · 계획 2 (`analyze.llm_plan`, `analyze.llm_plan_retry`) + 사건 n |
| `kosis_chat.analyze.node.step_event_lookup(inp, res, *, llm)` | `EventLookupIn(name, called)` → `EventLookupOut(start, call)` | plan_write_llm 안의 콜백(계산 중 호출), 질문당 사건마다 1 |
| `kosis_chat.compose.node.step_compose_llm(inp, res, *, llm)` | `ComposeLLMIn(question, computed, cautions, context_ids, attempt, problems, prompt, called)` → `ComposeLLMOut(parts, prompt, call, meta, route)` | `verify` / `template` · `ask`의 compose↔verify 루프가 `COMPOSE_MAX_ATTEMPTS`(=2, `compose/schemas.py` 하나)번 |
| `kosis_chat.legacy.fallback.step_fallback_chatbot(inp, res, *, chatbot, llm)` | `FallbackIn(question, golden_scope)` → `FallbackOut(answer, call)` | 챗봇 자체 2(답 + 형식 재시도). `call.summed=False`(챗봇 Answer가 자기 호출을 보고) |

- 기록 → 출력: `kosis_chat.graph.spans.spans_of(records, t0)`가 옛 `_Calls.span`과 같은 dict(이름·`start_ms`·`duration_ms`·status·attributes, `gen_ai.*`가 먼저, None 제외, 기록 순서). `kosis_chat.graph.legacy.to_answer(...)`·`fallback_answer(...)`가 옛 `_done`·fallback 합치기와 같은 `chatbot.Answer`(합계는 `totals_of`: 옛 `_Calls.add` 순서). `DECISION_KEYS`(29개, `spans` 제외)가 stat 답이 스스로 쓰는 키 집합(테스트 고정).
- `graph/legacy.py`는 옛 경로 `kosis_chat.chatbot`(Answer·judge_record)을 import한다: S6에서 챗봇이 legacy/로 갈 때까지 `tests/test_layout.py` `OLD_PATH_EXCEPTIONS`의 유일한 예외.
- `kosis_chat.pipeline`에 남긴 이름: `VERSION`(= `kosis_chat.llm.compose.VERSION`, 같은 객체), `FALLBACK_PROMPT`(legacy/fallback.py), `add_usage`(graph/legacy.py), `_llm_attributes`(graph/spans.py `llm_attributes`). 없앤 것: `_Calls`, `StatPipeline._ask_model _place _fill _build _written _lookup`(Phase 1·eval·Phase 2 grep 0건).

옮긴 모듈(같은 객체 재export, `test_moved_names_are_the_same_objects`):

| 새 위치 | 옛 경로(재export) |
|---|---|
| `kosis_chat.analyze.fill` | `kosis_chat.llm.analyze_fill` |
| `kosis_chat.analyze.write` (place 포함; `build`는 이제 `analyze.blocks` 모듈) | `kosis_chat.llm.analyze_plan` (`build`는 옛날엔 `kosis_chat.compute.build` shim 모듈, 함수는 같은 객체) |
| `kosis_chat.analyze.event_lookup` | `kosis_chat.llm.event_lookup` |
| `kosis_chat.compose.compose` | `kosis_chat.llm.compose` (`VERSION` 유지: `pipeline.VERSION`·Phase 2 `scripts/stat_baseline.py:53`) |

프롬프트(D15): `kosis_chat/analyze/prompts/{analyze-fill-v2,analyze-fill-v2-retry,analyze-plan-v1,analyze-plan-v1-examples,analyze-plan-v1-retry,event-lookup-v1}.md`, `kosis_chat/compose/prompts/{compose-v2,compose-v2-example,compose-v2-retry}.md`. 각 패키지 `prompts.py`의 `load(file)`(`importlib.resources` + `functools.cache`, 파일 끝 줄바꿈 하나 제거), `version(file)`(= 파일 이름에서 `.md` 뺀 것). 모듈 상수 `SYSTEM_PROMPT RETRY EXAMPLES EXAMPLE VERSION`은 import 때 읽은 값으로 그대로 있다. `pyproject.toml` package-data(`prompts/*.md`)로 wheel·`pip install -e`·PYTHONPATH 모두 같은 파일. `tests/test_prompts.py`가 옛 상수와 글자 단위 동일을 확인(이동 PR 전용, 이후 삭제).

**monkeypatch 주의(S3a·S3b 공통).** 옛 경로 shim은 이름을 복사해 둔 것이라, 옛 경로 모듈의 **속성을 다시 묶는** monkeypatch(`monkeypatch.setattr(kosis_chat.llm.compose, "SYSTEM_PROMPT", ...)`, `kosis_chat.compute.analyze.build_plan = ...` 등)는 새 모듈과 그 이름을 쓰는 코드에 닿지 않는다. 바꾸려면 새 위치(`kosis_chat.compose.compose` 등)를 패치하거나, 단계 함수의 키워드 인자(`build=`, `llm=`)로 넘긴다. 예외: `kosis_chat.pipeline.build_plan`은 `ask`가 부를 때마다 읽어 `step_plan_rules`·`step_plan_fill_llm`에 넘기므로 패치가 그대로 통한다(`tests/test_analyze_plan.py`). `write.place()`가 부르는 `written`도 이제 `kosis_chat.analyze.blocks`에서 찾으므로 `kosis_chat.compute.build.written` 패치는 닿지 않는다(현재 쓰는 곳 없음). 같은 객체를 **안에서 바꾸는** 패치(`setattr(SomeClass, ...)`, `time.sleep`)는 양쪽에 다 보인다.

## LangGraph 그래프 (T1-28 S4)

`StatPipeline.ask`는 이제 `kosis_chat.graph.build`의 그래프를 돈다(수동 라우터는 `tests/test_graph_parity.py`의 비교 기준 `sequential_ask`로만 남음). 반환 `Answer`·span·`decision_metadata`(키 순서 포함)·모델 호출 순서·`time.sleep` 호출은 S3b와 같다. `langgraph>=1.0,<2`가 Phase 1 의존성이 되었고(`requirements.txt`·`pyproject.toml`), `import kosis_chat.pipeline`이 langgraph를 불러온다. langgraph import는 `kosis_chat/graph/` 안에서만(`tests/test_layout.py`).

| 위치 | 이름 | 비고 |
|---|---|---|
| `kosis_chat.graph.build` | `NODE_TABLE`(노드 이름 → step 함수 13개), `build_graph(resources, node_overrides=None)`, `run(graph, question, *, context, meta, started, golden_scope=None) -> StatState`, `RECURSION_LIMIT`, `LONGEST_PATH`, `StatContext` | `node_overrides`는 같은 서명의 step으로 바꿔 끼움(E4), 모르는 이름은 `KeyError`. 그래프는 `StatPipeline.graph`(첫 `ask` 때 한 번 만듦) |
| `kosis_chat.graph.adapters` | `StatContext(llm, chatbot, build)`, `timed(step, *args, **kw)`, `*_node(step, res)` 어댑터 | 어댑터는 dict를 돌려준다(BaseModel 반환 안 씀). 질문마다 바뀌는 것(모델 접근·fallback 챗봇·`pipeline.build_plan`)은 LangGraph context로 |
| `kosis_chat.schemas.state` | `StatState`, 리듀서 `append`(records)·`merge`(meta) | Pydantic. 노드 이름과 겹치지 않게 `chart_out`·`caveats_out` |
| `kosis_chat.graph.legacy` | `answer_of(state, *, table_meta, prompt_version, fallback_prompt)` | 옛 `ask`의 출구별 `to_answer`/`fallback_answer` + 답한 질문의 `Answer.chart` |
| `kosis_chat.pipeline` | `StatPipeline.graph`(신규 property) | 없앤 것: `StatPipeline._compose _verdict_answer _clarify_answer _done`, 모듈의 `step_*`·`to_answer` 등 재import(Phase 1·eval·Phase 2 grep 0건) |

- chart 노드는 caveats 뒤에서 compose 경로와 병렬로 돈다(같은 superstep이면 LangGraph가 스레드로 돌림). chart는 `chart_out`만 쓰고 records·meta는 안 쓴다 → span 순서는 완료 순서와 무관하게 옛 순서. render는 `["compose_done", "chart"]` 합류 뒤에만.
- 노드 안에서 난 예외는 같은 객체로 `ask` 밖까지 나온다(LangGraph가 `add_note`로 "During task with name ..." 한 줄을 붙여 traceback에만 보임). `LLMKeyMissing`도 첫 LLM 노드에서 그대로.
- `pipeline.build_plan` monkeypatch는 그대로 통한다(`ask`가 매번 읽어 context로 넘김).

## 반환 객체에서 읽는 속성

- `StatPipeline`: `ask(question, golden_scope=None)`, `client` (setter: `eval/run_eval.py:229`), `org_id table prompt_version system_prompt answer_model decision_metadata settings retry_interval`. T1-28 S2부터 `resources`(StatResources)와 `index`(= `resources.index`)도 있음. S2c부터 `llm`(주입한 LLMConfig 또는 None)이 있고, `client`는 처음 읽을 때 만든다(키 없는 hcx면 그때 `LLMKeyMissing`).
