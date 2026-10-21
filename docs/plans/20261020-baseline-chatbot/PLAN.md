# Plan: 베이스라인 챗봇 (표 데이터 입력 → 답변)

- id: 20261020-baseline-chatbot
- status: APPROVED rev 3 (2026-10-20, 전민기)
- date: 2026-10-20 (rev 3: 코드 작성 전 설계 수정 권고 반영)
- related: docs/plans/20261020-stage1-table-summary/PLAN.md (rev 3)

## Intent

가장 단순한 형태로 "통계표 데이터 + 질문 → HCX → 답변"이 끝까지 도는 챗봇을 먼저 만든다.
이후 1차 PLAN의 기능(전처리 규칙, 질문 분석, 코드 계산, 결합 검증)을 파일 하나씩 붙이며 키운다.
숨은 요구는 두 가지다: (1) 나중에 다시 봐도 어디에 무엇이 있는지 바로 알 수 있도록 역할별 파일·폴더로 분리,
(2) 유효한 기준 점수를 남겨 이후 개선 효과를 숫자로 보여주기(개인 기술 성과 근거).
B0의 목적은 점수의 높고 낮음이 아니라 같은 조건으로 다시 잴 수 있는 기준 점수 확보다.

## 베이스라인이 하는 일 / 하지 않는 일

| 한다 | 하지 않는다 (이후 단계에서 추가) |
| --- | --- |
| 지원 입력 형식을 읽어 공통 자료형 TableInput으로 변환 | 질문 조건으로 행 필터링 |
| 최소 정리(값 문자열 → 숫자 변환, 원본 보존) | 계/소계·결측·단위 규칙 (PREPROCESSING.md) |
| 표 전체를 마크다운으로 바꿔 프롬프트에 넣기 (길이 초과 시 잘라 넣지 않고 오류) | 질문 유형 분석, 모호성 처리 |
| 시스템 프롬프트로 "원본 셀 또는 검증 가능한 계산식과 입력 셀로 추적되는 수치만 사용, 출처 표기" 지시 | 코드 계산(facts), 결합 검증 |
| 같은 표로 여러 질문을 묻는 CLI 대화 | 이전 대화 기억(최종 단계) |
| 골든셋 답변 저장 + 자동 보조 검사 + 사람 최종 판정으로 기준 점수 집계 | 자유 문장 완전 자동 채점, 통계표 검색(2차 RAG) |

## 폴더·파일 구조

```
kosis-stat-chat/
├─ kosis_chat/                  # 서비스 코드 (Python 패키지)
│  ├─ config.py                 # .env(API 키)와 configs/*.yaml 읽기
│  ├─ chatbot.py                # 전체 흐름 조립: 읽기 → 정리 → 직렬화 → 프롬프트 → 모델 → 답변
│  ├─ cli.py                    # 터미널 실행 진입점 (표 하나 지정 → 질문 반복)
│  ├─ data/
│  │  ├─ schema.py              # TableInput·Cell·TableMeta 정의와 검사 (아래 "입력 형식과 TableInput")
│  │  ├─ loader.py              # 지원 입력 형식별 어댑터 → TableInput
│  │  ├─ preprocessing.py       # 베이스라인: 최소 정리 / 이후: PREPROCESSING.md 규칙
│  │  └─ serializer.py          # TableInput → 프롬프트용 텍스트 (마크다운, 이후 JSON·문장형 추가)
│  └─ llm/
│     ├─ model.py               # LLM 클라이언트: HCXClient, MockClient (같은 인터페이스, 호출 상태 반환)
│     └─ prompts.py             # 시스템 프롬프트(버전별), 답변 형식 템플릿
├─ configs/
│  └─ model.yaml                # 백엔드(hcx/mock), 모델 이름, temperature, max_input_tokens, eval.runs
├─ eval/                        # HCX 품질 평가 (verify.sh에서 새 HCX 호출 안 함)
│  ├─ run_eval.py               # 고정 조건으로 HCX 실행 → 답변·자동 보조 검사·manifest 저장
│  ├─ review.py                 # 사람 판정 입력 시트 생성/반영 (pending → confirmed)
│  ├─ scorer.py                 # 확정 판정 집계 (자동 판정과 사람 판정 구분)
│  ├─ golden/                   # 골든셋 문항 (실제 수치 포함 → 커밋 안 함)
│  └─ results/                  # 실행별 답변·판정 (실제 수치 포함 → 커밋 안 함)
├─ tests/                       # pytest, MockClient만 (verify.sh에서 실행, 네트워크·키 불필요)
│  └─ fixtures/                 # 직접 만든 가상 표 (커밋)
├─ notebooks/
│  └─ colab_run.ipynb           # Colab에서 패키지 불러와 실행만 (로직 없음)
├─ data/
│  ├─ raw/                      # 받은 원본 (커밋 안 함)
│  └─ samples/                  # 평가·실행용 실제 표의 유일한 위치 (커밋 안 함)
├─ .env.example                 # 필요한 키 이름만 (값 없음)
└─ requirements.txt
```

이후 단계에서 추가될 파일 (지금은 만들지 않음, 위치만 예약):

| 단계 | 추가 파일 | 역할 |
| --- | --- | --- |
| 1차-B1 | data/preprocessing.py 확장 | PREPROCESSING.md R1~R9 |
| 1차-B2 | nlu/analyzer.py | 질문 유형·조건·모호성 분석 |
| 1차-B3 | analysis/planner.py, analysis/calculator.py | 계산 계획(분자·분모·기준 시점) → 근거 행 선택 → facts |
| 1차-B4 | analysis/validator.py, llm/render.py | fact 기반 수치 문장 템플릿 + 결합 검증 |
| 2차 | retrieval/ (indexer.py, retriever.py) | 메타데이터 RAG로 표 찾기 |
| 최종 | viz/chart.py, memory/session.py, web/ | 차트, 연속 대화, UI |

`chatbot.py`의 흐름 함수는 단계마다 한 줄씩 늘어나도록 만들어, 어느 단계에서 무엇이 추가됐는지 코드에서 바로 보이게 한다.

## 입력 형식과 TableInput

지원 입력 형식(파일)과 내부 공통 자료형(TableInput)을 구분한다. 모든 어댑터는 같은 TableInput을 반환하고, 이후 모듈은 TableInput만 본다.
누락된 메타는 모델이 추측해 채우지 않는다. 메타가 없으면 거부한다.

### 지원 입력 형식 (B0)

| 형식 | B0 지원 | 메타 출처 | 메타 부족 시 |
| --- | --- | --- | --- |
| A. `<name>.csv` + `<name>.meta.json` | 지원 | meta.json | 거부 (누락 필드 이름과 함께) |
| B. KOSIS OpenAPI JSON 응답 + `<name>.meta.json` | 지원 | 응답에서 얻을 수 있는 필드 + meta.json으로 보충 | 거부. 실제 응답 확인 전에는 응답만으로 메타가 채워진다고 가정하지 않는다 |
| C. KOSIS 웹 다운로드 wide CSV (시점이 열) | **미지원** | — | 거부. B0에서는 변환 스크립트도 제공하지 않는다 |

공통: 인코딩 UTF-8 (CP949면 읽기 실패로 거부하고 이유 표시), 구분자 쉼표, 헤더 1줄.

형식 A의 CSV 컬럼 (long 형식, 한 행 = 한 셀):

| 컬럼 | 필수 | 설명 |
| --- | --- | --- |
| item | 예 | 항목명 |
| dim:<분류명> | 0개 이상 | 분류마다 한 컬럼, 값은 분류 이름 (예: `dim:성별` = 남자) |
| code:<분류명> | 선택 | 원본 분류 코드가 있으면 보존 |
| period | 예 | 원본 시점 문자열 (2024, 202401, 2024Q1 …) |
| value | 예 | 원본 값 문자열 그대로 ("-", "x" 포함) |
| unit | 예 | 그 셀의 단위 |

### TableInput 필드 정의 (한 곳에서 정의, schema.py가 그대로 구현)

`TableInput = TableMeta + list[Cell]`

TableMeta:

| 필드 | 필수 | 자료형 | 원본/파생 | 누락 시 |
| --- | --- | --- | --- | --- |
| tbl_id | 예 | str | 원본 | 거부 |
| tbl_nm | 예 | str | 원본 | 거부 |
| survey_nm | 예 | str | 원본 | 거부 |
| org_nm | 예 | str | 원본 | 거부 |
| period_type | 예 | Y / M / Q / H | 원본 | 거부 |
| source_url | 예 | str | 원본 | 거부 |
| fetched_at | 예 | date (YYYY-MM-DD) | 원본 | 거부 |
| input_format | 예 | A / B | 파생 (loader) | — |
| file_sha256 | 예 | str | 파생 (loader) | — |
| note | 아니오 | str | 원본 | 빈 값 |
| units_summary | 아니오 | list[str] | 파생 (셀 단위 목록, 표시용) | 계산 |
| period_range | 아니오 | (str, str) | 파생 (셀 시점의 최소·최대) | 계산 |

Cell:

| 필드 | 필수 | 자료형 | 원본/파생 | 누락 시 |
| --- | --- | --- | --- | --- |
| item | 예 | str | 원본 | 거부 |
| dims | 예 | dict[분류명, 이름] (빈 dict 가능) | 원본 | — |
| dim_codes | 아니오 | dict[분류명, 코드] | 원본 (있을 때만) | 빈 dict |
| period_raw | 예 | str | 원본 | 거부 |
| value_raw | 예 | str | 원본 (변경 금지) | 거부 |
| unit | 예 | str | 원본 (셀 단위가 기준) | 거부 |
| value | 예 | float 또는 None | 파생 (preprocessing) | None |
| value_flag | 예 | ok / missing / suppressed / provisional / unparsed | 파생 | — |
| source_ref | 예 | str | 파생: 원본 코드가 있으면 tbl_id+코드+시점, 없으면 file_sha256+원본 행 번호 | — |

- 단위는 셀 단위가 기준이다. 표 수준 단위는 `units_summary`(파생, 표시용)일 뿐 계산에 쓰지 않는다.
- `value_raw`는 어떤 단계에서도 바꾸지 않는다. 숫자 변환 결과는 `value`·`value_flag`에만 둔다.
- 전처리 이후에도 `unit`, `value_raw`, `source_ref`가 유지되어야 한다 (테스트로 확인).

## 데이터 저장·커밋 정책

같은 실제 데이터가 경로만 달라져 커밋되는 일이 없도록, 실제 수치를 담는 것은 모두 커밋하지 않는다.

| 경로 | 내용 | 실제 수치 | 커밋 |
| --- | --- | --- | --- |
| data/raw/ | 받은 원본 | 있음 | 안 함 |
| data/samples/ | 실행·평가용 실제 표의 유일한 위치 | 있음 | 안 함 |
| eval/golden/ | 골든셋 문항·기대 facts | 있음 | 안 함 |
| eval/results/ | HCX 답변, 자동 검사, 사람 판정 | 있음 | 안 함 |
| tests/fixtures/ | 직접 만든 가상 표 | 없음 (가상) | 함 |
| docs/reports/eval-history.md, manifest 요약 | 집계 점수, 파일 해시, 버전 | 없음 | 함 |

- 평가는 실제 표를 복사하지 않고 `data/samples/`의 파일을 경로 + sha256으로 참조한다. 해시가 다르면 평가를 거부한다.
- 골든셋·결과의 공유 범위는 담당자 확인 후 바꿀 수 있다 (바꾸면 이 표를 먼저 수정).

## 프롬프트 원칙

- 수치 규칙(전 문서 공통 문구): **"원본 셀 또는 검증 가능한 계산식과 입력 셀로 추적되는 수치만 사용한다."**
- 계산값은 계산식과 입력값을 함께 쓴다.
  예: "2023년 대비 2024년 증가율 2.1% ((51,300 − 50,240) ÷ 50,240 × 100)"
- 금지: 표에 없는 값의 추정·외삽, 입력이 표에 없는 계산, 원인·정책 판단.
- 필수: 출처(조사명·표명), 답할 수 없으면 통계 수치 없이 이유와 가능한 범위.
- 프롬프트는 `prompts.py`에 버전 이름(`baseline-v1` …)으로 저장하고, 평가 결과에 버전을 남긴다.

## 입력 길이와 API 오류

- 입력 길이는 행 수가 아니라 사용 모델의 토큰 한도로 판단한다. 한도는 `configs/model.yaml`의 `max_input_tokens`(HCX 확인 후 기입).
- 한도를 넘으면 표를 잘라 넣지 않고 `input_too_long` 상태로 기록한다.
- 모델 호출 결과 상태: `ok` / `timeout` / `api_error` / `input_too_long`. 정상 답변이 아닌 상태는 답변 텍스트와 구분해 저장한다.
- 오류 상태 문항은 평가에서 제외하지 않는다. 집계에 "실행 실패" 건수로 따로 표시하고, 해당 문항은 FAIL로 센다.

## 비교 규약 (B0~B4 공통)

단계 간 점수 차이가 "바꾼 것" 때문임을 보장하기 위해 아래를 고정한다.

| 고정 대상 | 규칙 |
| --- | --- |
| 데이터 | `data/samples/`의 평가용 표 세트, 파일 sha256을 manifest에 기록. 해시 불일치 시 평가 거부 |
| 골든셋 | 버전 고정(v1). 문항 수정 시 v2를 만들고 이전 단계도 v2로 다시 돌린다 |
| 채점 기준 | 이 문서의 채점 기준을 B0~B4 동일 적용. 기준 수정 시 전 단계 재판정 |
| 모델 | 같은 HCX 모델 이름·버전, temperature 0, 같은 max tokens. 바뀌면 전 단계 재측정 |
| 실행 | 문항당 실행 횟수 `eval.runs`(기본 1). 2회 이상이면 각 실행 결과를 모두 보존하고 평균·변동(최소~최대)을 보고. 다수결 점수는 단일 호출 점수와 구분해 표시 |
| 변경 | 한 단계에서 바꾸는 것은 하나(B1 전처리, B2 질문 분석 …). 프롬프트 변경도 한 단계로 취급 |

모든 실행은 `manifest.json`을 남긴다: git commit, 골든셋 버전, 데이터 해시, 채점 기준 버전, 모델·파라미터, 프롬프트 버전, 단계 이름, 실행 시각.

## 채점 기준

### 채점 대상: 통계 수치 주장

모든 숫자 토큰이 아니라 **통계 수치 주장**만 채점한다.

- 통계 수치 주장: 표의 값이나 그 값에서 계산한 값을 말하는 숫자 (인구 수, 비율, 증감률, 차이 등)
- 채점 대상 아님: 연도·월 등 시점 표기, 연령 범위(30~34세), 목록 번호, 표 ID·출처 식별자, "2025년 자료가 없습니다" 같은 범위 안내 속 시점

### 골든 문항 기록 필드

{id, 질문, 표(sha256), 문항 종류, 기대 처리, 필수 facts[], 보조 facts[], 기대 불가 사유 범주, 허용 기본값, 확인해야 할 조건}

각 fact: {값, 단위, 시점, 분류, 허용 계산식(원본 셀이면 "원본"), 입력 셀(source_ref 목록), 반올림 규칙, 허용오차}

### 단위별 허용오차 (문항에 다른 값이 없을 때의 기본)

| 결과 단위 | 기본 허용오차 |
| --- | --- |
| 명·가구·건 등 수량 (원본 셀) | 정확 일치 (쉼표·"만 명" 표기 차이는 같은 값으로 인정) |
| 명·가구 등 수량 (계산값: 차이·합계) | 정확 일치 |
| 천 명 등 배수 단위 | 표기 단위의 마지막 자리 ±1 |
| % (구성비·증감률) | 소수 1자리 반올림 기준 ±0.05%p |
| %p (비율의 차이) | ±0.05%p, %와 %p 혼동은 오답 |

### 문항 종류와 판정 (상호 배타)

판정 순서: **FAIL 조건 → PASS 조건 → 나머지는 PARTIAL**. 앞 단계에서 정해지면 뒤는 보지 않으므로 한 답변이 두 판정에 동시에 해당하지 않는다.

| 문항 종류 | FAIL (먼저 검사) | PASS | PARTIAL |
| --- | --- | --- | --- |
| 답변 가능 | 필수 fact 하나라도 틀림(값·단위·시점·분류), 근거 없는 통계 수치 주장, 불필요한 거절, 실행 실패 | 필수 facts 모두 맞음 + 출처 있음 + 보조 facts 누락 없음 | 필수 facts는 맞지만 보조 정보나 출처 누락 |
| 답변 불가 | 없는 통계 수치를 제시, 다른 사유로 단정(예: 있는 자료를 없다고), 실행 실패 | 기대 사유 범주로 한계를 안내 | 수치 없이 보류했지만 사유 설명이 불충분 |
| 되묻기 필요 | 임의로 조건을 정해 통계 수치로 답함, 실행 실패 | 지정한 조건을 모두 확인하는 질문 | 확인 질문이 일부 조건만 다룸 |
| 기본값 허용 | 허용하지 않은 기본값 사용, 수치 오류, 실행 실패 | 허용 기본값을 명시하고 수치가 맞음 | 수치는 맞지만 기본값 명시 누락 |

코드상 문항 종류 이름: 답변 가능 = `answerable`, 답변 불가 = `not_answerable`, 되묻기 필요 = `clarify`, 기본값 허용 = `default_allowed`.

오류 유형(복수 가능)과 판정 효과 (`eval/scorer.py`의 RULES가 원본):

| 문항 종류 | FAIL로 만드는 유형 | PARTIAL로 만드는 유형 |
| --- | --- | --- |
| answerable | wrong_value, wrong_unit, wrong_period, wrong_dims, unsupported_claim, false_refusal, run_failure | missing_aux, no_source |
| not_answerable | missed_refusal, unsupported_claim, wrong_reason, run_failure | insufficient_reason |
| clarify | assumed_condition, unsupported_claim, run_failure | missing_clarification |
| default_allowed | wrong_value, wrong_unit, wrong_period, wrong_dims, disallowed_default, unsupported_claim, run_failure | unstated_default, no_source |

검토자는 오류 유형만 적으면 판정이 규칙으로 정해지고, 판정을 직접 적었는데 규칙과 다르면 반영을 거부한다.

### B0 채점 절차 (사람 최종 판정 중심)

자유 문장 완전 자동 채점은 B0 범위가 아니다. 자동 검사는 보조만 한다.

1. `run_eval.py`가 질문·기대 정답·원본 답변·호출 상태를 저장한다.
2. 자동 보조 검사(명확한 것만): 출처 문자열 포함 여부, 답변 속 숫자 후보 목록과 기대 fact 값 일치 후보, 호출 상태. 결과는 `automatic_checks`에 참고용으로만 저장한다.
3. 사람이 `review.py`가 만든 시트에서 최종 판정과 오류 유형을 확정한다.
4. `scorer.py`는 확정(confirmed) 판정만 집계한다.

문항 결과 기록 필드:

```
question_id, run_index, call_status, raw_answer, automatic_checks,
review_status: pending / confirmed, final_verdict: PASS / PARTIAL / FAIL,
error_types, review_note, reviewer
```

- 미검토(pending) 문항을 제외하거나 정답으로 집계하지 않는다. 집계에는 전체·확정·미검토 수를 함께 표시하고, 미검토가 1개라도 있으면 점수를 **잠정**으로 표시한다.
- 서비스 내부 검증기 통과율(B4 이후)과 외부 정답률(이 채점)을 별도 지표로 둔다.

## Mock 테스트와 HCX 평가 분리

| 구분 | tests/ (오프라인) | eval/ (HCX) |
| --- | --- | --- |
| 목적 | 코드가 규약대로 동작하는가 | 챗봇 답변 품질이 얼마인가 |
| 모델 | MockClient (고정 응답) | HCXClient |
| 실행 | `bash scripts/verify.sh` (항상, 네트워크·키 불필요) | `python -m eval.run_eval --stage B0` (수동, 새 HCX 호출) |
| 저장된 결과 재채점 | `python -m eval.scorer --run <id>` 는 오프라인 가능 | — |
| 결과 | 통과/실패 | 답변·판정·manifest, 점수는 eval-history.md에 누적 |

Mock 테스트 통과만으로 HCX 검증 완료를 선언하지 않는다.

## 주요 인터페이스 (설계만)

- `schema.TableInput`, `schema.TableMeta`, `schema.Cell`: 위 필드 정의 그대로, `schema.validate(table) -> list[Problem]`
- `loader.load_table(path) -> TableInput`: 형식 A·B 판별 → 어댑터 → `schema.validate`, 문제가 있으면 이유와 함께 예외
- `preprocessing.clean(table) -> TableInput`: `value`·`value_flag`만 채우고 원본 필드는 유지
- `serializer.to_markdown(table) -> str`: 메타 헤더 + 셀 단위가 보이는 표
- `model.LLMClient.generate(system, user) -> LLMResult(text, status, usage)` / 구현체 `HCXClient`, `MockClient`
- `chatbot.Chatbot(table).ask(question) -> Answer(text, status, table_meta, prompt_version)`
- 1차 입력이 "데이터"이므로 `Chatbot`은 표를 생성 시점에 받는다. 2차에서는 retriever가 `TableInput`을 만들어 같은 `Chatbot`에 넘긴다.

## Requirements

FR-001: 지원 입력 형식 A·B를 어댑터로 읽어 같은 TableInput을 만들고, 미지원 형식(C)·메타 누락·필수 컬럼 누락·인코딩 오류는 이유와 함께 거부한다.
FR-002: 최소 정리: `value_raw`를 보존한 채 `value`·`value_flag`를 채운다. 셀 단위·source_ref를 유지한다.
FR-003: 메타 헤더(조사명·표명·기관·기간) + 셀 단위가 보이는 마크다운 표 + 질문으로 프롬프트를 만든다.
FR-004: 시스템 프롬프트는 "프롬프트 원칙"을 따른다.
FR-005: HCX와 Mock 두 백엔드를 configs/model.yaml로 전환하고, 호출 상태(ok/timeout/api_error/input_too_long)를 반환한다.
FR-006: CLI에서 표 하나를 지정하고 질문을 반복해서 할 수 있다.
FR-007: run_eval.py가 비교 규약대로 HCX를 실행해 문항 결과 기록 필드와 manifest를 저장한다.
FR-008: review.py가 사람 판정 시트를 만들고 확정 판정을 결과에 반영한다.
FR-009: scorer.py가 확정 판정만 집계하고 전체·확정·미검토 수와 잠정 여부, 실행 실패 수를 표시한다.
FR-010: 단계별 결과를 docs/reports/eval-history.md 한 표에 누적한다 (행 = B0, B1, …).
NFR-001: 파일 하나 = 역할 하나. 다른 모듈의 내부에 의존하지 않고 위 인터페이스만 사용한다.
NFR-002: API 키는 .env로만 읽고 코드·노트북·로그에 출력하지 않는다.
NFR-003: Python 3.12, 코드·주석 영어. Colab(VS Code 연결)에서 `pip install -r requirements.txt` 후 바로 실행된다.
NFR-004: tests/는 HCX를 호출하지 않고 네트워크·키 없이 돈다.
NFR-005: 실제 수치를 담는 파일은 "데이터 저장·커밋 정책"에 따라 커밋되지 않는다.

## Acceptance Criteria

오프라인 (verify.sh로 확인):

AC-001: pytest(Mock)가 schema·loader·preprocessing·serializer·chatbot·scorer·review 테스트를 통과한다. 미지원 형식·메타 누락·인코딩 오류 거부 테스트와, 전처리 후 unit·value_raw·source_ref 유지 테스트 포함.
AC-002: scorer가 판정 순서(FAIL→PASS→PARTIAL)를 지키고, 미검토 문항이 있으면 잠정으로 표시하며, 미검토·실행 실패 문항을 제외하지 않는다 (테스트).
AC-003: `bash scripts/verify.sh`가 네트워크·API 키 없이 0으로 종료한다.
AC-004: 각 파일 첫 줄 docstring에 역할이 한 문장으로 적혀 있다.
AC-005: `git check-ignore`로 data/samples, eval/golden, eval/results가 무시됨을 확인한다.

HCX (수동 실행 증거로 확인, verify.sh로는 확인 불가):

AC-006: 실제 통계표 1개(형식 A 또는 B)로 `python -m kosis_chat.cli --table <표>`가 HCX 답변을 낸다.
AC-007: B0 run_eval 실행 결과와 manifest가 있고, 모든 문항이 confirmed 판정되어 eval-history.md 첫 행에 확정 점수(잠정 아님)가 남는다.
AC-008: 실행 실패·미실행 문항 수가 eval-history.md에 명시된다.

## Technical Decisions

D-001: 패키지 + 역할별 하위 폴더(data/, llm/), 노트북은 실행만
- Rationale: 노트북에 로직이 섞이면 재사용·테스트가 어렵다. Colab은 패키지를 불러 쓰기만 한다
- Rejected alternative: 단일 노트북 — 빠르지만 2차·최종 확장 시 분리 비용이 큼

D-002: LLM 클라이언트 인터페이스 + MockClient
- Rationale: HCX 접근 확인 전에도 개발·테스트 가능, 이후 모델 교체도 쉬움

D-003: 베이스라인은 표 전체를 넣는다 (필터 없음), 길이 초과면 잘라 넣지 않고 input_too_long
- Rationale: 개선 전 기준점. 길이 초과·실패 자체가 전처리 필요성의 근거 자료가 된다
- Risk: 인구총조사의 큰 표 → B0 대상은 모델 한도 안의 작은 표로 한정

D-004: 설정은 configs/model.yaml, 비밀은 .env
- Rationale: 코드 수정 없이 모델·파라미터 변경, 키 유출 방지

D-005: 기능 테스트(Mock)와 품질 평가(HCX) 분리, 완료 조건도 분리
- Rationale: verify.sh가 외부 API·비용·비결정성에 묶이지 않고, 점수는 고정 조건에서만 비교

D-006: 지원 입력 형식과 내부 자료형을 분리하고, 메타는 반드시 meta.json으로 받는다
- Rationale: 출처 표기와 추적이 입력 단계에서 보장됨. 응답만으로 메타가 채워진다고 가정하지 않음
- Rejected alternative: 메타 누락 시 모델이 표명·기관을 추측 — 출처 오류 위험

D-007: B0 채점은 사람 최종 판정, 자동 검사는 보조
- Rationale: 자유 문장의 값·시점·분류 자동 연결은 B0에서 신뢰할 만큼 만들기 어렵다. 판정 기준을 고정하고 사람이 확정하는 편이 B0~B4 비교에 더 정확하다

## Scope

### In scope
- 위 구조의 파일 생성과 최소 구현, 테스트, CLI, 노트북
- run_eval·review·scorer와 B0 기준 점수 확보 1회
- verify.sh에 pytest 등록, requirements.txt, .env.example, .gitignore 정책 반영

### Non-goals
- 1차 PLAN의 B1~B4 기능, 2차·최종 기능
- wide CSV 변환, 자유 문장 완전 자동 채점
- 웹 UI, 대화 기억

## 1차 PLAN과의 관계

이 베이스라인은 1차 PLAN 작업 순서의 맨 앞(B0)이다.

1. **B0 (먼저 완료):** 실제 표 1개로 CLI 실행 + 그 표의 골든셋 v1로 확정 기준 점수 확보. 이게 끝나기 전에는 B1을 시작하지 않는다.
2. **B1~B4:** B1 전처리 규칙 → B2 질문 분석 → B3 계산 계획·코드 계산 → B4 fact 기반 수치 문장 + 결합 검증. 각 단계 끝마다 같은 골든셋·데이터·모델·채점 기준으로 재평가해 eval-history.md에 한 행씩 누적한다.
3. B4까지 쌓인 결과가 1차 PLAN의 단일 표 E2E 게이트 근거가 된다. 1차 서비스의 목표 성능 기준은 1차 PLAN에 따로 있고, B0 기준 점수와 섞지 않는다.

eval-history.md 형식:

| 단계 | 바꾼 것 | 문항(전체/확정/미검토) | 잠정 | 답변 가능 PASS | 답변 불가 PASS | 되묻기 PASS | 기본값 PASS | 근거 없는 주장 | 실행 실패 | runs | manifest |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| B0 | 베이스라인 | | | | | | | | | 1 | |

## 일정과 지연 시 대응

| 확정할 것 | 기한 | 지연 시 |
| --- | --- | --- |
| HCX 호출 방식 | 10/21 | 10/22까지 안 되면 B0 실행을 미루고 B1 구현을 Mock으로 먼저 진행. 1차 PLAN의 직렬화 실험을 가장 먼저 제외 |
| B0 대상 표 1개 | 10/21 | 가상 fixture로 개발 계속, B0 실행만 대기 |
| 골든셋 v1 담당·내용 | 10/22 | 민기님이 B0용 최소 세트(약 17문항)를 먼저 작성하고 평가 엔지니어와 이후 보강 |

## Risks

- HCX API 형식이 아직 미확인 → model.py만 바꾸면 되도록 인터페이스 고정
- 사람 판정이 병목 → B0 문항 수를 약 17개로 제한, 판정 시트에 기대 facts를 같이 보여줌
- Google Drive 위 레포에서 Colab 실행 시 경로 문제 → config.py가 프로젝트 루트를 기준으로 경로 계산

## 결정 사항 (2026-10-20)

- 패키지 이름: `kosis_chat`
- 실제 표·골든셋·평가 결과: 커밋하지 않음 ("데이터 저장·커밋 정책"). 테스트는 `tests/fixtures/`의 가상 표 사용
- 평가 실행 횟수: 기본 1회 (`eval.runs`), 비용 확인 후 조정

## B0 완료 전까지 확정할 것 (코드 작성은 막지 않음)

- HCX 호출 방식(엔드포인트, 모델 이름, 토큰 한도): 2026-10-21 → 그전까지 `HCXClient`는 인터페이스만
- B0 대상 표 1개: 2026-10-21
- 골든셋 v1 내용과 작성 담당: 2026-10-22

## Open Questions

- (없음 — 남은 항목은 위 "B0 완료 전까지 확정할 것"과 "일정과 지연 시 대응"에서 관리)
