# **연구 계획서 · 국가통계 데이터 기반 AI 챗봇 및 맞춤형 통계 시각화 서비스**

> 갱신 2026-10-25 · 액션 아이템: [docs/action-items-20261025.md](docs/action-items-20261025.md)
>
> 이전 버전: [10-23](docs/readme-history/README-2026-10-23.md) · [10-21](docs/readme-history/README-2026-10-21.md)

---

**프로젝트명: 국가통계 데이터 기반 AI 챗봇 및 맞춤형 통계 시각화 서비스 개발**

국가데이터처(KOSIS)에서 제공하는 통계표 메타데이터와 통계 수치 데이터를 활용하여, 사용자가 자연어로 원하는 통계 정보를 검색하고 분석 결과를 확인할 수 있는 AI 서비스를 개발한다.

기존의 통계표 검색 및 안내 기능에서 나아가, **통계 수치를 AI가 분석하고 시각화 자료와 요약 결과까지 제공하는 서비스**를 구현하는 것이 최종 목표이다.

프로젝트는 총 3개 단계로 구성된다.

| 구분 | 프로젝트 | 핵심 목표 |
| --- | --- | --- |
| 실전 프로젝트 1 | 통계 수치 요약 서비스 | 통계 수치 분석 및 자연어 요약 (`summarize_stat`) |
| 실전 프로젝트 2 | 통계표 안내 챗봇 | 사용자 질의에 적합한 통계표 탐색 및 경로 안내 (`search_tables`) |
| 종합 프로젝트 | 맞춤형 통계 AI 챗봇 | 통계표 탐색부터 수치 분석·시각화(`make_chart`)까지 통합 |

본 프로젝트에서 개선하려는 부분은 다음과 같다.

- 사용자가 원하는 통계표를 찾더라도 실제 수치를 확인하고 증감률·구성비 등을 계산·분석하는 과정이 추가로 필요하다.
- 단순한 통계표 위치 안내만으로는 추이, 비교, 구성비 등 분석 결과를 즉시 확인하기 어렵다.
- 통계 데이터를 시각화하려면 별도의 차트 제작 과정이 필요하다.

## **1. Problem Definition**

### **1.1 문제점: 비즈니스 니즈**

**자연어 질문 하나로 통계표 탐색, 데이터 분석, 결과 요약 및 시각화까지 수행하는 근거 수치 고정형(Grounded) AI 서비스**를 구현한다.

국가통계포털(KOSIS)의 통계를 활용하려면 질문에 맞는 표를 찾고, 항목·시점·분류를 선택하고, 셀을 읽어 증감률이나 구성비를 계산한 뒤 차트를 만들어야 한다. 통계표 위치 안내에 더해, 실제 수치를 이해하고 활용하는 과정을 지원하는 서비스가 필요하다.

요구사항은 **자연어 질문으로 실제 통계표 수치에 근거한 답변·요약·차트를 출처와 함께 제공하는 것**이다. 모든 핵심 수치는 원본 셀 또는 원본 셀에 적용한 검증 가능한 계산식으로 추적되어야 한다. 자료가 없거나 질문을 해석할 수 없으면 숫자를 추정하지 않고, 불가 사유·가능한 범위 또는 확인 질문을 제시한다.

예시 시나리오:

**Turn 1 — 통계 수치 조회 및 분석**

사용자 질문:

> 2024년 서울특별시의 순이동 인구와 전입·전출 규모를 알려주세요.
> 

기대 답변:

> 2024년 국내인구이동통계를 확인한 결과, 서울특별시의 총전입은 [조회된 수치]명, 총전출은 [조회된 수치]명으로 순이동은 [계산된 수치]명(순유출/순유입)입니다.
> 
> 
> 출처: KOSIS / 국가데이터처 / 국내인구이동통계 / [통계표명 및 통계표 ID]
> 

**Turn 2 — 후속 시각화 요청**

사용자 질문:

> 이 통계표의 최근 5년간 추이를 차트로 만들어 주세요.
> 

기대 동작:

- 이전 대화에서 조회한 통계표와 조건(지역, 항목 등)을 대화 상태(Dialogue State)로 유지한다.
- 실제 데이터 구조(시계열 추이)에 적합한 차트 유형(꺾은선 그래프)을 선택한다.
- 차트와 주요 수치에 대한 간단한 설명을 제공한다.

대상은 제공된 인구 분야 14개 조사, 2,364개 통계표의 카탈로그다. 수치는 KOSIS OpenAPI 또는 제공된 표 파일을 통해 확보한다.

여기서 범위 밖인 ‘지역통계’는 제공 카탈로그 밖의 별도 지역통계 자료로의 확장을 뜻한다. **대상 인구 통계표 안에 존재하는 시도·시군구·읍면동 등 지역 분류는 해당 표가 실제 제공하는 수준과 시점 안에서 지원한다.** 예를 들어 시도별 인구동태와 인구밀도 비교는 연구 범위에 들어간다. 전국 단위 표에 없는 지역 값을 요구하거나 시도 표에 시군구 값을 요구하면 분류 불일치로 처리한다.

## 단계와 산출물

|                       | Phase 1                                                    | Phase 2                             | Final                                                                          |
| --------------------- | ---------------------------------------------------------- | ----------------------------------- | ------------------------------------------------------------------------------ |
| 기간                  | 10/20 ~ 10/28                                              | 10/28 ~ 11/4                        | 11/4 ~ 12/10                                                                   |
| 답하는 질문           | "이 표에서 숫자가 어때?"                                   | "어느 표를 봐야 해?"                | "찾아서, 계산해서, 그려줘"                                                     |
| 만드는 것             | 통계 수치 요약 모듈                                        | 통계표 안내 챗봇 (RAG)              | 툴 호출 에이전트 + 시각화                                                      |
| 핵심 기술             | 조회·계산·요약, 예외 처리                                  | 검색·순위·답 없음 판정              | 툴 선택, 멀티턴, 차트                                                          |
| 핵심 지표             | 수치 정확도, 불가 질문 거절률, 질의 이해(슬롯 필드 정확도) | Hit@k, MRR, 답 없음 F1              | 툴·파라미터 정확도, 멀티턴 성공률, 비용·지연                                   |
| 오케스트레이션        | 고정 파이프라인 (LangChain 미사용)                         | LangChain 도입 (검색기·리랭커 조합) | LangGraph StateGraph 위에서 단일 / Plan-and-Execute / 멀티 구성을 비교 후 결정 |
| 다음 단계로 넘기는 것 | 조회·계산·검증 모듈                                        | `search_tables()` 검색 모듈         | 완성 서비스                                                                    |

## 동작 예시

```
Q  인구동향조사에서 최근 5년 출생아 수가 어떻게 변했어?
A  2020년 [값]명에서 2024년 [값]명으로 [값]% 감소했습니다. 가장 큰 감소는 [연도]입니다.
   출처: KOSIS / 인구동향조사 / [통계표명]

Q  출생아가 줄어든 원인이 뭐야?
A  통계 수치만으로는 원인을 판단할 수 없습니다. 연도별 출생아 수와 증감률은 안내할 수 있습니다.
```

## 현재 상태 (2026-10-22)

한 방향으로 돈다. **데이터 특성 → 파이프라인 규칙 → 평가 기준 → 평가 결과 분석 → 다음 개선 대상.** 개선은 한 번에 하나만 바꾸고 같은 골든셋으로 "몇 → 몇"을 남긴다.

### 파이프라인 (LLM 담당과 코드 담당을 나눈다)

핵심 기능의 흐름을 먼저 고정하고, LangChain·멀티에이전트는 필요가 측정으로 확인된 시점에 넣는다.

```
질문 → [통계표 검색] → 메타데이터 조회 → 조회 조건 검증 → 데이터 조회 → Python 계산 → 요약 문장 → 숫자 검증 → 차트
        Phase 2          ─────────────────────── Phase 1 (현재 구현) ───────────────────────────────          Final
```

| 단계                                           | 담당                                                  | 구현 위치                                                      | 출력                                 |
| ---------------------------------------------- | ----------------------------------------------------- | -------------------------------------------------------------- | ------------------------------------ |
| 질의 분석 (유형·항목·기간·분류 슬롯)           | **LLM** (structured output)                           | `kosis_chat/judge/question.py`, `kosis_chat/llm/structured.py` | `StatQuery`                          |
| 답변 가능 판정 (8개 불가 사유, 되묻기, 기본값) | 규칙                                                  | `kosis_chat/judge/answerability.py`                            | 진행 / `Refusal` / 되묻기            |
| 조회·정규화 (단위, 결측 기호, 계·소계)         | 코드                                                  | `src/kosis_ai/normalize.py`, `kosis_chat/data/`                | tidy DataFrame                       |
| 계산 (값·증감률·비교·구성비·최대최소)          | pandas                                                | `kosis_chat/compute/`                                          | facts (값·단위·시점·분류·원본 셀·식) |
| 요약 문장                                      | **LLM** (`{f1}` 자리표시자만 씀, 숫자 직접 작성 금지) | `kosis_chat/llm/compose.py`, `kosis_chat/answer/template.py`   | 문장                                 |
| 숫자 검증 → 실패 시 1회 재생성 → 템플릿 대체   | 코드                                                  | `kosis_chat/answer/verify.py`                                  | 검증된 `StatAnswer`                  |
| 차트 (Vega-Lite)                               | 코드 (같은 facts 사용)                                | `kosis_chat/answer/chart.py`                                   | 차트 spec                            |

LLM이 하는 일은 **질문 해석과 문장 연결 두 가지뿐**이다. 숫자는 항상 facts에서 오고, 차트도 요약과 같은 facts를 그린다.

### 데이터 특성 → 파이프라인 규칙

EDA([docs/eda-full.md](docs/eda-full.md), 2,364표 전수)에서 발견한 특성을 규칙과 골든셋 함정 태그로 바꾼다.

| 데이터 특성                         | 파이프라인 반영                            | 골든셋 함정 태그                                |
| ----------------------------------- | ------------------------------------------ | ----------------------------------------------- |
| 단위 공란 551표                     | 항목명·메타데이터로 단위 복원 (306표 해결) | `unit_missing`                                  |
| 결측 기호 `-`, `X`, 행 자체 없음    | 기호 보존 + `missing_value` 거절           | `dash_symbol`, `x_symbol`, `structural_missing` |
| 계·소계와 말단 행 혼재              | 합산·순위 후보에서 합계성 행 제외          | `item_subtotal`, `same_level_mixed`             |
| 인구총조사 5년 주기                 | 비어 있는 해를 결측이 아니라 `no_period`로 | `census_5y_gap`                                 |
| 음수 값(순이동 등)                  | 증감률 분모 부호 처리                      | `negative_value`                                |
| 추계치·실측치 혼재, 월·분기·연 주기 | 수치 성격·주기를 조건 검증에 포함          | (Phase 2에서 추가)                              |

### 평가 체계

| 평가 영역        | 평가 대상                         | 방식                        | 현재 상태                                                   |
| ---------------- | --------------------------------- | --------------------------- | ----------------------------------------------------------- |
| 질의 이해        | 유형·항목·기간·분류를 맞게 뽑았나 | 정답 JSON과 **필드별** 비교 | ❌ 없음 → 골든 `required_facts`에서 슬롯 정답을 파생해 추가 |
| 데이터 정확성    | 올바른 수치를 조회·계산했나       | 원본 Parquet과 자동 대조    | ✅ `eval/grade.py`                                          |
| 답변 신뢰성      | 수치·단위·출처·해석이 일치하나    | 규칙 검증 + 사람 판정       | △ 규칙은 있으나 라벨 숫자 오탐                              |
| 답변 불가 판단   | 거절 사유가 타당한가              | 사유별 정답 라벨 비교       | ✅ `allowed_reasons` 비교 (`wrong_reason`)                  |
| 자연어 품질      | 이해하기 쉬운 설명인가            | 사람 평가 0~2점             | ❌ 없음 → 리뷰 양식에 한 칸 추가                            |
| (운영) 비용·지연 | LLM·API 호출 수, 토큰, 응답 시간  | 실행 manifest·이벤트 로그   | △ 토큰·지연·HTTP 시도 수만 있음, **단계별 기록 없음**       |

골든셋: 62문항(기본 40 + 엣지 22, 10개 표). 답변 가능 42 : 불가 17 : 되묻기 3 ≈ **2.5 : 1** (권장 기준 60건 이상·3:1에 근접). 정답은 Parquet 원값과 대조하고 오류 주입으로 채점기가 잡는지 확인한다(oracle 62/62, 주입 오류 검출 62/62). 의도대로 되는 질문만이 아니라 **한계라고 정한 부분도 골든셋 문항으로 넣어** 설계가 맞다는 근거로 쓴다.

### 가설과 실험

| ID  | 가설                                                                                   | 실험                                                   | 단계         |
| --- | -------------------------------------------------------------------------------------- | ------------------------------------------------------ | ------------ |
| H1  | 계산·렌더링·검증을 코드로 분리하면 근거 없는 수치가 줄어든다                           | B0 vs C, 같은 62문항                                   | Phase 1      |
| H2  | 규칙 기반 판정은 불필요한 거절을 늘리지 않고 불가 질문을 잡는다                        | 거절률 + 잘못된 거절 동시 측정                         | Phase 1      |
| H3  | 기간·주기 검증을 표 선정 전에 하면 "월별 질문에 연간 표" 오류가 준다                   | 검색 골든셋의 주기 조건 문항                           | Phase 2      |
| H4  | 하이브리드 검색 + 리랭킹이 개념어·유사 표 질의에서 BM25보다 낫다                       | BM25 → 임베딩 → 하이브리드 → 리랭커, 질의 유형별 Hit@k | Phase 2      |
| H5  | 슬롯 추출은 모델에 따라 차이가 크고, 문장 생성은 차이가 작다                           | 같은 슬롯 골든으로 HCX vs 오픈웨이트 후보              | Phase 2 초반 |
| H6  | 에이전트 구조(단일 / Plan-and-Execute / 멀티)의 이득은 비용·지연 증가보다 작을 수 있다 | 같은 툴·평가셋으로 3종 비교                            | Final        |

## 기술·모델 선택 원칙

**지금 정하는 것과 나중에 정하는 것을 나눈다.** 에이전트·모델 구조는 Phase 1·2 결과(그리고 데이터 특성)를 본 뒤 구체화한다. 근거 없이 구조부터 고르지 않는다.

| 항목           | 지금 결정                                                | 결정 시점·근거                                                                                                                                                                                  |
| -------------- | -------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 수치 처리      | pandas 결정적 계산 + 검증기 (확정)                       | —                                                                                                                                                                                               |
| 오케스트레이션 | Phase 1은 고정 파이프라인 (LangChain 미사용)             | Phase 2 RAG부터 **LangChain 도입** (문서 로딩·검색기·리랭커 조합)                                                                                                                               |
| 에이전트 구조  | LangGraph StateGraph (확정)                              | Final T3-07~10에서 단일 / Plan-and-Execute / 멀티 분할을 같은 평가셋으로 비교 후 ADR                                                                                                            |
| Tracing        | 자체 이벤트 로그(`eval/recording.py`)를 span 단위로 정리 | **OpenTelemetry + Jaeger로 확정** (Langfuse·LangSmith 미사용). Phase 2에서 검색 단계 계측 포함, Final에서 LangGraph 노드 단위 + 내부 API 호출 span. LLM 토큰·비용은 OpenTelemetry에서 별도 계측 |
| 생성 모델      | HCX-DASH-002 (공식 평가 대상)                            | 유지                                                                                                                                                                                            |
| 분류·추출 모델 | HCX                                                      | Phase 2 초반, **Gemma 4 31B** 등 오픈웨이트 후보를 슬롯 추출 필드 정확도·JSON 1회 통과율·지연으로 비교. 로컬 실행이면 데이터 반출 문제가 없으나 실행 환경(GPU)과 사용 가능 여부를 먼저 확인     |
| 문서 파싱 도구 | 미정                                                     | 통계표 설명·주석 문서를 RAG에 쓸 때 PDF → 목차 트리 추출 라이브러리 검토 (멘토 사례: Gemma 계열이 목차 트리 추출을 잘함)                                                                        |

규칙: 어떤 도구·모델이든 **골든셋 수치로 전→후를 보인 뒤** 채택한다. 도구 도입 자체를 성과로 세지 않는다.

## 기술 스택 및 시스템 아키텍처

팀이 확정한 RAG AI 챗봇 기술 스택(Next.js + FastAPI + LangGraph + ChromaDB + OpenTelemetry)을 이 프로젝트의 기준 스택으로 삼는다. 이 프로젝트는 PDF 문서 RAG가 아니라 **통계표 메타데이터 RAG + 수치 계산 파이프라인**이므로, 각 항목이 어느 단계에서 쓰이는지와 이 프로젝트에서 달라지는 점을 함께 적는다.

### 기술 스택

| 영역                | 기술 스택                                                  | 주요 역할                        | 적용 단계 · 이 프로젝트에서의 비고                                                                                    |
| ------------------- | ---------------------------------------------------------- | -------------------------------- | --------------------------------------------------------------------------------------------------------------------- |
| Frontend            | Next.js (App Router), TypeScript                           | 웹 클라이언트·라우팅             | Final. 차트는 백엔드가 준 Vega-Lite spec을 렌더링                                                                     |
| UI                  | React + Tailwind CSS + shadcn/ui                           | 채팅 인터페이스·대시보드         | Final                                                                                                                 |
| Chat Streaming      | Vercel AI SDK                                              | 실시간 응답 스트리밍             | Final                                                                                                                 |
| Backend             | FastAPI (Python)                                           | REST API·스트리밍 응답           | Final. Phase 1·2 모듈(`summarize_stat`, `search_tables`)을 그대로 import                                              |
| API Validation      | Pydantic                                                   | 요청·응답 검증                   | **Phase 1부터 사용 중** (`src/kosis_ai/schemas.py`, 단계 간 공용 계약)                                                |
| AI Workflow         | LangGraph (StateGraph)                                     | 파이프라인 노드 구성·상태 관리   | Final. Phase 1 파이프라인 단계를 노드로 옮김 (아래 매핑)                                                              |
| LLM Integration     | LangChain                                                  | LLM 호출, Prompt, Retriever 연동 | Phase 2부터                                                                                                           |
| LLM                 | HCX-DASH-002                                               | 근거 기반 답변 생성              | 공식 평가 대상. LangChain 모델 인터페이스 뒤에 두어 교체 가능하게 하고, 슬롯 추출용 Gemma 4 31B는 Phase 2 초반에 비교 |
| Embedding           | 임베딩 모델 (T2-05에서 후보 비교 후 선정)                  | 문서·질문 벡터 변환              | Phase 2. 같은 검색 골든셋으로 비교                                                                                    |
| Vector Database     | ChromaDB                                                   | 임베딩 저장·유사도 검색          | Phase 2. 단위는 통계표 1건 = 문서 1건 (표 ID 보존)                                                                    |
| RAG Pipeline        | LangChain Text Splitters                                   | Chunking·Overlap                 | Phase 2. 표 설명이 길 때만 분할, 조각은 원래 표 ID에 연결                                                             |
| Reranking           | Cross-Encoder (선택)                                       | 검색 결과 재평가                 | Phase 2 T2-07. 비용·지연과 함께 측정                                                                                  |
| Document Processing | PyMuPDF                                                    | PDF 텍스트 추출·전처리           | 조건부. 통계표 설명·주석을 PDF로 받을 때만 (현재 수치 데이터는 Parquet)                                               |
| Observability       | OpenTelemetry                                              | Trace·Span 계측                  | Phase 1 이벤트 로그를 span 단위로 정리해 두고 Final에서 계측                                                          |
| Workflow Tracing    | LangGraph Events + OpenTelemetry                           | 노드 실행 과정·지연·오류 추적    | Final                                                                                                                 |
| Trace Storage / UI  | Jaeger                                                     | 분산 Trace 저장·시각화           | Final. LLM 토큰·비용은 Jaeger가 아니라 OpenTelemetry에서 별도 계측                                                    |
| Evaluation          | Ragas                                                      | 검색·답변 품질 평가              | Phase 2·Final. LangGraph 실시간 요청과 분리해 비동기 평가 작업으로 실행                                               |
| Evaluation Dataset  | Golden Set (Q&A + Context)                                 | 평가용 정답·근거                 | Phase 1 `stat_golden.jsonl` 62문항 사용 중, Phase 2 검색 골든셋 100건+                                                |
| Evaluation Metrics  | Context Precision·Recall, Faithfulness, Answer Correctness | RAG 성능 측정                    | Phase 2·Final 안내 답변. **수치 정확도는 Ragas가 아니라 기존 코드 기반 채점기(원본 Parquet 대조)가 기준**             |
| Database            | PostgreSQL                                                 | 사용자·대화 기록·평가 결과       | Final (대화 상태·평가 결과). Phase 1·2는 파일(jsonl) 기록                                                             |
| Document Storage    | S3 호환 Object Storage                                     | 업로드 문서 원본                 | 해당 없음 (문서 업로드 기능 없음, 데이터는 제공 Parquet). 업로드를 추가할 때만 도입                                   |
| Testing             | Pytest, Vitest + React Testing Library                     | 파이프라인·API·UI 테스트         | Pytest는 Phase 1부터, Vitest는 Final                                                                                  |
| Deployment          | Docker / Docker Compose                                    | 컨테이너화·로컬 통합 환경        | Final. 키·제공 데이터는 이미지 밖에서 주입                                                                            |
| CI/CD               | GitHub Actions                                             | 테스트·배포 자동화               | Final. 오프라인 검증(`scripts/verify.sh`)부터 연결                                                                    |

### 시스템 아키텍처 (Final 목표)

```
┌─────────────────────────────────────────────────────────┐
│ FRONTEND   Next.js + React + TypeScript                 │
│            Tailwind CSS + shadcn/ui + Vercel AI SDK     │
│            Chat UI (답변 · 출처 · Vega-Lite 차트)        │
└───────────────────────┬─────────────────────────────────┘
                        │ HTTP / Streaming
                        ▼
┌─────────────────────────────────────────────────────────┐
│ BACKEND    FastAPI + Pydantic                           │
│                                                         │
│  ┌── LangGraph (StateGraph, 대화 상태 유지) ──────────┐  │
│  │ Query Processing (질의 분석 · 슬롯)                │  │
│  │      ↓                                             │  │
│  │ Retrieval ───────────────► ChromaDB (표 메타데이터)│  │
│  │      ↓                                             │  │
│  │ Reranking → 표·조건 선정 (search_tables)           │  │
│  │      ↓                                             │  │
│  │ Answerability (규칙 판정 · 거절 · 되묻기)          │  │
│  │      ↓                                             │  │
│  │ Fetch & Normalize ───────► Parquet / KOSIS API     │  │
│  │      ↓                                             │  │
│  │ Compute (pandas, 코드가 숫자 계산) → facts         │  │
│  │      ↓                                             │  │
│  │ Compose (LLM, 자리표시자 문장) ──► LLM API         │  │
│  │      ↓                                             │  │
│  │ Verify (숫자 검증 → 재생성 → 템플릿 대체)          │  │
│  │      ↓                                             │  │
│  │ Chart (make_chart, 같은 facts)                     │  │
│  └────────────────────────────────────────────────────┘  │
│        │                              │                  │
│   OpenTelemetry                    Ragas (비동기 평가)    │
└────────┼──────────────────────────────┼─────────────────┘
         ▼                              ▼
   ┌───────────┐                 ┌─────────────┐
   │ Jaeger    │                 │ PostgreSQL  │
   │ Trace/Span│                 │ Chat History│
   │ Latency   │                 │ Eval Results│
   │ Errors    │                 │ Golden Set  │
   └───────────┘                 └─────────────┘
```

Phase 1의 단계 함수가 그대로 LangGraph 노드가 된다. 같은 함수를 Phase 2·Final이 import하므로, 노드 단위 입출력은 지금의 Pydantic 계약을 유지한다.

| Phase 1 구현 (`kosis_chat/`)           | Final LangGraph 노드  | 비고                             |
| -------------------------------------- | --------------------- | -------------------------------- |
| `judge/question.py`                    | Query Processing      | LLM structured output            |
| (Phase 2) 검색기                       | Retrieval · Reranking | `search_tables`                  |
| `judge/answerability.py`               | Answerability         | 규칙, LLM 호출 없음              |
| `data/`, `src/kosis_ai/normalize.py`   | Fetch & Normalize     | 코드                             |
| `compute/`                             | Compute               | pandas, 숫자의 유일한 출처       |
| `llm/compose.py`, `answer/template.py` | Compose               | LLM은 문장만                     |
| `answer/verify.py`                     | Verify                | 실패 시 재생성 1회 → 템플릿 대체 |
| `answer/chart.py`                      | Chart                 | `make_chart`                     |

### 설계 원칙

| 설계 영역          | 확정 방향                                                                                                                   |
| ------------------ | --------------------------------------------------------------------------------------------------------------------------- |
| Frontend / Backend | Next.js와 FastAPI를 독립적으로 구성                                                                                         |
| 워크플로우         | LangGraph StateGraph 기반 노드 설계. 에이전트 분할 방식(단일 / Plan-and-Execute / 멀티)은 같은 평가셋으로 비교 후 결정 (H6) |
| Vector Search      | ChromaDB                                                                                                                    |
| Observability      | Langfuse·LangSmith 대신 OpenTelemetry + Jaeger                                                                              |
| Tracing 단위       | LangGraph 노드 단위 + 내부 API 호출(KOSIS, LLM) span                                                                        |
| Evaluation         | Golden Set 기반 평가. 수치는 코드 채점, 검색·답변 품질은 Ragas, 사람 판정은 자연어 품질 0~2점                               |
| 실험·최적화        | Chunk Size, Overlap, Top-K, Reranking 성능 비교 (Phase 2)                                                                   |
| 데이터 관리        | Vector는 ChromaDB, 관계형 데이터는 PostgreSQL                                                                               |
| 서비스 연동        | FastAPI → Next.js 스트리밍 응답                                                                                             |
| 배포               | Docker 기반 컨테이너 배포                                                                                                   |

이 스택은 "무엇을 만들까"의 기준이고 "언제 넣을까"는 위 표의 적용 단계를 따른다. 각 도구는 위 선택 원칙대로 **골든셋 수치로 전→후를 보인 뒤** 채택한다.

## 현재 상태 (2026-10-25)

- **Phase 1 · 계산 파이프라인 C**: 규칙 판정 → pandas 계산 → LLM 문장(자리표시자) → 숫자 검증 → 템플릿 대체 → Vega-Lite 차트까지 동작한다(`kosis_chat/pipeline.py`). B0 베이스라인(표 직렬화 + LLM 직접 답변)도 같은 하네스에서 돈다.
- 튜닝 모델은 **HCX-DASH-002** (HCX-007은 10/22부터 보류). 데이터는 `data/values/*.parquet`(2,364표) 하나로 통일했고, 큰 표는 질문마다 범위를 잘라 넣는다(`--scope oracle|auto|full`, [eval/README.md](eval/README.md) 2장).
- 골든셋 62문항(기본 40 + 엣지 22), 자동 채점(`eval/grade.py`)과 결과 뷰어(`view.html`)가 있다.

C 파이프라인 · 기본 40문항 자동 채점 결과 (사람 판정 전):

| 지표                                       | 값                       |
| ------------------------------------------ | ------------------------ |
| 수치 정확도 (answerable + default_allowed) | 27/28 (96%)              |
| 답변 불가 거절 (허용 사유)                 | 10/10, 잘못된 거절 0     |
| answerable 문항 PASS                       | 14/25                    |
| 모델 JSON 1회 통과                         | 12/29 (17건 형식 재시도) |
| 실행 실패                                  | 0                        |

분석과 다음 개선 방향 (세부 일정은 [액션 아이템](docs/action-items-20261025.md)):

- **계산은 거의 맞는데 문항이 떨어진다.** answerable FAIL 11건 중 **10건이 '1인 가구', '65세' 같은 라벨 속 숫자를 검증기가 근거 없는 수치로 오판**한 것이다. 진짜 오류는 `wrong_dims` 1, `assumed_condition` 1, 되묻기 실패 1건뿐이다. → 다음 개선 1순위는 모델이 아니라 **검증기의 라벨 숫자 처리**다.
- **HCX의 구조화 출력이 불안정하다** (1회 통과 12/29). 형식 재시도는 지연·비용을 늘린다. → 슬롯 추출 모델 비교의 근거가 된다.
- **실패가 어느 단계에서 났는지를 사람이 추정한다.** 이벤트 로그가 HTTP·LLM 호출 단위라서 판정·계산·검증 단계 기록이 없다. → 단계별 로그가 먼저다.
- B0와 같은 62문항 비교가 아직 없다. → Phase 1 제출 전에 "B0 → C" 한 줄을 확정한다.
- 최신 상태와 막힌 항목은 [docs/current-state.md](docs/current-state.md)를 본다 (새 세션·새 팀원의 시작점).

## 빠르게 실행해 보기

```bash
pip install -r requirements.txt            # Python 3.12 권장
cp .env.example .env                       # 키는 .env에만 둔다 (커밋 금지)

bash scripts/verify.sh                     # 오프라인 검증: pytest(Mock) + 실데이터 커밋 차단 확인

# API 키 없이 흐름만 확인 (Mock)
python -m kosis_chat.cli --table data/values/101_DT_1B8000H.parquet --backend mock -q "2023년 서울 출생건수는?"

# 실제 모델로 질문 (.env에 HCX_API_KEY 필요, 기본 백엔드: HCX-007 thinking 끔)
python -m kosis_chat.cli --table <표 파일> -q "질문"
# 모델·thinking 바꾸기 (thinking을 켜면 max-tokens를 늘린다)
python -m kosis_chat.cli --table <표 파일> --thinking low --max-tokens 5120 -q "질문"
python -m kosis_chat.cli --table <표 파일> --model HCX-DASH-002 -q "질문"
```

| 환경변수 | 용도 |
| --- | --- |
| `KOSIS_API_KEY` | KOSIS OpenAPI |
| `HCX_API_KEY` | 기본·평가 대상 모델 HCX (CLOVA Studio v3, `nv-`로 시작하는 키) |
| `GEMINI_API_KEY` | HCX 이전 개발용 모델 (Gemini, `--backend gemini`) |
| `PRETEST_API_KEY` | OpenAI 호환 API로 다른 모델 사전 테스트 |
| `LINEAR_API_KEY`, `LINEAR_TEAM_KEY` | TASK.md → Linear 자동 동기화 (선택) |

백엔드·모델 설정은 [configs/model.yaml](configs/model.yaml). Mock 실행 결과는 사전 테스트라 `eval-history`에 올리지 않는다.
`--form`을 쓴 실행은 결과 폴더에 그때의 `form.txt` 사본이, `manifest.json`에 해시가 남는다.

## 평가

모든 단계가 같은 방식으로 잰다. 한 번에 하나만 바꾸고, 같은 골든셋·데이터·모델로 다시 잰다.

```bash
python -m eval.run_eval --stage B0 --golden eval/golden/stat_golden.jsonl   # 골든셋 질문 실행, 답변 저장
python -m eval.review export|apply --run eval/results/<id>           # 사람이 PASS/PARTIAL/FAIL 판정
python -m eval.scorer --run eval/results/<id> --append-history       # 집계 → docs/reports/eval-history.md
```

골든셋(`eval/golden/`)과 실행 결과(`eval/results/`)에는 실제 수치가 들어 있어 커밋하지 않는다.

## 저장소 구조

```
kosis_chat/        서비스 코드: data/(입력·전처리·직렬화), llm/(모델 클라이언트·프롬프트), chatbot.py, cli.py
eval/              평가: run_eval.py, review.py, scorer.py (golden/, results/ 는 gitignore)
configs/           model.yaml (백엔드·모델 설정)
tests/             오프라인 테스트 (가짜 Parquet 표는 conftest.py가 만든다)
docs/
  current-state.md   현재 상태 인계 문서
  plans/             단계별 PLAN·TASKS (B0 베이스라인, 실전 1 설계·전처리 규칙)
  reports/           완료 리포트, eval-history.md
  notes/             일정·작업 방식·서비스 구성도 (project-workflow.md)
TASK.md            마이크로 태스크 로드맵 (47개, 마감·우선순위·의존성 표 포함)
CLAUDE.md          Claude Code 작업 지침
.claude/           agents, skills, hooks (팀 공용 개발 하네스)
```

## 작업 방식

1. **할 일은 [TASK.md](TASK.md)에서 고른다.** 상태(`[ ]` `[~]` `[x]`)를 고치면 hook이 Linear 이슈(팀 KOS)에 반영한다 (TASK.md → Linear 단방향).
2. **바로 만들지 않는다.** `/plan-gate`로 수행 아이디어와 3가지 옵션(예상 성능 향상·실질 이점·비용)을 `plans/<태스크ID>.md`에 쓰고,
   검증한 뒤 팀이 옵션을 고르면 구현한다. 승인된 plan이 없으면 hook이 코드 수정을 막는다.
3. 구현 후 평가 하네스로 **전 → 후 수치**를 plan의 Result와 TASK.md에 남긴다.
4. 큰 설계 변경은 `/plan` → `/tasks` → `/execute` → `/wrapup`(docs/plans, docs/reports)으로 문서화한다.
   세부 규칙은 [CLAUDE.md](CLAUDE.md), [AGENTS.md](AGENTS.md), [docs/notes/project-workflow.md](docs/notes/project-workflow.md).

## 데이터·보안 규칙

- API 키는 `.env`에만 둔다 (gitignore됨). 절대 커밋하지 않는다.
- 제공 데이터(`data/raw/`, `data/samples/`), 골든셋, 평가 결과는 저장소에 올리지 않는다.
- 답에 쓰는 숫자는 원본 셀이나 검증 가능한 계산식으로 추적되는 값만 허용하고, 핵심 수치마다 출처 표를 밝힌다.
- 수치로 뒷받침되지 않는 인과·정치적 주장은 하지 않는다.
