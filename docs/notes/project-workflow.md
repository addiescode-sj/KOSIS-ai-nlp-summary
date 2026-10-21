# KOSIS 챗봇 진행 지도

2026-10-21 기준. 실선 = 확정, 점선(`tbd`) = 추후결정.
같은 내용의 웹 페이지: https://claude.ai/artifact/LfUNCe2PzTd7aWdKYzmg64

## 1. 전체 일정

실전 1에서 만든 챗봇을 실전 2와 종합에서 키워 간다. 1차는 B0(베이스라인) 점수를 먼저 내고, 한 번에 하나씩 바꾸며 B1~B4로 올린다.

### 실전 1 · 통계 수치 요약 (~10/28)

표 데이터 + 자연어 질문 → 자연어 답변

| 날짜 | 할 일 | 상태 |
| --- | --- | --- |
| 10/20 | 설계, B0 코드(오프라인) | 완료 |
| 10/21 | HCX 호출 방식, B0용 표 1개 확인 | 오늘 |
| 10/22 | 골든셋 v1 → B0 실행·채점 | |
| 10/23 | B1 전처리 · B3 계산 · B4 검증 모듈 (LLM 없이 테스트) | |
| 10/24 | **E2E 게이트**: 실제 표 1개 전체 흐름 | |
| 10/25–26 | 표 4~9개로 확대, 전체 평가, 개선 1회 | |
| 10/27 | 게이트 통과 시 직렬화 3종 비교 | |
| 10/28 | 문서 정리, 제출 | |

### 실전 2 · 통계표 안내 (~11/04)

RAG로 메타데이터를 검색해 맞는 통계표를 찾아 준다.

- 확정: RAG 방식, 1차 챗봇 재사용
- 추후결정: 메타데이터 형식, 임베딩 모델, 벡터 저장소, 세부 일정

### 종합 · 통합 챗봇 (~12/07)

표 탐색 + 수치 분석 + 차트 + 연속 대화

- 확정: 1차·2차 기능을 합친 하나의 서비스
- 추후결정: 차트 도구, 화면(UI), 대화 기록 방식, 배포

## 2. 작업 방식 (하네스 + 평가 루프)

모든 단계는 같은 순서로 진행한다. 계획 승인 후 코딩, 오프라인 검증(verify.sh)과 HCX 평가는 따로 확인한다. 평가 결과는 같은 골든셋·데이터·모델로 eval-history.md에 한 줄씩 쌓는다.

```mermaid
flowchart LR
  subgraph H["개발 하네스 (단계마다 반복)"]
    direction LR
    P["/plan<br/>PLAN.md"] --> A{"사용자 승인"}
    A -- 수정 요청 --> P
    A -- 승인 --> T["/tasks<br/>TASKS.md"]
    T --> E["/execute<br/>코드 작성"]
    E --> V["scripts/verify.sh<br/>pytest · 실데이터 커밋 차단"]
    V -- 실패 --> E
    V -- 통과 --> W["/wrapup<br/>REPORT.md · current-state.md"]
  end
  subgraph L["평가 루프 (HCX, 수동 실행)"]
    direction LR
    R1["eval.run_eval<br/>골든셋 질문 실행"] --> R2["eval.review<br/>사람이 판정 (CSV)"]
    R2 --> R3["eval.scorer<br/>PASS / PARTIAL / FAIL 집계"]
    R3 --> R4["eval-history.md<br/>B0 → B1 → … → B4"]
  end
  W --> R1
  R4 -. "다음 단계: 한 번에 하나만 바꿈" .-> P
```

- 비교 규약: 데이터 해시, 골든셋 버전, 모델, temperature 0, runs 1 고정. 실행마다 manifest.json.
- 사전 테스트: `openai_compat` 백엔드로 다른 모델 연결 가능. 결과는 `pretest`로 표시되고 점수표에 들어가지 않는다.
- 규칙: API 키는 .env에만. 실데이터·골든셋·평가 결과는 커밋 금지. 표에서 추적할 수 없는 수치, 인과·정치적 주장 금지.

## 3. 실전 1 답변 흐름 (B0 → B4)

B0는 표를 그대로 프롬프트에 넣고 HCX가 답한다. B1~B4는 수치를 코드가 계산하고 LLM은 설명만 맡도록 단계적으로 바꾼다. 박스 안 이름은 실제 파일(B1~B4는 예정 파일).

```mermaid
flowchart TB
  IN["입력<br/>통계표 (CSV 또는 KOSIS JSON) + meta.json + 질문"]
  IN --> LD["불러오기 · 스키마 검사<br/>data/loader.py · data/schema.py"]
  LD --> PRE["전처리: 결측 · 단위 · 계/소계 표시, 원본 보존<br/>data/preprocessing.py (B1에서 R1~R9로 확장)"]
  PRE --> B0{"단계"}
  B0 -- "B0 (지금)" --> SER["표 직렬화 (Markdown)<br/>data/serializer.py"]
  SER --> GEN0["HCX 답변 생성<br/>llm/prompts.py · llm/model.py"]
  GEN0 --> OUT
  B0 -- "B1~B4 (10/23~)" --> NLU["B2 질문 분석: 유형 · 조건 · 모호성<br/>nlu/analyzer.py"]
  NLU --> AMB{"모호한가?"}
  AMB -- 예 --> ASK["기본값 적용 또는 되묻기"]
  AMB -- 아니오 --> PLAN["B3 계산 계획 · 근거 행 선택 · 계산<br/>analysis/planner.py · analysis/calculator.py"]
  PLAN --> CAN{"계산 가능한가?<br/>합산 가능성 · 분모 · 결측"}
  CAN -- 아니오 --> NA["답변 불가 + 이유 + 가능한 범위"]
  CAN -- 예 --> REN["B4 수치 문장 템플릿 + LLM 설명<br/>llm/render.py"]
  REN --> VAL{"결합 검증 V1~V5<br/>analysis/validator.py"}
  VAL -- 실패 --> REN
  VAL -- 통과 --> OUT["답변 + 출처(통계표명 · 작성기관 · 시점)"]
  NA --> OUT
  ASK --> OUT
```

## 4. 서비스 구성도 (라이브러리 포함)

실선 박스는 지금 코드에 들어간 라이브러리. 점선 박스는 추후결정이며, 적힌 이름은 검토 후보일 뿐이다.

```mermaid
flowchart LR
  subgraph SRC["데이터"]
    K["KOSIS OpenAPI<br/>(JSON 응답)"]
    F["제공 파일<br/>CSV + meta.json"]
    MD["통계표 메타데이터<br/>(형식 추후결정)"]:::tbd
  end
  subgraph APP["kosis_chat (Python 3.12)"]
    direction TB
    CFG["설정<br/>PyYAML · python-dotenv<br/>configs/model.yaml · .env"]
    DATA["데이터 계층<br/>csv · json · hashlib · dataclasses"]
    CORE["답변 엔진<br/>chatbot.py"]
    LLM["LLM 계층 (llm/model.py)<br/>HCXClient · OpenAICompatClient (urllib)<br/>MockClient"]
    RAG["2차 RAG 검색<br/>임베딩 · 벡터 저장소 추후결정<br/>(후보: FAISS, Chroma)"]:::tbd
    VIZ["종합 차트<br/>추후결정<br/>(후보: matplotlib, Plotly)"]:::tbd
    MEM["종합 연속 대화 기록<br/>추후결정"]:::tbd
  end
  subgraph MODEL["모델"]
    HCX["HyperCLOVA X (HCX)<br/>평가 대상 · 호출 방식 10/21 확인"]
    PT["사전 테스트 모델<br/>OpenAI · Gemini · Ollama 등"]
  end
  subgraph UI["사용 화면"]
    CLI["CLI<br/>argparse · kosis_chat/cli.py"]
    NB["Colab 노트북<br/>notebooks/colab_run.ipynb"]
    WEB["웹 화면<br/>추후결정<br/>(후보: Streamlit, Gradio)"]:::tbd
  end
  subgraph QA["검증 · 평가"]
    PY["pytest<br/>scripts/verify.sh"]
    EV["eval/ run_eval · review · scorer<br/>eval-history.md"]
  end
  K --> DATA
  F --> DATA
  MD -.-> RAG
  CFG --> CORE
  DATA --> CORE
  RAG -.-> CORE
  CORE --> LLM
  LLM --> HCX
  LLM --> PT
  CORE -.-> VIZ
  CORE -.-> MEM
  CLI --> CORE
  NB --> CORE
  WEB -.-> CORE
  PY --> APP
  EV --> CORE
  classDef tbd stroke-dasharray: 6 4
```

| 영역 | 지금 쓰는 것 | 추후결정 |
| --- | --- | --- |
| 언어 · 실행 환경 | Python 3.12, Google Colab(VS Code 연결), Google Drive 레포 | 배포 환경 |
| 설정 · 비밀값 | PyYAML, python-dotenv | HCX 환경변수 이름 (10/21) |
| 데이터 처리 | 표준 라이브러리 csv, json, hashlib, dataclasses | pandas 도입 여부 |
| LLM 호출 | urllib (OpenAI 호환), HCX 호출부는 비어 있음 | HCX SDK 또는 REST, 모델 버전, JSON 출력 지원 |
| 테스트 · 평가 | pytest, 자체 eval/ 도구 | 합격 기준 수치 팀 합의 |
| 검색 (2차) | 없음 | 임베딩 모델, 벡터 저장소, 메타데이터 형식 |
| 차트 · 화면 (종합) | CLI, 노트북 | 차트 라이브러리, 웹 프레임워크, 대화 기록 |

## 5. 지금 기다리는 것

- 10/21: HCX 호출 방식과 모델 이름 / B0용 실제 표 1개 (분류 2단 이하, 연 단위, 계 포함) / HCX 튜닝 기능 포함 여부
- 10/22: 골든셋 v1 (약 17문항) → B0 실행 → 사람 판정 → 첫 점수
- 1차 열린 질문: 데이터를 API로 받는지 파일로 받는지 / 제출 형태 (CLI·노트북·웹) / 팀 내 담당 모듈 범위
