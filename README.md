# **연구 계획서 · 국가통계 데이터 기반 AI 챗봇 및 맞춤형 통계 시각화 서비스**

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

| | 실전 1 | 실전 2 | 종합 |
| --- | --- | --- | --- |
| 기간 | 10/20 ~ 10/28 | 10/28 ~ 11/4 | 11/4 ~ 12/10 |
| 답하는 질문 | "이 표에서 숫자가 어때?" | "어느 표를 봐야 해?" | "찾아서, 계산해서, 그려줘" |
| 만드는 것 | 통계 수치 요약 모듈 | 통계표 안내 챗봇 (RAG) | 툴 호출 에이전트 + 시각화 |
| 핵심 기술 | 조회·계산·요약, 예외 처리 | 검색·순위·답 없음 판정 | 툴 선택, 멀티턴, 차트 |
| 핵심 지표 | 수치 정확도, 불가 질문 거절률 | Hit@k, MRR, 답 없음 F1 | 툴·파라미터 정확도, 멀티턴 성공률, 비용·지연 |
| 다음 단계로 넘기는 것 | 조회·계산·검증 모듈 | `search_tables()` 검색 모듈 | 완성 서비스 |

## 동작 예시

```
Q  인구동향조사에서 최근 5년 출생아 수가 어떻게 변했어?
A  2020년 [값]명에서 2024년 [값]명으로 [값]% 감소했습니다. 가장 큰 감소는 [연도]입니다.
   출처: KOSIS / 인구동향조사 / [통계표명]

Q  출생아가 줄어든 원인이 뭐야?
A  통계 수치만으로는 원인을 판단할 수 없습니다. 연도별 출생아 수와 증감률은 안내할 수 있습니다.
```

## 현재 상태 (2026-10-22)

- **실전 1 · B0 베이스라인**: 표 데이터 + 질문 → HCX → 답변이 끝까지 도는 가장 단순한 챗봇(`kosis_chat/`)과
  평가 도구(`eval/`)가 동작한다. HCX(CLOVA Studio) 실제 호출을 확인했다.
- 튜닝 모델은 **HCX-DASH-002** (HCX-007은 10/22부터 보류). 답변 형식 지시는 `--form <파일>`로 질문마다 붙여 실험한다.
- 알려진 한계: B0는 표 전체를 프롬프트에 넣으므로 큰 표(예: DT_1JC1502, 1,814행)는 DASH 입력 한도를 넘어 `input_too_long`이 된다.
- B0 이후 한 번에 하나씩 바꾸며 B1(전처리) → B3(코드 계산) → B4(숫자 검증)로 올리고, 같은 골든셋으로 `eval-history.md`에 한 줄씩 쌓는다.
- 최신 상태와 막힌 항목은 [docs/current-state.md](docs/current-state.md)를 본다 (새 세션·새 팀원의 시작점).

## 빠르게 실행해 보기

```bash
pip install -r requirements.txt            # Python 3.12 권장
cp .env.example .env                       # 키는 .env에만 둔다 (커밋 금지)

bash scripts/verify.sh                     # 오프라인 검증: pytest(Mock) + 실데이터 커밋 차단 확인

# API 키 없이 흐름만 확인 (Mock)
python -m kosis_chat.cli --table tests/fixtures/pop_a.csv --backend mock -q "2023년 여자 인구는?"

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
python -m eval.run_eval --stage B0                                   # 골든셋 질문 실행, 답변 저장
python -m eval.review export|apply --run eval/results/<id>           # 사람이 PASS/PARTIAL/FAIL 판정
python -m eval.scorer --run eval/results/<id> --append-history       # 집계 → docs/reports/eval-history.md
```

골든셋(`eval/golden/`)과 실행 결과(`eval/results/`)에는 실제 수치가 들어 있어 커밋하지 않는다.

## 저장소 구조

```
kosis_chat/        서비스 코드: data/(입력·전처리·직렬화), llm/(모델 클라이언트·프롬프트), chatbot.py, cli.py
eval/              평가: run_eval.py, review.py, scorer.py (golden/, results/ 는 gitignore)
configs/           model.yaml (백엔드·모델 설정)
tests/             오프라인 테스트 + 가짜 통계표(fixtures)
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
