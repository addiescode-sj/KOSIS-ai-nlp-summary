# KOSIS 통계 AI 서비스

멋쟁이사자처럼 AI NLP 부트캠프 6기 팀 프로젝트 (국가데이터처 통계 데이터 AI 서비스 과제).
자연어 질문 하나로 **통계표 찾기 → 수치 조회 → 계산 → 요약 → 차트**까지 하고, 후속 질문도 이어서 처리하는
국가통계(KOSIS) 챗봇을 단계별로 만든다.

> 핵심 원칙: **숫자는 코드가 계산하고 LLM은 문장만 쓴다.** 답할 수 없는 질문에는 숫자를 지어내지 않고 이유를 말한다.
> 과제에 합격 점수는 없다. "무엇을 바꿔서 몇 → 몇"을 수치로 남기는 것이 결과물이다.

## 단계와 산출물

세 단계는 별개 과제가 아니다. 실전 1의 모듈을 실전 2가 쓰고, 종합에서 둘을 합쳐 완성품을 만든다.

| | 실전 1 | 실전 2 | 종합 |
| --- | --- | --- | --- |
| 기간 | 10/20 ~ 10/28 | 10/28 ~ 11/4 | 11/4 ~ 12/10 |
| 답하는 질문 | "이 표에서 숫자가 어때?" | "어느 표를 봐야 해?" | "찾아서, 계산해서, 그려줘" |
| 만드는 것 | 통계 수치 요약 모듈 | 통계표 안내 챗봇 (RAG) | 툴 호출 에이전트 + 시각화 |
| 핵심 기술 | 조회·계산·요약, 예외 처리 | 검색·순위·답 없음 판정 | 툴 선택, 멀티턴, 차트 |
| 핵심 지표 | 수치 정확도, 불가 질문 거절률 | Hit@k, MRR, 답 없음 F1 | 툴·파라미터 정확도, 멀티턴 성공률, 비용·지연 |
| 다음 단계로 넘기는 것 | 조회·계산·검증 모듈 | `search_tables()` 검색 모듈 | 완성 서비스 |

채점 항목은 모든 단계 공통이다: 기능 완성도, **데이터 정확성(가장 중요)**, 질의 이해, 응답 품질, 예외 처리, 기술 설계, 결과 검증.
배점은 개인 기술 성과 40% · 팀 완성도 30% · 협업 20% · 발표·문서 10%.

## 동작 예시

```
Q  인구동향조사에서 최근 5년 출생아 수가 어떻게 변했어?
A  2020년 [값]명에서 2024년 [값]명으로 [값]% 감소했습니다. 가장 큰 감소는 [연도]입니다.
   출처: KOSIS / 인구동향조사 / [통계표명]

Q  출생아가 줄어든 원인이 뭐야?
A  통계 수치만으로는 원인을 판단할 수 없습니다. 연도별 출생아 수와 증감률은 안내할 수 있습니다.
```

## 현재 상태 (2026-10-21)

- **실전 1 · B0 베이스라인**: 표 데이터 + 질문 → LLM → 답변이 끝까지 도는 가장 단순한 챗봇(`kosis_chat/`)과
  평가 도구(`eval/`)의 오프라인 코드가 완성됐고 Mock 기반 테스트(44개)가 통과한다.
- 아직 안 된 것: 실제 LLM 실행, 실제 통계표, 골든셋. 그래서 B0 기준 점수는 아직 없다.
- B0 이후 한 번에 하나씩 바꾸며 B1(전처리) → B3(코드 계산) → B4(숫자 검증)로 올리고, 같은 골든셋으로 `eval-history.md`에 한 줄씩 쌓는다.
- 최신 상태와 막힌 항목은 [docs/current-state.md](docs/current-state.md)를 본다 (새 세션·새 팀원의 시작점).

## 빠르게 실행해 보기

```bash
pip install -r requirements.txt            # Python 3.12 권장
cp .env.example .env                       # 키는 .env에만 둔다 (커밋 금지)

bash scripts/verify.sh                     # 오프라인 검증: pytest(Mock) + 실데이터 커밋 차단 확인

# API 키 없이 흐름만 확인 (Mock)
python -m kosis_chat.cli --table tests/fixtures/pop_a.csv --backend mock -q "2023년 여자 인구는?"

# 실제 모델로 질문 (.env에 GEMINI_API_KEY 필요, 기본 백엔드: Gemini)
python -m kosis_chat.cli --table <표 파일> -q "질문"
```

| 환경변수 | 용도 |
| --- | --- |
| `KOSIS_API_KEY` | KOSIS OpenAPI |
| `GEMINI_API_KEY` | 개발용 기본 모델 (Gemini) |
| `HCX_API_KEY`, `HCX_API_URL` | 최종 대상 모델 HCX (호출 방식 확인 중) |
| `PRETEST_API_KEY` | OpenAI 호환 API로 다른 모델 사전 테스트 |
| `LINEAR_API_KEY`, `LINEAR_TEAM_KEY` | TASK.md → Linear 자동 동기화 (선택) |

백엔드·모델 설정은 [configs/model.yaml](configs/model.yaml). Mock·사전 테스트(`openai_compat`) 실행 결과는 `eval-history`에 올리지 않는다.

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
