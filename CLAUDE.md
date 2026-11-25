# KOSIS 통계 AI 서비스 — 작업 지침

멋쟁이사자처럼 AI NLP 부트캠프 6기 팀 프로젝트. 국가데이터처(KOSIS) 통계표를 대상으로
**실전 1(수치 요약 모듈) → 실전 2(통계표 검색 챗봇) → 종합(툴 호출 에이전트 + 시각화)**을 만든다.
앞 단계의 모듈을 다음 단계가 그대로 재사용하는 구조다. 로드맵과 진행 상황은 [TASK.md](TASK.md).

## 0. 작업 절차 (필수 — hook으로 강제됨)

모든 TASK는 `plan-gate` skill(`/plan-gate`)로 진행한다. 산출물부터 만들지 않는다.

1. **Pick** — TASK.md에서 태스크를 `[~]`로 바꾸고 브랜치 `T1-05-slug` 생성
2. **Plan** — `plans/<TASK-ID>.md`에 수행 아이디어와 옵션 A/B/C 작성
   (옵션마다 예상 성능 향상 수치 + 실질 반영 이점 + 비용·리스크)
3. **Verify** — Verify 체크리스트를 모두 통과시키고, 독립 리뷰(서브에이전트) 결과 반영
4. **Decide** — 사용자에게 옵션 선택을 받아 `status: approved` 기록
5. **Implement** — 선택한 옵션만 구현
6. **Measure** — 평가 하네스로 전→후 수치를 plan의 `## Result`와 TASK.md에 기록, `[x]` 처리

`.claude/hooks/plan_gate.py`가 승인된 plan 없이 `plans/`, `TASK.md`, `CLAUDE.md`, `.claude/` 밖의
파일을 수정하면 차단한다. 우회하지 말 것 (Bash로 파일 쓰기 포함).

## 1. 평가 기준 (모든 판단의 기준)

데이터 정확성 > 예외 처리 > 질의 이해 > 기능 완성도 > 응답 품질 > 기술 설계 > 결과 검증.
과제에 합격 점수는 없다. **"무엇을 바꿔서 몇 → 몇"**을 남기는 것이 결과물이다.
배점: 개인 기술 성과 40 / 팀 완성도 30 / 협업 20 / 발표·문서 10 → 개인 담당 부분의 수치를 꼭 남긴다.

## 2. 설계 원칙

- **숫자는 코드(pandas)가 계산하고 LLM은 문장만 쓴다.** LLM이 만든 숫자는 버그다.
- 최종 답의 모든 숫자는 숫자 검증기(verifier)를 통과해야 한다.
- 모든 답에 출처: `KOSIS / 조사명 / 통계표명 (+링크)`.
- 답할 수 없으면 숫자를 지어내지 말고 이유를 말한다 (없는 연도, 분류 불일치, 단위 다름, 인과 질문, 결측, 범위 밖).
- 모듈 입출력은 Pydantic 스키마로 고정한다. 실전 2·종합이 다시 짜지 않고 import해서 쓴다.
  스키마를 깨는 변경은 plan에 명시하고 승인받는다.
- 단계별 공개 API (종합에서 툴이 됨):
  - 실전 1: `summarize_stat(question, table_id) -> StatAnswer`
  - 실전 2: `search_tables(question, k) -> list[TableCandidate]` (+ 답 없음 판정)
  - 종합: 위 둘 + `make_chart(data, kind) -> Chart`를 에이전트 툴로 등록
- 필요한 것만 만든다. 구현 하나·인터페이스 하나면 추상화하지 않는다.

## 3. 디렉터리 (T0-01에서 생성)

```
src/kosis_ai/      schemas.py, kosis_client.py, normalize.py, slots.py, compute.py,
                   answerability.py, summarize.py, verify.py, search/, agent/, chart.py
eval/golden/       stat_golden.jsonl (실전1), search_golden.jsonl (실전2), agent_golden.jsonl (종합)
eval/              harness.py — 모든 단계 공통 채점기, eval/reports/에 날짜별 결과
plans/             태스크별 plan (옵션·검증·결정·결과 기록)
docs/              제출 문서 (질문 유형, 검색 전략, ADR, 한계·개선)
```

## 4. 코딩 규칙

- 코드·주석·식별자는 영어. 문서(md)는 한국어.
- Python 3.11+, `uv`, Pydantic v2, pandas, pytest. 비밀값은 `.env` (`KOSIS_API_KEY`, LLM 키) — 커밋 금지.
- KOSIS API 응답은 `data/cache/`에 캐시해 평가가 재현 가능하게 한다.
- 평가 실행: `uv run python -m eval.harness --suite stat|search|agent` (T1-11에서 확정 후 갱신)
- 비자명한 로직(계산·검증·판정)은 pytest 하나 이상을 남긴다.
- 베이스라인(B0) 코드는 `kosis_chat/`, 설계 문서는 `docs/plans/`, 개인 하네스 지침은 `docs/notes/dev-harness-CLAUDE.md`.
