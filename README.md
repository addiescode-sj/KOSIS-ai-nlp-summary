# dev-harness

개발 작업 특화 범용 하네스. 하네스 엔지니어링 강의의 핵심 원칙(SDD, 역할 분리,
컨텍스트-문서 분리, hook 기반 완료 검증)을 어떤 프로젝트에든 이식할 수 있는
형태로 담았습니다.

## 핵심 설계

1. **문서가 진실의 원천** — 계획(PLAN.md), 작업(TASKS.md), 결과(REPORT.md),
   현재 상태(current-state.md)는 전부 파일로 남습니다. 대화 컨텍스트에는
   Handoff Index만 남기고, 새 세션은 `docs/current-state.md`만 읽고 이어갑니다.
2. **역할 분리는 도구 권한으로 강제** — `.claude/agents/`의 5개 subagent는
   frontmatter `tools` allowlist로 hard boundary를 갖습니다.
3. **완료는 hook이 판정** — Stop hook이 `scripts/verify.sh`를 실행해 실패하면
   에이전트가 완료 선언을 못 합니다.

## 워크플로우

```
/plan <요청>      planner가 의도(암묵지 포함) 분석 → docs/plans/<id>/PLAN.md
   ↓ 사용자 승인 (승인 없이는 진행 불가)
/tasks            PLAN.md → micro task 분해 → docs/plans/<id>/TASKS.md
   ↓
/execute          subagent 최대 5개로 phase별 실행 + phase마다 verifier 검증
   ↓
/wrapup           scribe가 docs/reports/<id>-REPORT.md 작성 + current-state 갱신
                  (Stop hook이 verify.sh 통과를 강제)
```

## 구성 요소

| 경로 | 역할 |
| --- | --- |
| `AGENTS.md` / `CLAUDE.md` | 하네스 운영 규칙 (컨텍스트 위생, 완료 기준) |
| `.claude/agents/` | planner / explorer / implementer / verifier / scribe |
| `.claude/skills/` | /plan, /tasks, /execute, /wrapup |
| `.claude/settings.json` | 권한(deny/allow) + SessionStart/PreToolUse/Stop hooks |
| `scripts/session_start.ts` | 새 세션에 current-state와 미완료 plan 주입 |
| `scripts/bash_guard.ts` | 파괴적 명령(rm -rf /, force push 등) 차단 |
| `scripts/stop_verify.ts` | 완료 선언 전 verify.sh 강제 실행 |
| `scripts/verify.sh` | 프로젝트별 검증 명령 등록 지점 |
| `docs/templates/` | PLAN / TASKS / REPORT 템플릿 |

Hook 스크립트는 TypeScript(zod로 hook payload 검증)이며 Node.js 23.6+의
네이티브 TS 실행으로 빌드 없이 동작합니다.

## 다른 프로젝트에 적용하기

```bash
# 프로젝트 루트에서
cp -R /path/to/dev-harness/.claude .
cp -R /path/to/dev-harness/scripts .
mkdir -p docs && cp -R /path/to/dev-harness/docs/templates docs/
cp /path/to/dev-harness/docs/current-state.md docs/
cp /path/to/dev-harness/AGENTS.md /path/to/dev-harness/CLAUDE.md .
npm install zod   # hook scripts need zod (Node.js >= 23.6)
```

그 다음 두 가지만 프로젝트에 맞게 수정:

1. `scripts/verify.sh` — 테스트/린트 명령 추가 (Stop hook이 이걸 실행)
2. `docs/current-state.md` — 프로젝트 설명 한 줄

## 사용 예

```
> /plan 로그인 실패 5회 시 계정 잠금 기능 추가

  → planner가 codebase 조사 + 모호한 지점은 [NEEDS CLARIFICATION]으로 질문
  → docs/plans/20260720-account-lockout/PLAN.md 생성
  → "PLAN.md를 검토하고 승인해주세요"

> 승인

> /tasks   → TASKS.md 생성 (T-001..., [P] 병렬 표시, AC 추적)
> /execute → implementer/verifier가 phase별 실행·검증, TASKS.md 체크 갱신
             완료 시 REPORT.md + current-state.md 갱신 후 Handoff Index 보고
```
