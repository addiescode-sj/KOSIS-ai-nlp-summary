# Current State

> A new session starts here. Keep this short and current — it is the handoff
> packet between sessions.

## Project

KOSIS statistics AI chatbot (국가데이터처 mentoring project): natural-language question -> find the right KOSIS table -> fetch real values via KOSIS OpenAPI -> summary + chart with source, LLM = HCX. Three stages: (1) numeric summary, due 2026-10-28, (2) table-guide chatbot with RAG, due 2026-11-04, (3) integrated chatbot with charts and multi-turn, due 2026-12-07. Owner role: AI engineering.

## Current phase

- Stage 1 → **B0 baseline**. PLAN rev 3 approved 2026-10-20. Offline code (TASKS Phase 1–5) written and passing with MockClient (44 tests).
- B0 is **not complete**: no HCX run, no real table, no golden set yet (Phase 6 blocked).
- No real data in the repo (tests use made-up tables in tests/fixtures/).

## Last completed work

- Harness adopted from dev-harness; hook commands quoted for the space in the repo path.
- .gitignore covers .env / secrets / Python artifacts / data/raw / data/samples / eval/golden / eval/results.
- B0 baseline PLAN + TASKS (rev 3), Stage 1 PLAN + PREPROCESSING (rev 3) written.
- REPORT template now records FAIL / INCOMPLETE and not-run checks.

## In flight

- B0 baseline: [PLAN](plans/20261020-baseline-chatbot/PLAN.md) rev 3 (approved), [TASKS](plans/20261020-baseline-chatbot/TASKS.md) — Phase 0–5 done, Phase 6 blocked, Phase 7 open.
- Stage 1 (B1–B4): [PLAN](plans/20261020-stage1-table-summary/PLAN.md) + [PREPROCESSING](plans/20261020-stage1-table-summary/PREPROCESSING.md) DRAFT rev 3, to be finalized before B1.

## Next

1. Owner runs `bash scripts/verify.sh` locally after `pip install -r requirements.txt` (only verified in a Linux container so far) and commits.
2. 2026-10-21: confirm HCX call method + token limit, pick the B0 table. 2026-10-22: golden set v1 owner and content.
3. Phase 6: implement HCXClient (T-023), then B0 HCX run → human review → confirmed baseline score in docs/reports/eval-history.md.

## Blocked / unverified

- KOSIS OpenAPI key, HCX access and token limit, provided metadata format — not verified (2026-10-21).
- B0 completion needs real HCX evidence; Mock tests alone do not complete B0.

## How to verify

- `bash scripts/verify.sh` — offline only (no network, no API key): pytest (MockClient) + checks that real-data paths are gitignored and untracked.
- Try it: `python -m kosis_chat.cli --table tests/fixtures/pop_a.csv --backend mock -q "2023년 여자 인구는?"`
- HCX quality evaluation is manual (not part of verify.sh): `python -m eval.run_eval --stage B0` → `python -m eval.review export|apply --run eval/results/<id>` → `python -m eval.scorer --run eval/results/<id> --append-history`
- Default model (2026-10-21, user decision): Gemini `gemini-3.8-flash` via google-genai (`backend: gemini`, thinking_level low, no temperature). Key `GEMINI_API_KEY` in .env or Colab secrets. Run: `python -m kosis_chat.cli --table <file> -q "..."` or notebooks/colab_run.ipynb. Gemini runs can go to eval-history (model column records backend:model); HCX stays the final target.
- Other pre-test models: `openai_compat` block + `PRETEST_API_KEY`; these runs (and mock) are marked `pretest` and never appended to eval-history.

## Context Index

- docs/plans/20261020-baseline-chatbot/PLAN.md — contains: file layout, input formats + TableInput fields, data/commit policy, prompt rules, API errors, comparison protocol, scoring rules — use_for: B0 implementation
- docs/plans/20261020-baseline-chatbot/TASKS.md — contains: B0 tasks by phase with blocked items — use_for: /execute
- docs/plans/20261020-stage1-table-summary/PLAN.md — contains: stage 1 flow, calc plan + evidence rows, template rendering + validation V1–V5, service target criteria — use_for: B1–B4
- docs/plans/20261020-stage1-table-summary/PREPROCESSING.md — contains: derived fields, rules R1–R9, test cases — use_for: B1 implementation
- docs/notes/ — contains: reference docs that are not plans or reports (project-workflow.md: schedule, workflow, stage-1 flow, service diagram with libraries) — use_for: overview, onboarding
- docs/reports/ — contains: completed work reports, eval-history.md (after B0) — use_for: history and evidence

## Hard rules from the project brief

- Use only numbers traceable to a source cell or to a verifiable formula over source cells; cite the source table for every key figure.
- No causal or political claims the numbers cannot support.
- API keys live only in .env (gitignored); never commit them. Real data, golden sets and eval results are not committed.
