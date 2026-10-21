# Current State

> A new session starts here. Keep this short and current — it is the handoff
> packet between sessions.

## Project

KOSIS statistics AI chatbot (국가데이터처 mentoring project): natural-language question -> find the right KOSIS table -> fetch real values via KOSIS OpenAPI -> summary + chart with source, LLM = HCX. Three stages: (1) numeric summary, due 2026-10-28, (2) table-guide chatbot with RAG, due 2026-11-04, (3) integrated chatbot with charts and multi-turn, due 2026-12-07. Owner role: AI engineering.

## Current phase

- Stage 1 → **B0 baseline**. PLAN rev 3 approved 2026-10-20. Offline code (TASKS Phase 1–5) written and passing with MockClient (44 tests).
- B0 is **not complete**: no HCX run, no real table, no golden set yet (Phase 6 blocked).
- No real data in the repo (tests use made-up tables in tests/fixtures/).
- **Active task: T1-01 EDA** (Linear KOS-9, due 2026-10-22), branch `T1-01-eda-kos-9` (pushed, head 72003de). Team workflow is plan-gate (CLAUDE.md §0); [plans/T1-01.md](../plans/T1-01.md) is approved (Option B: B0 loader/cleaner + metadata hierarchy in a notebook).

## T1-01 handoff (2026-10-21)

- Shared data in `data/` (gitignored, not committed): 14 catalog CSVs (one per survey, 2,364 tables total, UTF-8 BOM) and `통계표_메타데이터.jsonl` (34 MB, same 2,364 tables: classifications with lvl/upItmId/leaf, items, period code lists). **No statistic values** in either. 879 tables have an empty unit.
- Downloading every table is not planned (≥144M cells by metadata). Values are fetched only for the 5 target tables:
  DT_1B8000G (B0 table), DT_1B26006_A01 (%), DT_1B83A35 (monthly), DT_1BPA001 (multi-level, projection; cut at 2025), DT_1B81A24 (blank unit).
- `notebooks/kosis_fetch_tables.ipynb`: fetches latest 10 periods per period type into `data/raw/kosis/<survey>/<org>_<tbl>_<prdSe>.json` + `.meta.json` (format B), cache in `data/cache/kosis/`. First run (15:22) failed for all 5 with `'PRD_DE'` (PRD meta gives ranges only); fixed in 72003de to take period codes from the shared metadata. **Re-run OK at 15:27**: all 5 tables `ok`, 10 value files (M/Q/Y per table), 21,373 cells (DT_1B8000G 7,415 · DT_1B26006_A01 1,620 · DT_1B83A35 5,646 · DT_1BPA001 6,642 · DT_1B81A24 50). Loader pass/fail is checked in EDA unit 1.
- `notebooks/eda_units.ipynb`: EDA split into units 0–8, each code cell has a markdown cell above (what / how / how to read / what goes into docs/eda.md). Owner wants to run and interpret **one unit at a time**. `notebooks/eda.ipynb` is the older all-in-one version with the owner's uncommitted local edits (outputs).
- Next: EDA unit 0–1 (loader pass/fail) → units 2–7 one by one → write `docs/eda.md` (structure and counts only, no cell values) → fill `## Result` in plans/T1-01.md → TASK.md `[x]`.
- Local only, uncommitted: `configs/model.yaml` (duplicate `openai_compat` block merged; backend `openai_compat` with OpenRouter nemotron — owner to decide commit vs `gemini`), `notebooks/eda.ipynb`.
- This PC has no Python on PATH (Microsoft Store stub), so hooks (plan_gate, linear_sync) do not run here and notebooks were not executed by Claude. Linear MCP is not connected yet; KOS-9 status was not synced.

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

- KOSIS OpenAPI key works (fetch run on 2026-10-21 got meta responses); HCX access and token limit not verified.
- B0 completion needs real HCX evidence; Mock tests alone do not complete B0.

## How to verify

- `bash scripts/verify.sh` — offline only (no network, no API key): pytest (MockClient) + checks that real-data paths are gitignored and untracked.
- Try it: `python -m kosis_chat.cli --table tests/fixtures/pop_a.csv --backend mock -q "2023년 여자 인구는?"`
- HCX quality evaluation is manual (not part of verify.sh): `python -m eval.run_eval --stage B0` → `python -m eval.review export|apply --run eval/results/<id>` → `python -m eval.scorer --run eval/results/<id> --append-history`
- Default model (2026-10-21, user decision): Gemini `gemini-3.8-flash` via google-genai (`backend: gemini`, thinking_level low, no temperature). Key `GEMINI_API_KEY` in .env or Colab secrets. Run: `python -m kosis_chat.cli --table <file> -q "..."` or notebooks/colab_run.ipynb. Gemini runs can go to eval-history (model column records backend:model); HCX stays the final target.
- Other pre-test models: `openai_compat` block + `PRETEST_API_KEY`; these runs (and mock) are marked `pretest` and never appended to eval-history.

## Context Index

- plans/T1-01.md — contains: approved EDA plan, measurement table (9 items mapped to R1–R9), target tables — use_for: T1-01
- TASK.md — contains: team roadmap T0–T3 with Linear IDs, deadlines, dependencies — use_for: picking tasks

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
