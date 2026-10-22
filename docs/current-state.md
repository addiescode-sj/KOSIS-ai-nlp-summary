# Current State

> A new session starts here. Keep this short and current — it is the handoff
> packet between sessions.

## Project

KOSIS statistics AI chatbot (국가데이터처 mentoring project): natural-language question -> find the right KOSIS table -> fetch real values via KOSIS OpenAPI -> summary + chart with source, LLM = HCX. Three stages: (1) numeric summary, due 2026-10-28, (2) table-guide chatbot with RAG, due 2026-11-04, (3) integrated chatbot with charts and multi-turn, due 2026-12-07. Owner role: AI engineering.

## Current phase

- Stage 1 → **B0 baseline**. PLAN rev 3 approved 2026-10-20. Offline code (TASKS Phase 1–5) written and passing with MockClient (44 tests).
- B0 is **not complete**: no HCX run, no golden-set run on real data yet (Phase 6 blocked).
- **T1-01 EDA done** (Linear KOS-9). Scope grew from 5 tables to all 2,364 tables. Merged to master in 5262a88 (`scripts/eda_full.py`, `notebooks/eda_full.ipynb`, `docs/eda-full.md`, `scripts/check_data_coverage.py`, TASK.md `[x]`).

## Data status (2026-10-22)

- Delivered data lives in `data_fix/` (gitignored, never commit):
  - `population_files/` — the scope definition: `raw_data/population.xls` (KOSIS 주제별통계 > 인구) → `data/population_table_catalog.jsonl`, 2,364 tables (14 surveys) with their listed periods.
  - `tables/<org>_<tbl>/` — values for all 2,364 tables in one folder: API tables have `status.json` + `*.json.gz` (1,811), the rest are KOSIS bulk ZIPs (553). Read only chunks listed in `status.json`.
  - `population/population/` — collector output: catalog, `table_metadata/`, logs.
- Coverage vs the scope (`python scripts/check_data_coverage.py` → `data_fix/coverage/`): every table has data. 2,348 complete; 15 only look short because the list says "년 (start ~ end)" while KOSIS publishes every 5th year (not missing); **1 real gap: DT_1B85032 (인구동향조사) 2018**.
- Converted to Parquet (local script `scripts/build_parquet.py`, not on master): `data_new/values/<org>_<tbl>.parquet`, one file per table, same long format for API and ZIP (org_id, tbl_id, source, prd_se, period, item_id, item_name, unit, dim1–8 name/code/label, value_raw, value, value_kind) + `tables.csv`, `table_metadata/`. 2,364 tables, 617,592,845 rows, 1.79 GB, 0 errors. No preprocessing: symbols kept in value_raw; ZIP units only when the item name has one in parentheses.
- Plan: rename current `data/` → `data_old/` and `data_new/` → `data/` (final data). Move working files that code expects (e.g. `data/golden.jsonl`, `data/cache/`) into the new `data/`.
- Older dump `data/all_data_zip/` and `data/eda_full/` results predate `data_fix/` (65 tables had no values there). `docs/eda-full.md` numbers are from that run.

## Local, uncommitted (branch `T1-01-eda-kos-9`)

- Gemini test changes: `configs/model.yaml`, `kosis_chat/llm/model.py` (retry on 429/503/504), `tests/test_gemini.py`, `eval/compare.py`.
- Partial-data EDA (5/29 tables): `notebooks/eda.ipynb`, `eda_units.ipynb`, `kosis_fetch_tables.ipynb`, `docs/eda.md`, `plans/T1-01.md`. Kept off master on purpose.
- `scripts/build_parquet.py`, `requirements.txt` (+pyarrow).
- To switch to master: commit these as a WIP commit on this branch first, then `git checkout master && git pull origin master`.
- This PC has no Python on PATH (Microsoft Store stub), so hooks (plan_gate, linear_sync) do not run here. Run Python from Anaconda Prompt (`conda activate kosis`).

## Last completed work

- Harness adopted from dev-harness; hook commands quoted for the space in the repo path.
- .gitignore covers .env / secrets / Python artifacts / data/raw / data/samples / eval/golden / eval/results.
- B0 baseline PLAN + TASKS (rev 3), Stage 1 PLAN + PREPROCESSING (rev 3) written.
- REPORT template now records FAIL / INCOMPLETE and not-run checks.

## In flight

- B0 baseline: [PLAN](plans/20261020-baseline-chatbot/PLAN.md) rev 3 (approved), [TASKS](plans/20261020-baseline-chatbot/TASKS.md) — Phase 0–5 done, Phase 6 blocked, Phase 7 open.
- Stage 1 (B1–B4): [PLAN](plans/20261020-stage1-table-summary/PLAN.md) + [PREPROCESSING](plans/20261020-stage1-table-summary/PREPROCESSING.md) DRAFT rev 3, to be finalized before B1.

## Next

1. Finalize data folders (`data/` → `data_old/`, `data_new/` → `data/`), then WIP-commit the local branch and move to master.
2. Decide whether to re-fetch DT_1B85032 2018 (only real gap).
3. T1-05 preprocessing from `docs/eda-full.md` findings (missing units, region double counting, hidden ZIP column groups). Golden set re-run on the final data.
4. Confirm HCX call method + token limit.
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

- docs/eda-full.md — contains: full-data EDA findings (2,364 tables) — use_for: T1-05 preprocessing, T2 RAG design
- data_fix/coverage/table_coverage.csv (local) — contains: per-table coverage vs scope — use_for: data gaps
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
