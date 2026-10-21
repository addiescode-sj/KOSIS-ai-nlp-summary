# Tasks: 베이스라인 챗봇 (B0)

- plan: docs/plans/20261020-baseline-chatbot/PLAN.md (rev 3)
- status: IN_PROGRESS (Phase 0–5 done offline on 2026-10-20; Phase 6 blocked on HCX / table / golden set)

## Phase 0: Design fixes (docs only)

- [x] T-000 rev 3 반영: 입력 형식·TableInput 통일, 채점 기준(상호 배타·통계 수치 주장·단위별 허용오차), B0 사람 판정 절차, 데이터 커밋 정책, API 오류·입력 길이, 완료 조건 분리
  > done: PLAN.md rev 3, stage1 PLAN rev 3, PREPROCESSING rev 3, current-state, REPORT template

## Phase 1: Package, config, Mock (can start first)

- [x] T-001 [P] `requirements.txt` + `.env.example`(키 이름만) + `configs/model.yaml`(backend: mock, max_input_tokens: TBD, eval.runs: 1) (NFR-002, NFR-003)
  > done: requirements.txt, .env.example, configs/model.yaml
- [x] T-002 [P] `kosis_chat/llm/model.py` — LLMClient, LLMResult(text, status, usage), MockClient, HCXClient(인터페이스만) (FR-005, NFR-004)
  > done: kosis_chat/llm/model.py
- [x] T-003 `kosis_chat/config.py` — 프로젝트 루트 기준 경로, .env·model.yaml 로드 (D-004) after: T-001
  > done: kosis_chat/config.py
- [x] T-004 [P] `.gitignore` — eval/golden/, eval/results/ 추가 (NFR-005, AC-005)
  > done: .gitignore (data/samples/, eval/golden/, eval/results/), check-ignore not run yet

## Phase 2: Schema & loader (input contract fixed in PLAN rev 3)

- [x] T-005 `kosis_chat/data/schema.py` — TableMeta·Cell·TableInput, validate() (FR-001)
  > done: kosis_chat/data/schema.py
- [x] T-006 [P] `tests/fixtures/` — 가상 표: 형식 A 정상, 형식 A 메타 누락, 형식 B 정상(+meta.json), 형식 C wide, CP949 파일 (AC-001)
  > done: tests/fixtures/ (A 정상·코드 포함, 메타 누락, B 정상, wide; CP949는 테스트에서 생성)
- [x] T-007 `kosis_chat/data/loader.py` — 형식 A·B 어댑터, C·메타 누락·인코딩 오류 거부 (FR-001) after: T-005
  > done: kosis_chat/data/loader.py
- [x] T-008 `kosis_chat/data/preprocessing.py` — value·value_flag 채우기, 원본 필드 유지 (FR-002) after: T-007
  > done: kosis_chat/data/preprocessing.py
- [x] T-009 `tests/test_data.py` — 거부 사유, 전처리 후 unit·value_raw·source_ref 유지 (AC-001) after: T-006, T-008
  > done: tests/test_data.py, pytest pass
  - verify: `pytest tests/test_data.py`

## Phase 3: Baseline pipeline

- [x] T-010 `kosis_chat/data/serializer.py` — 메타 헤더 + 셀 단위가 보이는 마크다운 (FR-003) after: T-008
  > done: kosis_chat/data/serializer.py
- [x] T-011 [P] `kosis_chat/llm/prompts.py` — `baseline-v1` (공통 수치 규칙 문구, 계산식 예시) (FR-004)
  > done: kosis_chat/llm/prompts.py
- [x] T-012 `kosis_chat/chatbot.py` — ask() → Answer(text, status, …), 토큰 한도 초과 시 input_too_long (FR-003, FR-005) after: T-002, T-003, T-010, T-011
  > done: kosis_chat/chatbot.py
- [x] T-013 `kosis_chat/cli.py` — `--table` 지정 후 질문 반복, 비정상 상태 표시 (FR-006) after: T-012
  > done: kosis_chat/cli.py, Mock CLI run OK
- [x] T-014 `tests/test_chatbot.py` — Mock으로 흐름, 상태별(ok/timeout/api_error/input_too_long) 처리 (AC-001) after: T-012
  > done: tests/test_chatbot.py, pytest pass
- [x] T-015 `scripts/verify.sh` — pytest 등록, 네트워크·키 없이 통과 (AC-003) after: T-014
  > done: scripts/verify.sh (pytest + gitignore/tracked checks), pytest.ini
  - verify: `bash scripts/verify.sh`

## Phase 4: Evaluation tooling (offline-testable)

- [x] T-016 `eval/run_eval.py` — 데이터 해시 확인, 문항 결과 기록 필드 + manifest 저장, runs≥2면 실행별 보존 (FR-007) after: T-012
  > done: eval/run_eval.py
- [x] T-017 `eval/review.py` — 판정 시트 생성(기대 facts 함께 표시) / 확정 판정 반영 (FR-008) after: T-016
  > done: eval/review.py
- [x] T-018 `eval/scorer.py` — 확정 판정 집계, 전체·확정·미검토·잠정·실행 실패, eval-history.md 행 추가 (FR-009, FR-010) after: T-016
  > done: eval/scorer.py
- [x] T-019 `tests/test_eval.py` — 판정 순서, 미검토 → 잠정, 미검토·실행 실패 미제외, Mock 결과로 run_eval→review→scorer 왕복 (AC-002) after: T-017, T-018
  > done: tests/test_eval.py, pytest pass
- [x] T-020 [P] `notebooks/colab_run.ipynb` — 설치·CLI 실행만 (NFR-003) after: T-013
  > done: notebooks/colab_run.ipynb
- [x] T-021 [P] 모든 모듈 첫 줄 docstring에 역할 한 문장 (AC-004) after: T-019
  > done: all modules start with a one-line docstring (checked)

## Phase 5: Offline verification

- [x] T-022 `bash scripts/verify.sh` + `git check-ignore data/samples eval/golden eval/results` (AC-001~AC-005, 오프라인만) after: T-015, T-019, T-004
  > done: verify.sh exit 0 in a temp git repo (44 passed); tracked data/samples file makes it fail as intended. Not yet run on the owner's machine

## Phase 6: B0 run with HCX (blocked until owner inputs arrive)

- [ ] T-023 `kosis_chat/llm/model.py` — HCXClient 구현, max_input_tokens 기입 (FR-005) — BLOCKED: HCX 호출 방식 (10/21)
- [ ] T-024 B0 대상 표 1개를 형식 A 또는 B로 `data/samples/`에 준비 (AC-006) — BLOCKED: 대상 표 선정 (10/21)
- [ ] T-025 `eval/golden/v1/` 골든셋 작성 (문항 종류 4가지, fact별 계산식·입력 셀·허용오차) — BLOCKED: 담당·내용 (10/22) after: T-024
- [ ] T-026 HCX로 CLI 실행 확인 (AC-006) after: T-023, T-024
- [ ] T-027 `python -m eval.run_eval --stage B0` → review로 전 문항 confirmed → scorer로 eval-history.md 첫 행 (AC-007, AC-008) after: T-025, T-026

## Phase 7: Completion

- [ ] T-028 완료 판정: T-022(오프라인) 결과 + T-026·T-027(HCX 실행·확정 점수) 증거를 함께 확인. HCX 미실행·실패·잠정 점수면 완료가 아니라 REPORT에 INCOMPLETE로 기록 after: T-022, T-027
- [ ] T-029 Wrapup: REPORT + current-state 갱신 after: T-028

<!--
Conventions:
- [P] = parallelizable (disjoint file scope from other [P] tasks in the phase)
- after: T-xxx = dependency
- Execution appends one result line under each finished task:
  > done: <files>, <verify result>
-->
