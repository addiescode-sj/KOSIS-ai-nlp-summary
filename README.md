# T1-28 S0 반영 방법 (PC Phase 1 레포)

Phase 1 레포(`바탕 화면\KOSIS-ai-nlp-summary`)에서 master 최신(e248a45)을 받은 뒤 브랜치를 만들고 아래 파일을 넣어 주세요.

```bat
cd "%USERPROFILE%\OneDrive\바탕 화면\KOSIS-ai-nlp-summary"
conda activate kosis
git switch master && git pull && git switch -c T1-28-s0-baseline
```

## 1. 파일 복사 (CRLF)
- `eval/parity.py` (신규) sha256 a6190ce3f5ca386d…
- `tests/test_parity.py` (신규) sha256 501b716bb8f54210…
- `plans/T1-28.md` (신규, status: approved) sha256 8a14ab2e6da5fd65…

## 2. TASK.md에 두 줄 추가 (파일을 덮어쓰지 말고 붙여 넣기)
T1-26 줄 바로 아래:
```
- [~] **T1-28** Phase 1 모듈 리팩터 (단계 폴더 + LangGraph 노드 + StatModule 진입점) — `StatPipeline.ask`를 단계 폴더·Pydantic In/Out 노드·LangGraph 그래프로 재구성하고 진입점을 `StatModule(...).answer(StatInput) -> StatOutput`으로 바꾼다. 단계 S0~S7마다 mock 90문항 결과 바이트 동일(`python -m eval.parity`), S5에서 HCX 판정 90/90·PASS 86/90 유지, 단계 병합마다 Phase 2 핀 범프 (진행: S0 패리티 도구 `eval/parity.py`, `plans/T1-28.md`)
```
일정 표의 `| T1-15 | KOS-23 |…` 줄 바로 아래:
```
| T1-28 | KOS-124 | 11/25 | P1 | T1-11 | 전민기 |
```
(마감 11/25·P1·선행 T1-11은 제가 정한 값이에요. 다르면 고쳐 주세요.)

## 3. 실행
```bat
set KOSIS_WORKSPACE=%CD%
python -m pytest tests/test_parity.py -q
python -m pytest -q
python -m eval.run_eval --pipeline stat --backend mock --stage T1-28-s0-mock-a --interval 0
python -m eval.run_eval --pipeline stat --backend mock --stage T1-28-s0-mock-b --interval 0
python -m eval.parity --base eval\results\<mock-a 폴더> --run eval\results\<mock-b 폴더>
```
- `--golden`은 기본값이 `eval/golden/golden.jsonl`이에요. T1-27 때 90문항을 돌린 골든 파일이 따로 있으면 두 줄 모두에 같은 `--golden <파일>`을 붙여 주세요.
- 마지막 줄이 `identical`이 나오면 S0 완료예요. 다르게 나오면 출력 그대로 보내 주세요. 집합(set) 순서 때문에 실행마다 달라질 수 있는 곳(judge/answerability.py:137-139, 273-275)을 먼저 보고, 제외 목록을 늘리지 않고 정렬로 고칠게요.
