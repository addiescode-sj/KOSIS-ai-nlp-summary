# 골든셋 프롬프트 테스트 방법

골든셋 문항을 HCX에 보내 답변을 받고, 사람이 판정한 뒤 점수를 집계하는 순서를 정리한다.
`--form`으로 답변 형식 지시 파일을 바꿔 가며 프롬프트를 실험할 때도 이 순서를 그대로 쓴다.
(2026-10-22 기준, 튜닝 모델 HCX-DASH-002)

## 1. 준비

| 항목 | 위치 | 비고 |
| --- | --- | --- |
| HCX 키 | `.env`의 `HCX_API_KEY=nv-...` | 커밋 금지 |
| 모델 설정 | `configs/model.yaml`의 `model` 블록 | 기본 `HCX-DASH-002`, temperature 0, max_tokens 1024 |
| 골든셋 | `eval/golden/b0-loadable/stat_golden_loadable.jsonl` (22문항) | gitignore, 팀 공유본에서 받는다 |
| 표 파일 | `data/cache/<표>.json` + 같은 이름의 `.meta.json` | gitignore. 골든셋의 `table` 경로와 sha256이 맞아야 실행된다 |
| 답변 형식 지시 | 예: `eval/golden/form.txt` | 자주 바뀌므로 인자로 넘긴다 |

- 전체 골든셋(`eval/golden/stat_golden.jsonl`, 40문항) 중 DT_1B8000F·DT_1B8000H 표는 응답에 단위(UNIT_NM)가 없어 B0 로더가 읽지 못한다.
  그래서 지금은 나머지 4개 표 22문항만 모은 `b0-loadable`을 쓴다.
- 실행은 Anaconda Prompt에서 `conda activate kosis` 후 레포 루트에서 한다.

## 2. 프롬프트가 만들어지는 방식

문항마다 HCX에 아래 두 메시지를 보낸다.

1. **system**: `kosis_chat/llm/prompts.py`의 `SYSTEM_PROMPTS[prompt_version]` (현재 `baseline-v1`)
2. **user**:
   ```
   [통계표]
   <메타 헤더 + 표 전체 Markdown>

   [질문]
   <골든셋 question>

   [답변 형식]          ← --form을 줬을 때만 붙는다
   <form 파일 내용>
   ```

- `--form` 파일은 UTF-8로 읽고 앞뒤 공백은 지운다. 파일 내용은 그대로 붙으므로 제목이나 접두어가 필요하면 파일 안에 쓴다.
- 시스템 프롬프트와 form 지시가 서로 다르면(예: 출처를 쓰라 / 생략하라) 모델이 둘 다 받는다.
  form으로 정한 형식을 계속 쓸 거라면 `prompts.py`에 새 버전을 추가하고 `model.yaml`의 `prompt_version`을 바꾼다.

## 3. 실행

```bat
python -m eval.run_eval --stage <이름> --golden eval/golden/b0-loadable --interval 5 --form eval/golden/form.txt
```

| 인자 | 뜻 |
| --- | --- |
| `--stage` | 결과 폴더 이름에 붙는 실험 이름. 예: `B0-dash`, `B0-dash-form-v2` |
| `--golden` | 골든셋 폴더 또는 .jsonl 파일 |
| `--interval N` | 문항 사이 N초 대기. HTTP 429(호출 한도)를 막는다. 5초 권장, 429가 남으면 15초 |
| `--form PATH` | 답변 형식 지시 파일. 빼면 form 없이 실행 |
| `--model` | 모델 바꾸기. 예: `HCX-007` |
| `--thinking`, `--max-tokens` | HCX-007 thinking 설정. 켜면 max-tokens를 5120 정도로 늘린다 |
| `--backend mock` | 키 없이 흐름만 확인 (사전 테스트, 점수 기록 안 됨) |

결과는 `eval/results/<시각>-<stage>/`에 저장된다 (gitignore).

| 파일 | 내용 |
| --- | --- |
| `results.jsonl` | 문항별 답변(`raw_answer`), 호출 상태(`call_status`), 오류, 판정 |
| `manifest.json` | 모델·프롬프트 버전·골든셋·데이터 해시, form 경로와 sha256 |
| `golden.jsonl` | 이번 실행에 쓴 골든셋 사본 |
| `form.txt` | `--form`을 줬을 때 그 시점의 form 파일 사본 |

### 호출 실패 확인

호출이 실패한 문항은 자동으로 FAIL(`run_failure`)이 된다. 판정 전에 실패 원인을 먼저 본다.

```bat
python -c "import json;[print(r['question_id'],r['call_status'],(r['call_error'] or '')[:150]) for r in map(json.loads,open(r'eval\results\<실행 폴더>\results.jsonl',encoding='utf-8')) if r['call_status']!='ok']"
```

| 오류 | 원인과 대응 |
| --- | --- |
| `HTTP 429` (42901) | 호출 한도. 자동 재시도(5/15/30/60초) 후에도 실패한 것 → `--interval`을 늘린다 |
| `input_too_long` (40003 Context length exceeded) | 표가 모델 입력 한도보다 크다. DT_1JC1502(1,814행)는 DASH에서 항상 이렇게 된다 → 프롬프트로는 해결 안 됨, 표를 줄이는 전처리가 필요 |
| `HTTP 401` | 키가 틀림 |
| `HTTP 404` | 키에 열려 있지 않은 모델 이름 |

## 4. 사람 판정

```bat
python -m eval.review export --run eval\results\<실행 폴더>
```

1. `review.csv`를 엑셀로 연다. 호출에 성공한 문항만 들어 있다.
2. 행마다 `expected`(정답 fact)와 `raw_answer`를 비교한다.
3. 틀린 점이 있으면 `error_types`에 적는다(여러 개는 `;`로 구분). 맞으면 비워 둔다.
4. `reviewer`에 이름을 쓴다. 이름이 없는 행은 미판정으로 남는다.
5. `final_verdict`는 비워 두면 규칙으로 정해진다.

```bat
python -m eval.review apply --run eval\results\<실행 폴더>
```

문항 종류(`kind`)별로 쓸 수 있는 error_types:

| kind | FAIL | PARTIAL |
| --- | --- | --- |
| answerable | wrong_value, wrong_unit, wrong_period, wrong_dims, unsupported_claim, false_refusal | missing_aux, no_source |
| not_answerable | missed_refusal, unsupported_claim, wrong_reason | insufficient_reason |
| clarify | assumed_condition, unsupported_claim | missing_clarification |
| default_allowed | wrong_value, wrong_unit, wrong_period, wrong_dims, disallowed_default, unsupported_claim | unstated_default, no_source |

- form에서 출처를 생략하라고 지시한 실행은 `no_source`를 붙이지 않는다(실험 조건이므로). 대신 `review_note`에 적어 둔다.

## 5. 점수 집계

```bat
python -m eval.scorer --run eval\results\<실행 폴더>
```

- PASS / PARTIAL / FAIL 수와 오류 유형별 건수를 출력하고 `summary.json`으로 저장한다.
- 미판정 행이 남아 있으면 `[잠정]`으로 표시된다.
- 기준 점수로 남길 실행만 `--append-history --change "<바꾼 것>"`을 붙여 `docs/reports/eval-history.md`에 한 줄 추가한다. mock 실행은 기록되지 않는다.

## 6. 프롬프트 실험을 비교할 때

- 한 번에 하나만 바꾼다(form 파일, 시스템 프롬프트 버전, 모델 중 하나).
- 같은 골든셋, 같은 표, 같은 `--interval`로 실행한다.
- 비교 기준: form 없이 DASH로 돌린 실행(`--stage B0-dash`).
- `--stage` 이름에 무엇을 바꿨는지 드러나게 쓴다. form 내용은 실행 폴더의 `form.txt`와 `manifest.json`의 sha256으로 구분한다.
