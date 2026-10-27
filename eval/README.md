# 골든셋 프롬프트 테스트 방법

골든셋 문항을 HCX에 보내 답변을 받고, 사람이 판정한 뒤 점수를 집계하는 순서를 정리한다.
`--form`으로 답변 형식 지시 파일을 바꿔 가며 프롬프트를 실험할 때도 이 순서를 그대로 쓴다.
(2026-10-23 기준, 튜닝 모델 HCX-DASH-002, 데이터는 Parquet)

## 1. 준비

| 항목 | 위치 | 비고 |
| --- | --- | --- |
| HCX 키 | `.env`의 `HCX_API_KEY=nv-...` | 커밋 금지 |
| 모델 설정 | `configs/model.yaml`의 `model` 블록 | 기본 `HCX-DASH-002`, temperature 0, max_tokens 1024 |
| 골든셋 | `eval/golden/stat_golden.jsonl` (40문항) | gitignore, 팀 공유본에서 받는다 |
| 표 파일 | `data/values/<기관>_<표>.parquet` + `data/tables.csv` | gitignore. 골든셋의 `table` 경로와 sha256이 맞아야 실행된다 |
| 답변 형식 지시 | 예: `eval/golden/form.txt` | 자주 바뀌므로 인자로 넘긴다 |

- 골든셋은 10/23에 Parquet 기준으로 바꿨다(`scripts/golden_to_parquet.py`, 원본은 `eval/golden/backup/stat_golden.cache-v1.jsonl`).
  문항마다 `table`(Parquet 경로), `table_sha256`(Parquet 파일 해시), `scope`(oracle 범위)가 있다.
- 데이터를 다시 빌드하거나 고치면 Parquet 해시가 바뀌어 실행이 멈춘다. 그때는 `python scripts/golden_to_parquet.py --rehash`로 해시만 갱신한다.
- 실행은 Anaconda Prompt에서 `conda activate kosis` 후 레포 루트에서 한다.

## 2. 프롬프트가 만들어지는 방식

문항마다 HCX에 아래 두 메시지를 보낸다.

1. **system**: `kosis_chat/llm/prompts.py`의 `SYSTEM_PROMPTS[prompt_version]` (현재 `baseline-v1`)
2. **user**:
   ```
   [통계표]
   <메타 헤더 + 이 질문의 범위만 자른 표 Markdown>

   [질문]
   <골든셋 question>

   [답변 형식]          ← --form을 줬을 때만 붙는다
   <form 파일 내용>
   ```

- Parquet 표는 통째로 넣으면 입력 한도를 넘기 때문에(DT_1JC1502는 15.9만 행), 문항마다 일부만 잘라 넣는다. `--scope`로 고른다(`kosis_chat/data/scope.py`).
  - `oracle`(기본): 골든셋 문항의 `scope`(항목·분류 코드·시점 범위)만. 지역 등 분류는 정답과 **같은 수준 전부**를 넣는다(예: 시도 17개+전국). "검색이 완벽하면 답을 잘 쓰는가"를 보는 상한선.
    문항별 범위와 글자 수는 `eval/golden/scope_report.md`에서 본다(가장 긴 문항이 약 9천 자).
  - `auto`: 질문 글만 보고 고른다. 연도("2019~2023년", "이후", "최근 N년"), 분류 이름(서울/서울특별시, 충북 등 줄임말, 여성→여자), 항목 이름을 찾고,
    분류 이름이 없으면 최상위 수준(전국+시도)만 넣는다. 그래도 3만 자가 넘으면 오래된 시점부터 뺀다. 실제 서비스·RAG와 비교하는 기준.
  - `full`: 표 전체. 작은 표에서만 쓴다.
  - 문항마다 실제로 보낸 범위(`scope`: 모드, 범위, 칸 수, 글자 수)가 `results.jsonl`에 남는다.
- `[통계표]` 직렬화는 `--serialization`(또는 `model.yaml`의 `serialization`)으로 고른다.
  - `long`(기본): 값 하나가 한 행 (`| 항목 | 분류… | 시점 | 값 | 단위 |`)
  - `wide`: (항목, 분류) 한 행에 시점을 열로 펼침. 단위가 하나면 표 위에 한 번만 쓰고, 시점이 20개를 넘으면 20개씩 나눠 쓴다. 글자 수가 약 17~30%로 준다.
- 단위가 없는 표(KOSIS가 `UNIT_NM`을 안 준 DT_1B8000F/H 등)는 항목명·분류명 괄호에서 단위를 찾고, 없으면 `단위 미상`으로 넣는다(`kosis_chat/data/loader.py` `infer_unit`).
- `--form` 파일은 UTF-8로 읽고 앞뒤 공백은 지운다. 파일 내용은 그대로 붙으므로 제목이나 접두어가 필요하면 파일 안에 쓴다.
- 시스템 프롬프트와 form 지시가 서로 다르면(예: 출처를 쓰라 / 생략하라) 모델이 둘 다 받는다.
  form으로 정한 형식을 계속 쓸 거라면 `prompts.py`에 새 버전을 추가하고 `model.yaml`의 `prompt_version`을 바꾼다.

## 3. 실행

```bat
python -m eval.run_eval --stage <이름> --golden eval/golden/stat_golden.jsonl --scope oracle --form eval/golden/form.txt
```

| 인자 | 뜻 |
| --- | --- |
| `--stage` | 결과 폴더 이름에 붙는 실험 이름. 예: `B0-dash`, `B0-dash-form-v2` |
| `--golden` | 골든셋 폴더 또는 .jsonl 파일 |
| `--interval N` | 문항 사이 N초 대기(기본 1초). HTTP 429(호출 한도)가 남으면 5초, 그래도 남으면 15초 |
| `--form PATH` | 답변 형식 지시 파일. 빼면 form 없이 실행 |
| `--scope oracle\|auto\|full` | 문항마다 넣을 표 범위(기본 `oracle`). 2장 참고 |
| `--serialization long\|wide` | 표 직렬화 방식. 빼면 `model.yaml` 값 |
| `--model` | 모델 바꾸기. 예: `HCX-007` |
| `--thinking`, `--max-tokens` | HCX-007 thinking 설정. 켜면 max-tokens를 5120 정도로 늘린다 |
| `--backend mock` | 키 없이 흐름만 확인 (사전 테스트, 점수 기록 안 됨) |

결과는 `eval/results/<시각>-<stage>/`에 저장된다 (gitignore).

| 파일 | 내용 |
| --- | --- |
| `results.jsonl` | 문항별 답변(`raw_answer`), 호출 상태(`call_status`), 오류, 판정 |
| `manifest.json` | 모델·프롬프트 버전·직렬화·범위 모드(`scope_mode`)·골든셋·데이터 해시, form 경로와 sha256 |
| `golden.jsonl` | 이번 실행에 쓴 골든셋 사본 |
| `form.txt` | `--form`을 줬을 때 그 시점의 form 파일 사본 |

### 실행 로그

실행하면 단계마다 터미널에 한 줄씩 찍히고, 같은 내용이 `logs/<시각>-<stage>.log`에 저장된다(gitignore). `--log-dir`로 폴더를 바꿀 수 있다.

```
10:42:35 INFO    [load 3/5] 101_DT_1JC1502.parquet: 158,985 cells | hash 0.01s, load 2.10s, prepare 0.80s
10:42:36 INFO    data load done: 5 tables in 3.40s
10:42:41 INFO    [call 1/40] S-VAL-03 ok 4.20s (attempts 1, scope 19 cells / 858 chars, tokens 900/45)
10:52:11 INFO    done: total 590.0s = data load 0.4s + API calls 540.0s + interval wait 39.0s (+ other)
```

- `attempts`가 1보다 크면 429·5xx 재시도가 있었던 것(재시도마다 WARNING 줄).
- 문항별 `latency_seconds`, `attempts`, `usage`(토큰)는 `results.jsonl`에, 전체 요약은 `manifest.json`의 `timing`에 남는다.

### 호출 실패 확인

호출이 실패한 문항은 자동으로 FAIL(`run_failure`)이 된다. 판정 전에 실패 원인을 먼저 본다.

```bat
python -c "import json;[print(r['question_id'],r['call_status'],(r['call_error'] or '')[:150]) for r in map(json.loads,open(r'eval\results\<실행 폴더>\results.jsonl',encoding='utf-8')) if r['call_status']!='ok']"
```

| 오류 | 원인과 대응 |
| --- | --- |
| `HTTP 429` (42901) | 호출 한도. 자동 재시도(5/15/30/60초) 후에도 실패한 것 → `--interval`을 늘린다 |
| `input_too_long` (40003 Context length exceeded) | 표가 모델 입력 한도보다 크다. `--scope full`로 큰 표(DT_1JC1502 등)를 넣으면 이렇게 된다 → `oracle`이나 `auto`로 실행 |
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
   - `quality_score`에 자연어 품질 0~2점을 적는다(답한 문항: 2 = 질문에 바로 답하고 읽기 쉬움, 1 = 어색하거나 설명 부족, 0 = 질문 일부를 건너뛰었거나 오해를 부름. 거절·되묻기 문항: 사유와 확인 가능한 범위를 분명히 말하면 2점). 비워 두면 정오 판정만 확정되고, 다음 `export` 때 그 행이 품질 칸만 비운 채 다시 나온다.
   - judge 점수는 시트에 보여 주지 않는다(블라인드). 사람 점수가 judge 점수에 끌리지 않아야 둘의 일치도가 의미가 있다.
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
- 자연어 품질(`quality`): 품질 점수가 있는 확정 행의 평균(채점 n / 확정 N)과 0점 문항. 실행 폴더에 `quality.jsonl`(judge 결과)이 있으면 `(question_id, run_index)`로 맞춰 사람-judge 일치율, 3×3 혼동표, 답한 문항 / 거절·되묻기별 일치율, 가중 κ를 `quality_agreement`에 함께 낸다. eval-history 표의 칸은 바꾸지 않았다.
- 미판정 행이 남아 있으면 `[잠정]`으로 표시된다.
- 기준 점수로 남길 실행만 `--append-history --change "<바꾼 것>"`을 붙여 `docs/reports/eval-history.md`에 한 줄 추가한다. mock 실행은 기록되지 않는다.

## 6. 프롬프트 실험을 비교할 때

- 한 번에 하나만 바꾼다(form 파일, 시스템 프롬프트 버전, 모델 중 하나).
- 같은 골든셋, 같은 표, 같은 `--scope`, 같은 `--interval`로 실행한다.
- 비교 기준: form 없이 DASH로 돌린 실행(`--stage B0-dash`).
- `--stage` 이름에 무엇을 바꿨는지 드러나게 쓴다. form 내용은 실행 폴더의 `form.txt`와 `manifest.json`의 sha256으로 구분한다.

## 7. 자동채점 (StatAnswer 출력)

`summarize_stat`처럼 `StatAnswer`(Pydantic)를 돌려주는 파이프라인은 사람 판정 없이 채점한다. 자유 텍스트 답(B0)은 위 4번의 사람 판정을 그대로 쓴다.

```bat
python -m eval.harness --suite stat --answers answers.jsonl --stage T1-12
python -m eval.harness --suite stat --oracle --stage oracle    # 골든 정답을 스스로 채점: 40/40 PASS여야 한다
python -m eval.harness --suite stat --golden eval/golden --oracle --stage oracle   # 폴더의 *.jsonl 전부 (엣지 문항 포함 62/62)
```

- `answers.jsonl`: 한 줄에 `{"question_id": ..., "answer": <StatAnswer JSON>}`를 쓴다. 답이 없는 문항은 `run_failure`가 된다.
- 골든은 `eval/golden.py`의 `GoldenItem`(Pydantic)으로 읽는다. 문항 종류별 필수 조건(예: not_answerable이면 `allowed_reasons`)을 검사한다. 또 `table_sha256`과 Parquet 해시를 맞춰 보고, `source_ref`가 있는 fact를 모두 Parquet 셀과 대조한다. 하나라도 어긋나면 채점하지 않는다. `scope`는 실행기 전용이라 채점에 쓰지 않는다.
- 규칙(`eval/grade.py`)은 사람 판정과 같은 `RULES`/`decide_verdict`를 쓴다. LLM은 쓰지 않는다.
  - 셀 fact는 `source_ref`와 값으로 맞춘다. 셀이 없을 때, 기간만 다르면 `wrong_period`, 나머지는 `wrong_dims`가 된다.
  - 계산 fact는 입력 셀(`source_ref` 집합)로 짝을 짓고, 값은 골든 `round` 자릿수의 반 단위까지 허용한다. fact id나 식을 다르게 써도 된다.
  - 숫자 오류(`unsupported_claim`)는 세 가지다. Parquet과 다른 셀 값, 다시 계산하면 안 맞는 계산 fact, 연결 문장(TextPart) 속 숫자(fact의 시점은 제외).
  - 거절은 `allowed_reasons` 안의 사유여야 PASS다. 되묻기 문항에 답하면 `assumed_condition`, 기본값을 밝히지 않으면 `unstated_default`가 된다.
  - 거절 문구가 충분한지, 되묻기 선택지가 좋은지는 코드로 판정하지 않는다. 리포트의 "사람 확인 필요"로 센다.
- 슬롯 정확도(T1-23, 질의 이해): 답의 `query`(StatQuery)를 골든 정답 셀에서 뽑은 슬롯과 비교한다. 판정(PASS/FAIL)에는 영향이 없는 별도 지표다.
  - 필드: 질문 유형, 항목, 분류(축별 라벨 집합), 기간(정답 셀 시점의 처음·끝 = 예측 시점의 처음·끝), 순위 축(최대최소만, `query.over`).
  - 표에서 라벨이 하나뿐인 축은 양쪽에서 뺀다. 합계 라벨(전국·계 등)은 비교 문항이 아닐 때만 뺀다("전국보다"의 전국은 채점).
  - 대상은 answerable·default_allowed. 실행 실패나 빈 query(예비 챗봇)는 모든 필드 틀림이다.
  - 한 축에 라벨이 여러 개 필요한데 비교 문항이 아니면 StatQuery가 담지 못한다. 리포트에 "슬롯 기록 한계"로 따로 적는다.
  - `grade_run`은 경로별(rules / llm_fill / fallback_chatbot) "모두 맞음" 비율도 적는다. oracle 실행은 query가 없어 슬롯 행을 출력하지 않는다.
- 결과는 `eval/reports/<시각>-<stage>/`에 저장된다. 지표(수치 정확도·거절률·숫자 오류 건수·kind/type/EDA 함정(`trap=`)별 분해)는 `report.md`/`report.json`에, 문항별 근거는 `grades.jsonl`(gitignore, 데이터 값 포함)에 남는다.

## 8. EDA 엣지 문항 (T1-19)

`docs/eda-full.md`의 함정을 겨냥한 문항 22개는 별도 파일 `eval/golden/stat_golden_edge.jsonl`에 있다(문항 id `S-EDG-*`). `--golden eval/golden`으로 폴더를 넘기면 기존 40문항과 함께 읽힌다.

- 정답 숫자는 손으로 쓰지 않는다. `eval/golden/edge_spec.json`에 질문과 셀 좌표(`Cell.ref()`)·계산식만 적고 스크립트가 값을 채운다.
  ```bat
  python scripts/build_golden_edge.py
  python scripts/golden_to_parquet.py --golden eval/golden/stat_golden_edge.jsonl --report eval/golden/scope_report_edge.md
  ```
  두 번째 명령이 oracle 범위(`scope`)를 채운다. 지역 외 차원(연령·성별·국적 등)은 `SCOPE_OVERRIDES`의 `S-EDG-*` 항목으로 좁혔다(안 좁히면 최대 104만 자).
- 새 셀 값의 웹 대조 행은 `eval/golden/verify_checklist_edge.md`에 미확인(`[ ]`)으로 생긴다.
- 문항마다 `eda_traps` 태그가 있다. 태그 정의와 기대 동작은 `plans/T1-19.md`의 함정 카탈로그를 본다.

## 9. 회귀 테스트 게이트 (T1-24)

수정(프롬프트·규칙·검증기·모델)을 할 때마다 같은 골든셋으로 다시 돌리고, 기준선과 **문항별로** 비교한다.
합계가 올라도 기존에 맞던 문항이 틀리게 바뀌었으면(Regression) 실패로 본다.

```
run_eval → grade_run → view_run → regress
```

```bash
uv run python -m eval.regress --run eval/results/<run_id>            # eval/baselines/stat.json과 비교
uv run python -m eval.regress --run eval/results/<run_id> --accept   # 이 실행을 기준선으로 고정 (커밋)
uv run python -m eval.regress --run eval/results/<새> --base eval/results/<이전>   # 로컬 실행 둘을 비교
```

- 판정 순위는 PASS > PARTIAL > FAIL이다. 순위가 내려가면 Regression, 올라가면 Improvement로 센다.
- 종료 코드: 0 = 회귀 없음, 1 = 회귀 있음, 2 = 비교 거부. 골든·scoring_version·채점기(`eval/grade.py`, `eval/scorer.py`) 해시 중 하나라도 다르면 비교를 거부한다.
  채점기가 바뀌면 판정이 달라져도 모델 회귀가 아니기 때문이다. 이때는 기준 실행을 `grade_run`으로 다시 채점한 뒤 `--accept`한다.
- 모델·temperature·프롬프트·scope·직렬화·데이터 해시가 다르면 리포트 맨 위에 "바뀐 평가 조건"으로 표시한다. 실험에서 바꾼 것이 하나뿐인지 여기서 확인한다.
- 회귀 문항마다 새로 생긴 오류와 원인 단계(1 판정 / 2 분석 / 3 계산 / 4-5 문장·검증 / 실행), 경로(route/compose) 변화를 적는다. 거절 판단 회귀(false_refusal·missed_refusal)는 리포트 위쪽에 따로 모아 보여 준다.
- 결과 파일: `<run>/grades/<최신>/regress.md`, `regress.json`
- `eval/baselines/stat.json`에는 문항별 판정·오류 유형·경로·평가 조건만 들어 있다. 셀 값과 답변 문장은 넣지 않으므로 커밋한다. 기준선은 회귀를 확인한 뒤 의도적으로 올릴 때만 `--accept`한다.
## 10. 복합 통계 문항 (T1-25)

계산·파싱 모듈 고도화용 문항 28개가 `eval/golden/stat_golden_complex.jsonl`에 있다(n년 평균·누적합·두 기간 평균 비교·격차 변화·증감률 비교·구성비 변화·평균 대비·월 합산·어려운 기간 표현·거절).

- 문항 id에 연산이 들어 있다(`S-CX-AVG-01`). 하네스 리포트는 이 접두어로 `op=AVG` 같은 분해 행을 만든다.
- 다시 만들기: spec(`eval/golden/complex_spec.json`, 셀 ref와 derivation만)에서
  ```bash
  uv run python scripts/build_golden_edge.py --spec eval/golden/complex_spec.json --out eval/golden/stat_golden_complex.jsonl --checklist eval/golden/verify_checklist_complex.md
  uv run python scripts/golden_to_parquet.py --golden eval/golden/stat_golden_complex.jsonl --report eval/golden/scope_report_complex.md
  ```
- 고도화 전후 측정: `run_eval --golden eval/golden/stat_golden_complex.jsonl` → `grade_run` → 리포트의 `op=` 행 비교. 고도화 전 기준선(mock)은 PASS 1/28이다(`plans/T1-25.md`).
- 연속 증감·처음 기준을 넘은 해·전년 대비 증감 최대는 넣지 않았다. 지금 채점기로는 정답 연도·횟수를 채점할 수 없기 때문이다.

## 11. raw 모드 (`--pipeline raw`, 모델 단독 출력 보기)

규칙·분석·계산·형식 검사·재요청·출처 정리를 모두 빼고, 프롬프트 `raw-v2` + 표 데이터 + 질문만 모델에 한 번 보낸다(`kosis_chat/raw.py`, `kosis_chat/llm/raw_prompt.py`).
모델은 자유 문장으로 답하고(답변/거절/되묻기도 모델이 정함), 응답은 고치거나 검사하지 않고 그대로 `raw_answer`에 남는다. **채점하지 않는다.** 사람이 읽고 파이프라인 답과 나란히 본다.

```bash
python -m eval.run_eval --stage raw-dash --backend hcx --model HCX-DASH-002 --pipeline raw --scope oracle --serialization wide --interval 5
python -m eval.view_raw --run eval/results/<raw_run>                       # <raw_run>/raw_view.html
python -m eval.compare_runs --raw_path eval/results/<raw_run> --pipe_path eval/results/<파이프라인 실행>   # <raw_run>/compare_<파이프라인 실행>.html
```

- 표는 챗봇과 같은 방식으로 자르고(`--scope`) 직렬화한다(`--serialization`). 문항마다 모델에 보낸 표 글자의 해시가 결과 행 `scope.table_text_sha256`에 남는다. 같은 조건으로 모델만 바꿔 돌리면 두 raw 실행의 입력이 같은지 이 해시로 확인할 수 있다.
- `--answer-format text --judge off`를 자동으로 쓰고, `--prompt-version`은 쓰지 않는다. `--form`은 거부한다.
- `view_raw`: 문항마다 질문, 모델 답 원문, 시간·토큰, 모델에 보낸 입력(표 + 질문, 접힘)을 보여 준다. 그래프·채점 없음, 인터넷 불필요.
- `compare_runs`: 먼저 두 실행의 `golden.jsonl` 질문(id와 질문 글)이 같은지 확인하고, 다르면 차이를 출력하고 멈춘다(종료 코드 2). 같으면 문항마다 왼쪽에 AI 답(raw), 오른쪽에 파이프라인 답(`stat_answer.text`, 텍스트 챗봇이면 `raw_answer`)과 결과(answered/refused/clarify), 경로, 사실 목록(접힘)을 놓는다. 중간에 멈춘 실행의 빠진 문항은 "실행 안 됨"으로 표시한다.
- `grade_run`은 raw 실행을 채점하지 않고 종료 코드 2로 끝난다. `view_run`도 쓸 수 있지만 판정은 모두 `-`이다.

## 자연어 품질 자동 평가 (T1-26)

답변 문장이 읽기 쉬운지를 0~2점으로 매긴다. 숫자가 맞는지는 7장의 채점기가 보므로 여기서는 문장만 본다.

```
run_eval → grade_run → judge (선택) → view_run → regress
```

```bash
uv run python -m eval.judge --run eval/results/<run_id>                                   # HCX-007, 호출 간격 5초
uv run python -m eval.judge --run eval/results/<run_id> --labels eval/quality_labels.jsonl  # 보정 라벨 검출률까지
```

- **코드 표기 검사**(결정적, LLM 없음): 단위 중복(`844명/㎢ 명/㎢`), 연월 코드 노출(`202412년`), 숫자+단위 뒤 조사 불일치(`가구과`, 단위 끝 글자의 받침으로 판정), 내부 라벨(`일반가구_계`), 미완성 조사 템플릿(`와(과)`·`은(는)`). 출처 줄은 검사하지 않는다.
- **LLM judge**(HCX-007, temperature 0): 코드가 못 보는 의미 문제만 본다. 태그는 `ignored_part`(질문 일부 무시), `unsupported_judgement`(표에 없는 평가), `missing_detail`(요구 정보 누락), `awkward_label`(라벨을 그대로 옮긴 어색한 표현), `unclear_refusal`, `other`.
- 점수: 답한 문항은 2 = 바로 답하고 읽기 쉬움, 1 = 어색하거나 설명 부족, 0 = 질문 일부를 건너뛰었거나 오해를 부름. 거절·되묻기 문항은 사유와 확인 가능한 범위를 분명히 말하면 2점이다. 코드 표기 검사에 걸리면 최대 1점이다.
- judge 호출이 실패한 문항은 점수 없음으로 두고 평균에서 뺀다. 같은 (루브릭 버전, judge 모델, 질문, 답변)은 `data/cache/judge.jsonl`에 캐시해 다시 부르지 않는다.
- judge 점수는 비결정적이라 회귀 게이트에 넣지 않는다. 주 용도는 **파이프라인 전→후 비교**다. 답변 모델과 judge 모델이 같으면 리포트에 자기 선호 경고를 적는다(예: AI 단독 답을 HCX-007로 만든 경우).
- 결과: `<run>/quality.md`, `quality.json`, `quality.jsonl`(문항별 점수·태그·표기 문제·이유).
- `eval/quality_labels.jsonl`: 보정 라벨. `source: human`은 사람이 판정한 사례(10/26 Notion), `candidate`는 Claude가 찾은 표기 후보로 사람 확인 전이다. 리포트는 둘을 나눠 검출률을 적는다. T1-21 사람 점수가 나오면 judge 점수와 일치도를 잰다.
