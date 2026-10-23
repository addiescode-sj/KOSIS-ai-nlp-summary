# 질문별 범위 모드와 골든셋 Parquet 전환 (2026-10-23)

데이터를 Parquet 하나로 통일하고(형식 A CSV·형식 B KOSIS JSON·`.meta.json` 삭제), 큰 표를 질문마다 잘라 넣는 범위 모드를 추가했다.
골든셋도 Parquet 경로·해시와 문항별 범위(`scope`)로 바꿨다.

## 1. 왜

- Parquet에는 표 전체가 들어 있다. DT_1JC1502는 15.9만 행(읍면동 4,159개)이라 넓은 형식으로도 약 265만 자이고, DASH 입력 한도(약 3만 자)를 크게 넘는다.
- 예전에 eval이 돌던 건 골든셋이 시도 21개만 받은 API 캐시(1,806행)를 썼기 때문이다.
- 그래서 질문마다 필요한 부분만 넣는다. 자르는 방법을 두 가지로 두고 비교한다.

## 2. 범위 모드 (`kosis_chat/data/scope.py`)

| 모드 | 무엇을 넣나 | 쓰임 |
| --- | --- | --- |
| `oracle` | 골든셋 문항의 `scope`(항목 코드, 분류 코드, 시점 범위) | 검색이 완벽할 때의 상한선. eval 기본값 |
| `auto` | 질문 글에서 찾은 연도·분류 이름·항목 이름. 분류 이름이 없으면 최상위 수준(전국+시도). 3만 자가 넘으면 오래된 시점부터 뺌 | 실제 서비스·RAG 비교 기준. cli 기본값 |
| `full` | 표 전체 | 작은 표, 비교용 |

- 분류 코드 수준: KOSIS 숫자 코드는 길이로 겹친다(시도 `11` > 시군구 `11010` > 읍면동 `11010530`). 같은 수준은 길이가 같고 부모 코드가 같은 것.
  숫자가 아닌 코드(`S1`, `A10`)는 한 수준으로 본다.
- auto 규칙: 연도(`2019~2023년`, `2015년부터 2020년까지`, `1970년 이후`, `최근 N년`, `최근`=최근 5개 시점), `1985년생`은 시점이 아니라 분류로 본다.
  지역 줄임말(서울, 충북, 제주도, 세종시)과 여성→여자, 남성→남자. 비중·비율 질문은 합계 항목(…계)을 같이 넣는다.
  표에 없는 연도를 물으면 시점을 자르지 않아 모델이 "없다"고 답할 수 있게 한다. 읍면동처럼 아래 수준 이름은 3글자 이상만 찾는다("더하면"에 `하면`면이 걸리는 문제).
- eval 결과(`results.jsonl`)에 문항마다 실제로 보낸 범위·칸 수·글자 수가 `scope`로 남고, `manifest.json`에 `scope_mode`가 남는다.

## 3. 골든셋 전환 (`scripts/golden_to_parquet.py`)

- `table`: `data/cache/101_DT_X_<해시>.json` → `data/values/101_DT_X.parquet` (표 6개 → 5개, DT_1JC1502 두 캐시가 한 파일로)
- `table_sha256`: Parquet 파일 sha256. 데이터를 다시 만들면 바뀌므로 `--rehash`로 갱신한다.
- `scope`: 정답 근거(source_ref)의 항목만, 분류는 근거와 **같은 수준 전부**(사용자 결정), 시점은 근거 시점과 질문 연도를 덮는 범위.
  default_allowed는 최근 5개 시점까지. 근거가 없는 12문항과 `S-EXT-01`은 스크립트의 `SCOPE_OVERRIDES`에 직접 적었다.
- 원본은 `eval/golden/backup/stat_golden.cache-v1.jsonl`, 문항별 검토표는 `eval/golden/scope_report.md` (둘 다 gitignore).
- 사전 확인: 골든셋 정답 근거 67개 모두 Parquet에서 같은 값으로 찾음. not_answerable 근거(빈칸, X, 세종 2012년부터, 기대수명 2025 '-')도 그대로.

### 측정 (넓은 형식 글자 수)

| | oracle | auto |
| --- | --- | --- |
| 가장 긴 문항 | S-EXT-01 9,432자 (8000F 1970~2025 전체) | S-EXT-04 1,831자 |
| 대부분 | 800~3,300자 | 350~1,700자 |
| 정답 근거가 범위 안에 있는 문항 (근거 있는 28문항) | 28 (만든 방식상 전부) | 26 |

- auto가 놓친 2문항: `S-SHR-03`("3인 이하"에 1인·2인이 안 걸림), `S-SHR-04`("5인 이상"에 6인 이상이 안 걸림).
  "N인 이하/이상" 같은 범위 표현은 auto 규칙으로 풀지 않았다. RAG 시험 때 비교할 대목.
  (`S-VAL-08` "다 더하면"은 처음에 읍면동 `하면`면이 걸려 놓쳤고, 아래 수준 이름을 3글자 이상만 찾도록 고쳐 통과.)

## 4. 지운 것

- 로더 형식 A(CSV + `.meta.json`), 형식 B(KOSIS JSON + `.meta.json`), Parquet의 `.meta.json` 분기. `TableMeta.input_format`은 `parquet` 하나.
- `tests/fixtures/` 10개 파일 → `tests/conftest.py`가 같은 내용의 가짜 Parquet 두 개(`FAKE_TBL_001` 성별 인구, `FAKE_TBL_002` 지역 계층 가구)와 `tables.csv`를 만든다.

## 5. data 폴더 정리 (사용자 PC, gitignore)

| 전 | 후 |
| --- | --- |
| `data/build_log.txt`, `data/done.jsonl`, `data/zip_item_map.csv` | `data/build/` |
| `data/population_table_catalog.jsonl`, `data/table_coverage.csv` | `data/reference/` |
| `data/golden.jsonl` (예전 골든셋 사본) | 삭제 |
| `data/cache/` (API 캐시 JSON·`.meta.json`, kosis/ 1,851개) | 삭제 |
| `data_old/` (10/21 이전 데이터) | 삭제 |
| `eval/golden/b0-loadable/` (캐시 기준 22문항 부분셋) | 삭제 |

남는 구조: `data/tables.csv`, `data/values/`(2,364표), `data/table_metadata/`, `data/build/`, `data/reference/`. `data_fix/`(받은 원본)는 Parquet을 다시 만들 때 필요해서 남겼다.

## 6. 실행

```bat
python -m pytest -q
python scripts/golden_to_parquet.py
python -m eval.run_eval --stage B0-oracle --scope oracle --serialization wide
python -m eval.run_eval --stage B0-auto --scope auto --serialization wide
python -m kosis_chat.cli --table data/values/101_DT_1JC1502.parquet -q "2023년 제주도 1인 가구는?"
```
