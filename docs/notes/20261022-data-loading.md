# 데이터 불러오기 개선 (2026-10-22)

> 10/23 후속: 형식 A/B와 `.meta.json`은 삭제되고 Parquet만 남았다. 골든셋도 Parquet으로 바뀌었다 → [20261023-scope-modes.md](20261023-scope-modes.md)

골든셋 40문항을 더 많이 돌리기 위해 데이터 불러오기 방식을 세 가지 바꿨다. 이 문서는 바꾼 내용, 그 근거가 된 예시, 확인 방법, 보류한 일을 정리한다.

- 브랜치: `prompt-v2-source` (master 34bcf1e 위, **미커밋**)
- 상태: 구현 완료, 사용자 PC 확인 대기

## 1. 왜 바꿨나: 골든셋이 다 돌지 않던 이유

골든셋(`eval/golden/stat_golden.jsonl`) 40문항, 표 6개 기준.

| 문항 수 | 표 | 막힌 이유 |
| --- | --- | --- |
| 12 | DT_101094A001, DT_1B08024, DT_1JC1502(944행 캐시) | 정상 실행 |
| 9 | DT_1JC1502 (`101_DT_1JC1502_c35fdfd8da.json`, 1,806행) | 입력 길이 초과. 직렬화 약 6.5만 자 → HCX-DASH-002가 40003 Context length exceeded |
| 19 | DT_1B8000H(11), DT_1B8000F(8) | 길이와 무관. KOSIS 응답에 `UNIT_NM`이 없어 로더가 표 전체를 거부 → 모델에 가지도 못함 |

(앞서 "18문항"이라고 한 것은 착오. 지금 파일 기준 19문항: 답변 가능 10, 답변 불가 6, 기본값 2, 되묻기 1.)

### 예시: DT_1JC1502가 1,806행인 이유

골든셋 캐시는 표 전체가 아니라 팀원이 KOSIS API에서 **시도 단위만 골라 받은** 파일이다.

- 지역 21개: 전국, 동부·읍부·면부, 17개 시도 (지역 코드가 모두 2자리, 시군구·읍면동 없음)
- 시점 11개: 2015~2025
- 항목 9개: 일반가구_계, 1인~7인 이상 등 (단위: 가구)
- 21 × 11 × 항목 ≈ 1,806행 (연도에 따라 빠진 항목이 있어 딱 떨어지지 않음)
- 같은 표의 다른 캐시 `101_DT_1JC1502_6827bccc6d.json`(944행, 1문항)은 광주광역시 구·동 단위만 받은 파일
- Parquet(`data/values/101_DT_1JC1502.parquet`)는 전국 읍면동 전체라 158,985행

## 2. 바꾼 내용

### 1) Parquet 로더 (형식 C)

- `load_table("data/values/<org>_<tbl>.parquet")`가 동작한다. `input_format = "C"`.
- 메타는 옆의 `<name>.meta.json`이 있으면 그것, 없으면 `data/tables.csv`의 `key` 행에서 채운다(조사명·통계표명·작성기관·URL). `tables.csv`에 받은 날짜가 없어 `fetched_at`은 Parquet 파일 날짜를 쓴다.
- 셀 출처(`source_ref`)는 API 표와 같은 규칙: `<tbl_id>|<item_id>|<분류 코드>|<시점>`.
- `requirements.txt`에 `pyarrow` 추가.
- 골든셋은 아직 `data/cache/*.json`을 가리킨다(바꾸지 않음). 이유는 5절.

### 2) 단위 보정 (거부 대신 추정)

단위를 이 순서로 찾는다(`kosis_chat/data/loader.py` `infer_unit`). 어디서 찾았는지는 셀의 `unit_source`에 남는다.

1. `UNIT_NM`(또는 Parquet `unit`)이 있으면 그대로 → `given`
2. 항목명 괄호 → `item_name`
3. 분류명 괄호 → `dim_label`
4. 수동 보정표 → `override`
5. 그래도 없으면 `단위 미상` → `unknown` (표를 거부하지 않음)

괄호 안이 아는 단위(`명, 건, 가구, 호, 개, 세, 년, %, 원, 명/㎢` 등)이거나 `…당`(천명당, 출생아 천명당)일 때만 단위로 본다.

예시 (실제 골든셋 표):

| 이름 | 찾은 단위 | 출처 |
| --- | --- | --- |
| `출생건수 (명)` (8000H 항목명) | 명 | item_name |
| `조사망률(천명당)` (8000F 분류명) | 천명당 | dim_label |
| `영아사망률(출생아 천명당)` | 출생아 천명당 | dim_label |
| `기대수명(출생시 기대여명) - 남` | 년 | override (괄호가 단위가 아님) |
| `합계출산율` (8000H, 괄호 없음) | 명 | override |

결과: 두 표 모두 단위 미상 0셀. 8000F 865셀(dim_label 697, override 168), 8000H 5,208셀(item_name 4,740, override 468).

### 3) 넓은 형식 직렬화 (`--serialization wide`)

- 지금(long): 값 하나가 한 행. 항목·지역·단위가 매 행 반복.
- wide: (항목, 분류) 한 행에 시점을 열로 펼친다. 단위가 하나면 표 위에 `단위: 가구`처럼 한 번만, 여러 개면 단위 열을 둔다. 시점이 20개를 넘으면 20개씩 블록으로 나눈다(열이 많으면 모델이 칸을 잘못 읽기 쉬움).
- 기본값은 그대로 `long`(`configs/model.yaml` `serialization: long`). 결과 `manifest.json`에 `serialization`이 남는다.

예시 (DT_1JC1502 캐시):

```
단위: 가구
| 항목 | 행정구역별(읍면동) | 2015 | 2016 | … | 2025 |
| 일반가구_계 | 전국 | 19111030 | 19367696 | … | 22495096 |
| 1인 | 전국 | 5203440 | 5397615 | … | 8244308 |
```

글자 수 (메타 헤더 포함, 실제 코드로 측정):

| 표 (골든셋 캐시) | 문항 | long | wide |
| --- | --- | --- | --- |
| DT_1JC1502 1,806행 | 9 | 65,184자 | 19,634자 |
| DT_1JC1502 944행 | 1 | 30,760자 | 15,273자 |
| DT_1B8000H 5,208행 | 11 | 210,421자 | 47,634자 |
| DT_1B8000F 865행 | 8 | 45,404자 | 9,421자 |
| DT_1B08024 339행 | - | 13,323자 | 4,248자 |
| DT_101094A001 351행 | - | 13,132자 | 3,434자 |

지금까지 3.1만 자는 성공, 6.5만 자는 실패. DT_1JC1502(1.96만 자)는 들어갈 가능성이 높고, DT_1B8000H(4.76만 자)는 한도 사이라 확실하지 않다.

## 3. 확인한 것 (클라우드, PC 레포 코드 사본)

- 테스트: 기존 58개 + 새 14개(`tests/test_loading.py`) = **72 passed** (pyarrow 설치 환경).
- Parquet 로드: 골든셋 표 5개 모두 형식 C로 로드됨.

| Parquet | 행 | long | wide | 로드 시간 |
| --- | --- | --- | --- | --- |
| 101_DT_101094A001 | 351 | 13,140자 | 3,442자 | 즉시 |
| 101_DT_1B08024 | 339 | 13,331자 | 4,256자 | 즉시 |
| 101_DT_1B8000F | 865 | 45,415자 | 9,432자 | 0.1초 |
| 101_DT_1B8000H | 7,018 | 283,931자 | 61,159자 | 0.2초 |
| 101_DT_1JC1502 | 158,985 | 656만 자 | 265만 자 | 5.8초 |

## 4. 사용자 PC에서 확인할 것 (conda env kosis, 레포 루트)

```bat
pip install pyarrow
pytest -q
python -m kosis_chat.cli --table data/cache/101_DT_1B8000F_9b1fc63bd2.json --backend mock --serialization wide -q x
python -m eval.run_eval --stage B0-wide-all --golden eval/golden/stat_golden.jsonl --interval 5 --serialization wide
```

- 두 번째: 72 passed 기대.
- 세 번째: 단위 없던 표가 거부 없이 로드되는지(mock이라 API 호출 없음).
- 네 번째: 40문항 전체. 기대: 8000F 8문항·DT_1JC1502 10문항이 실행됨, 8000H 11문항은 길이 초과(`input_too_long`)가 날 수 있음. 비교 기준은 기존 b0-loadable 결과(long).

## 5. 남은 한계

- DT_1B8000H: wide로도 4.76만 자. 실패하면 행 필터(보류 항목 4)가 필요.
- Parquet 원본은 표 전체라 골든셋을 Parquet로 옮기면 DT_1JC1502가 265만 자가 된다. 골든셋 전환은 질문별 범위(보류 항목 4)와 같이 해야 한다.
- ZIP 출처 Parquet 값은 `19111030.0000` 형식 그대로 나간다(보류 항목 5).
- `.gitignore`의 `data/` 규칙이 `kosis_chat/data/`도 무시한다. 이번에는 그 폴더에 새 파일을 만들지 않아 영향 없음(기존 파일만 수정).

## 6. 보류 (나중에 리마인드)

- **4. 질문별 범위 두 모드**: 정답 범위 모드(골든셋 정답의 시점·분류로 잘라 넣기 → 모델 실력만 측정) vs 자동 필터 모드(질문에서 조건 추출 → 필터 품질까지 측정). DT_1B8000H, Parquet 전환에 필요.
- **5. 값 정리**: Parquet `value_kind`(ok, dash, blank, x_secret) → 결측·비공개 표시 매핑, ZIP 값 `.0000` 정리.

## 바뀐 파일

| 파일 | 내용 |
| --- | --- |
| `kosis_chat/data/loader.py` | `infer_unit`, 형식 B에서 `UNIT_NM` 필수 해제, 형식 C(Parquet) 로더 |
| `kosis_chat/data/schema.py` | `INPUT_FORMATS`에 C, `Cell.unit_source` |
| `kosis_chat/data/serializer.py` | `serialize(style)`, `to_wide_markdown` |
| `kosis_chat/config.py`, `configs/model.yaml` | `serialization` 설정(기본 long) |
| `kosis_chat/chatbot.py` | 설정한 직렬화 사용 |
| `kosis_chat/cli.py`, `eval/run_eval.py` | `--serialization`, manifest 기록, `--table` 도움말 |
| `requirements.txt` | pyarrow |
| `eval/README.md` | 직렬화·단위 보정 설명 |
| `tests/test_loading.py` | 새 테스트 14개 |
