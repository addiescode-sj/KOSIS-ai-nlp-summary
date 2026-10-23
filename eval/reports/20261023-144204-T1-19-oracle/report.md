# 평가 리포트 — T1-19-oracle

- 실행: 2026-10-23T14:42:04+09:00 · 골든 golden (62문항, sha256 09830f2a0828)
- 답변: oracle

| 지표 | 값 |
| --- | --- |
| 수치 정확도 (answerable + default_allowed) | 42/42 (100%) |
| 거절률 (not_answerable, 허용 사유) | 17/17 (100%) |
| 잘못된 거절 | 0 |
| 숫자 오류 건수 | 0 |
| 실행 실패 | 0 |
| 사람 확인 필요 | 20 |

| 분해 | PASS | PARTIAL | FAIL |
| --- | --- | --- | --- |
| kind=answerable | 38 | 0 | 0 |
| kind=not_answerable | 17 | 0 | 0 |
| kind=clarify | 3 | 0 | 0 |
| kind=default_allowed | 4 | 0 | 0 |
| type=compare | 12 | 0 | 0 |
| type=extreme | 5 | 0 | 0 |
| type=none | 3 | 0 | 0 |
| type=share | 7 | 0 | 0 |
| type=trend | 8 | 0 | 0 |
| type=value | 27 | 0 | 0 |
| trap=asterisk_symbol | 3 | 0 | 0 |
| trap=census_5y_gap | 3 | 0 | 0 |
| trap=dash_symbol | 2 | 0 | 0 |
| trap=decimal_restore | 1 | 0 | 0 |
| trap=fake_paren_unit | 1 | 0 | 0 |
| trap=item_subtotal | 5 | 0 | 0 |
| trap=label_reuse | 4 | 0 | 0 |
| trap=latest_period | 2 | 0 | 0 |
| trap=mixed_units | 3 | 0 | 0 |
| trap=negative_value | 4 | 0 | 0 |
| trap=no_total_dim | 4 | 0 | 0 |
| trap=projection_year | 8 | 0 | 0 |
| trap=same_level_mixed | 3 | 0 | 0 |
| trap=structural_missing | 2 | 0 | 0 |
| trap=sub_annual_period | 4 | 0 | 0 |
| trap=unit_missing | 22 | 0 | 0 |
| trap=x_symbol | 1 | 0 | 0 |
