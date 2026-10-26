# 평가 리포트 — T1-25-oracle-all

- 실행: 2026-10-26T11:50:38+09:00 · 골든 golden (90문항, sha256 504a00eff236)
- 답변: oracle

| 지표 | 값 |
| --- | --- |
| 수치 정확도 (answerable + default_allowed) | 67/67 (100%) |
| 거절률 (not_answerable, 허용 사유) | 20/20 (100%) |
| 잘못된 거절 | 0 |
| 숫자 오류 건수 | 0 |
| 실행 실패 | 0 |
| 사람 확인 필요 | 23 |

| 분해 | PASS | PARTIAL | FAIL |
| --- | --- | --- | --- |
| kind=answerable | 62 | 0 | 0 |
| kind=not_answerable | 20 | 0 | 0 |
| kind=clarify | 3 | 0 | 0 |
| kind=default_allowed | 5 | 0 | 0 |
| type=compare | 18 | 0 | 0 |
| type=extreme | 5 | 0 | 0 |
| type=none | 3 | 0 | 0 |
| type=share | 10 | 0 | 0 |
| type=trend | 26 | 0 | 0 |
| type=value | 28 | 0 | 0 |
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
| op=AVG | 3 | 0 | 0 |
| op=GAP | 3 | 0 | 0 |
| op=GRW | 3 | 0 | 0 |
| op=MON | 2 | 0 | 0 |
| op=PAVG | 3 | 0 | 0 |
| op=PRD | 3 | 0 | 0 |
| op=REF | 4 | 0 | 0 |
| op=SHR | 3 | 0 | 0 |
| op=SUM | 2 | 0 | 0 |
| op=VSA | 2 | 0 | 0 |
