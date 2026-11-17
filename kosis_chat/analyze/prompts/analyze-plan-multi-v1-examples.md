[예 1] 표 A=출생아 수(명), 표 B=연앙인구(천명). 질문: 2024년 A도의 인구 1,000명당 출생아는?
{"steps": [{"id": "b", "op": "select", "table": "A", "item": {"codes": ["T1"]}, "dims": {"지역": {"codes": ["A"]}}, "periods": {"kind": "list", "periods": ["2024"]}},
 {"id": "p", "op": "select", "table": "B", "item": {"codes": ["T2"]}, "dims": {"지역": {"codes": ["A"]}}, "periods": {"kind": "list", "periods": ["2024"]}},
 {"id": "r", "op": "combine", "a": "b", "b": "p", "fn": "per", "scale": 1000, "note": "인구 1,000명당 출생아"}],
 "show": ["b", "p", "r"], "present": "value"}
[예 2] 표 A=출생아 수, 표 B=사망자 수. 질문: 2020년과 2024년 자연증가는 얼마나 달라졌어?
{"steps": [{"id": "b1", "op": "select", "table": "A", "item": {"codes": ["T1"]}, "periods": {"kind": "list", "periods": ["2020"]}},
 {"id": "d1", "op": "select", "table": "B", "item": {"codes": ["T1"]}, "periods": {"kind": "list", "periods": ["2020"]}},
 {"id": "b2", "op": "select", "table": "A", "item": {"codes": ["T1"]}, "periods": {"kind": "list", "periods": ["2024"]}},
 {"id": "d2", "op": "select", "table": "B", "item": {"codes": ["T1"]}, "periods": {"kind": "list", "periods": ["2024"]}},
 {"id": "n1", "op": "combine", "a": "b1", "b": "d1", "fn": "diff", "note": "2020년 자연증가"},
 {"id": "n2", "op": "combine", "a": "b2", "b": "d2", "fn": "diff", "note": "2024년 자연증가"},
 {"id": "g", "op": "combine", "a": "n2", "b": "n1", "fn": "diff", "note": "자연증가 변화"}],
 "show": ["b1", "d1", "n1", "b2", "d2", "n2", "g"], "present": "trend"}
