[예 1] 질문: 2010년보다 2020년에 A도와 B도의 사망자 수 차이가 커졌어?
{"steps": [{"id": "x1", "op": "select", "item": {"codes": ["T1"]}, "dims": {"지역": {"codes": ["A"]}}, "periods": {"kind": "list", "periods": ["2010"]}},
 {"id": "y1", "op": "select", "item": {"codes": ["T1"]}, "dims": {"지역": {"codes": ["B"]}}, "periods": {"kind": "list", "periods": ["2010"]}},
 {"id": "x2", "op": "select", "item": {"codes": ["T1"]}, "dims": {"지역": {"codes": ["A"]}}, "periods": {"kind": "list", "periods": ["2020"]}},
 {"id": "y2", "op": "select", "item": {"codes": ["T1"]}, "dims": {"지역": {"codes": ["B"]}}, "periods": {"kind": "list", "periods": ["2020"]}},
 {"id": "g1", "op": "combine", "a": "x1", "b": "y1", "fn": "diff", "orient": "larger_first", "note": "2010년 격차"},
 {"id": "g2", "op": "combine", "a": "x2", "b": "y2", "fn": "diff", "orient": "larger_first", "note": "2020년 격차"},
 {"id": "d", "op": "combine", "a": "g2", "b": "g1", "fn": "diff", "note": "격차 변화"}],
 "show": ["x1", "y1", "x2", "y2", "g1", "g2", "d"], "present": "compare"}
[예 2] 질문: 2023년 전체 출생아 중 3월 출생아의 비중은? (월 표, 연 값 있음)
{"steps": [{"id": "p", "op": "select", "item": {"codes": ["T1"]}, "periods": {"kind": "list", "periods": ["202303"]}},
 {"id": "t", "op": "select", "item": {"codes": ["T1"]}, "periods": {"kind": "list", "periods": ["2023"]}},
 {"id": "s", "op": "combine", "a": "p", "b": "t", "fn": "share"}],
 "show": ["p", "t", "s"], "present": "share"}
