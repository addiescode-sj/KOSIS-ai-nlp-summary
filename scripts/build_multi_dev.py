"""Build the multi-table dev set (T1-30, D6): questions over two or three related tables, answers from the Parquet.

    python scripts/build_multi_dev.py            # KOSIS_WORKSPACE (or this folder) holds data/tables.csv + values/
    python scripts/build_multi_dev.py --out eval/dev/multi_table_dev.jsonl

Every expected number is read here with pandas from the Parquet rows (not through kosis_chat), and the final number
is computed here from those cells. Each row also carries a gold plan (v3 blocks, tables named A·B·C) for the oracle
run of eval/multi_dev.py, which checks that the pipeline computes what this script computed. The questions were
written for development, not taken from any golden set; they cover the six relations of the human golden guide
(분자·분모, 짝 현상, 같은 대상 두 출처, 옛 표 + 새 표, 추계 vs 실적, 세 표) plus refusals.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "eval" / "dev" / "multi_table_dev.jsonl"

MARRIAGE, DIVORCE = "101_DT_1B83A35", "101_DT_1B85033"  # 시도/시군구/월별 혼인, 이혼 (건; 연 값은 YYYY 시점)
MARRIAGE_JOB, DIVORCE_JOB = "101_DT_1B83A32", "101_DT_1B85029"  # 시도/직업별 혼인·이혼 (남편 기준, 건)
BIRTH_SIDO = "101_DT_1B81A02"  # 시도/출생장소별 출생 (명)
POP_MID = "101_DT_1B040M5_1"  # 시군구/성/연령별 주민등록연앙인구 (명, 2023~)
BIRTH, DEATH = "101_DT_1B80A01", "101_DT_1B80A13"  # 전국 출생 (명), 전국 사망 (명)
PROJ = "101_DT_1BPA004"  # 장래 인구변동요인 (천명), 가정 1 = 중위
FOREIGN_2010, FOREIGN_2015 = "101_DT_1JA1002", "101_DT_1JA1502"  # 외국인 (명): 2010 총조사, 2015~ 등록센서스

SIDO = {"00": "전국", "11": "서울", "21": "부산", "22": "대구", "29": "세종", "31": "경기"}

# How to read one cell of each table: item code, then {dim column: code} (the other dims are their totals).
CELL = {
    MARRIAGE: lambda region: ("T3", {"dim1": region}),
    DIVORCE: lambda region: ("T4", {"dim1": region}),
    MARRIAGE_JOB: lambda region: ("T1", {"dim1": region, "dim2": "00"}),
    DIVORCE_JOB: lambda region: ("T1", {"dim1": region, "dim2": "00"}),
    BIRTH_SIDO: lambda region: ("T1", {"dim1": region}),
    POP_MID: lambda region: ("T10", {"dim1": region, "dim2": "0", "dim3": "000"}),
    BIRTH: lambda region: ("T1", {"dim1": "0", "dim2": "00", "dim3": "00"}),
    DEATH: lambda region: ("T2", {"dim1": "0", "dim2": "000"}),
    PROJ: lambda what: ("T10", {"dim1": "1", "dim2": what}),
    FOREIGN_2010: lambda region: ("T10", {"dim1": region, "dim2": "0"}),
    FOREIGN_2015: lambda region: ("T10", {"dim1": region, "dim2": "0"}),
}


class Data:
    def __init__(self, data_dir: Path) -> None:
        self.data_dir = data_dir
        self.frames: dict = {}

    def frame(self, key: str):
        if key not in self.frames:
            self.frames[key] = pq.read_table(self.data_dir / "values" / f"{key}.parquet").to_pandas()
        return self.frames[key]

    def axes(self, key: str) -> dict[str, str]:
        """dim column ('dim1') -> category name, as the pipeline names the dims."""
        df = self.frame(key)
        return {f"dim{i}": df[f"dim{i}_name"].dropna().iloc[0] for i in range(1, 9) if df[f"dim{i}_name"].notna().any()}

    def cell(self, key: str, item: str, dims: dict[str, str], period: str) -> dict:
        df = self.frame(key)
        mask = (df.item_id == item) & (df.period == period)
        for dim, code in dims.items():
            mask &= df[f"{dim}_code"] == code
        rows = df[mask]
        if len(rows) != 1:
            raise LookupError(f"{key} {item} {dims} {period}: {len(rows)} rows")
        row = rows.iloc[0]
        return {"value": float(row["value"]) if row["value"] == row["value"] else None, "raw": row["value_raw"]}


def select(step_id: str, label: str, key: str, arg: str, period: str, data: Data) -> dict:
    item, dims = CELL[key](arg)
    names = data.axes(key)
    return {"id": step_id, "op": "select", "table": label, "item": {"codes": [item]},
            "dims": {names[d]: {"codes": [c]} for d, c in dims.items()}, "periods": {"kind": "list", "periods": [period]}}


def read(data: Data, key: str, arg: str, period: str) -> float:
    item, dims = CELL[key](arg)
    return data.cell(key, item, dims, period)["value"]


def q(qid, relation, question, tables, cells, combine, final, rnd, present="compare", kind="answer", refusal=None):
    """cells: [(step id, table index, arg, period)]; combine: plan combine steps; final(values) -> the answer."""
    return dict(qid=qid, relation=relation, question=question, tables=tables, cells=cells, combine=combine,
                final=final, round=rnd, present=present, kind=kind, refusal=refusal)


def per(a, b, scale=1000, sid="r", note=None):
    return {"id": sid, "op": "combine", "a": a, "b": b, "fn": "per", "scale": scale, **({"note": note} if note else {})}


def comb(sid, a, b, fn, **kw):
    return {"id": sid, "op": "combine", "a": a, "b": b, "fn": fn, **kw}


QUESTIONS = [
    # 짝 현상: 혼인과 이혼 (건 = 건)
    q("M01", "짝 현상", "2024년 부산에서 혼인 건수가 이혼 건수보다 몇 건 많았어?", [MARRIAGE, DIVORCE],
      [("m", 0, "21", "2024"), ("d", 1, "21", "2024")], [comb("x", "m", "d", "diff")], lambda v: v["m"] - v["d"], 0),
    q("M02", "짝 현상", "2023년 서울은 혼인 100건당 이혼이 몇 건이었나요?", [MARRIAGE, DIVORCE],
      [("m", 0, "11", "2023"), ("d", 1, "11", "2023")], [per("d", "m", 100)], lambda v: v["d"] / v["m"] * 100, 1, "value"),
    q("M03", "짝 현상", "2024년 전국 혼인 건수는 이혼 건수의 몇 배야?", [MARRIAGE, DIVORCE],
      [("m", 0, "00", "2024"), ("d", 1, "00", "2024")], [comb("x", "m", "d", "ratio")], lambda v: v["m"] / v["d"], 1),
    q("M04", "짝 현상", "경기도에서 혼인에서 이혼을 뺀 건수가 2020년에서 2024년 사이 얼마나 변했어?", [MARRIAGE, DIVORCE],
      [("m1", 0, "31", "2020"), ("d1", 1, "31", "2020"), ("m2", 0, "31", "2024"), ("d2", 1, "31", "2024")],
      [comb("n1", "m1", "d1", "diff"), comb("n2", "m2", "d2", "diff"), comb("g", "n2", "n1", "diff")],
      lambda v: (v["m2"] - v["d2"]) - (v["m1"] - v["d1"]), 0, "trend"),
    q("M05", "짝 현상", "2015년 대구 혼인 건수랑 이혼 건수 차이 알려줘", [MARRIAGE, DIVORCE],
      [("m", 0, "22", "2015"), ("d", 1, "22", "2015")], [comb("x", "m", "d", "diff")], lambda v: v["m"] - v["d"], 0),
    # 분자·분모: 출생아 ÷ 주민등록 연앙인구 (명 / 명)
    q("P01", "분자·분모", "2024년 서울의 주민등록 연앙인구 1,000명당 출생아 수는?", [BIRTH_SIDO, POP_MID],
      [("b", 0, "11", "2024"), ("p", 1, "11", "2024")], [per("b", "p")], lambda v: v["b"] / v["p"] * 1000, 1, "value"),
    q("P02", "분자·분모", "2023년 세종시는 인구 천 명당 아기가 몇 명 태어났어?", [BIRTH_SIDO, POP_MID],
      [("b", 0, "29", "2023"), ("p", 1, "29", "2023")], [per("b", "p")], lambda v: v["b"] / v["p"] * 1000, 1, "value"),
    q("P03", "분자·분모", "2024년 부산 연앙인구 대비 출생아 비율은 몇 %야?", [BIRTH_SIDO, POP_MID],
      [("b", 0, "21", "2024"), ("p", 1, "21", "2024")], [per("b", "p", 100)], lambda v: v["b"] / v["p"] * 100, 1, "value"),
    q("P04", "분자·분모", "경기도 인구 1,000명당 출생아 수가 2023년보다 2024년에 얼마나 달라졌어?", [BIRTH_SIDO, POP_MID],
      [("b1", 0, "31", "2023"), ("p1", 1, "31", "2023"), ("b2", 0, "31", "2024"), ("p2", 1, "31", "2024")],
      [per("b1", "p1", sid="r1"), per("b2", "p2", sid="r2"), comb("g", "r2", "r1", "diff")],
      lambda v: round(v["b2"] / v["p2"] * 1000, 1) - round(v["b1"] / v["p1"] * 1000, 1), 1, "trend"),
    q("P05", "분자·분모", "2024년 전국 주민등록 연앙인구 천명당 혼인 건수는?", [MARRIAGE, POP_MID],
      [("m", 0, "00", "2024"), ("p", 1, "00", "2024")], [per("m", "p")], lambda v: v["m"] / v["p"] * 1000, 1, "value"),
    # 짝 현상: 전국 출생과 사망 (명 = 명)
    q("N01", "짝 현상", "2024년 우리나라 자연증가(출생아 수 - 사망자 수)는?", [BIRTH, DEATH],
      [("b", 0, "", "2024"), ("d", 1, "", "2024")], [comb("x", "b", "d", "diff")], lambda v: v["b"] - v["d"], 0),
    q("N02", "짝 현상", "2010년 출생아 수는 사망자 수의 몇 배였어?", [BIRTH, DEATH],
      [("b", 0, "", "2010"), ("d", 1, "", "2010")], [comb("x", "b", "d", "ratio")], lambda v: v["b"] / v["d"], 1),
    q("N03", "짝 현상", "출생아에서 사망자를 뺀 값이 2000년에서 2020년 사이 얼마나 줄었어?", [BIRTH, DEATH],
      [("b1", 0, "", "2000"), ("d1", 1, "", "2000"), ("b2", 0, "", "2020"), ("d2", 1, "", "2020")],
      [comb("n1", "b1", "d1", "diff"), comb("n2", "b2", "d2", "diff"), comb("g", "n2", "n1", "diff")],
      lambda v: (v["b2"] - v["d2"]) - (v["b1"] - v["d1"]), 0, "trend"),
    q("N04", "짝 현상", "2025년 사망자 수는 출생아 수보다 몇 % 많아?", [DEATH, BIRTH],
      [("d", 0, "", "2025"), ("b", 1, "", "2025")], [comb("x", "d", "b", "pct_change")], lambda v: (v["d"] / v["b"] - 1) * 100, 1),
    # 추계 vs 실적: 장래추계 중위 (천명) vs 실제 (명) — 천명이 명으로 환산된다
    q("F01", "추계 vs 실적", "2024년 출생아 수는 장래인구추계 중위 전망과 몇 명 차이 나?", [BIRTH, PROJ],
      [("a", 0, "", "2024"), ("p", 1, "20", "2024")], [comb("x", "a", "p", "diff")], lambda v: v["a"] - v["p"] * 1000, 0),
    q("F02", "추계 vs 실적", "2023년 실제 출생아는 중위 추계 출생아보다 몇 % 많거나 적었어?", [BIRTH, PROJ],
      [("a", 0, "", "2023"), ("p", 1, "20", "2023")], [comb("x", "a", "p", "pct_change")],
      lambda v: (v["a"] / (v["p"] * 1000) - 1) * 100, 1),
    q("F03", "추계 vs 실적", "2025년 사망자 수 실적과 중위 추계 사망자 수 차이는?", [DEATH, PROJ],
      [("a", 0, "", "2025"), ("p", 1, "30", "2025")], [comb("x", "a", "p", "diff")], lambda v: v["a"] - v["p"] * 1000, 0),
    # 옛 표 + 새 표: 외국인 2010 (총조사) + 2015~ (등록센서스)
    q("O01", "옛 표 + 새 표", "전국 외국인 수가 2010년에서 2020년까지 얼마나 늘었어?", [FOREIGN_2010, FOREIGN_2015],
      [("a", 0, "00", "2010"), ("b", 1, "00", "2020")], [comb("x", "b", "a", "diff")], lambda v: v["b"] - v["a"], 0, "trend"),
    q("O02", "옛 표 + 새 표", "서울 외국인은 2010년 대비 2024년에 몇 % 증가했어?", [FOREIGN_2010, FOREIGN_2015],
      [("a", 0, "11", "2010"), ("b", 1, "11", "2024")], [comb("x", "b", "a", "pct_change")],
      lambda v: (v["b"] / v["a"] - 1) * 100, 1, "trend"),
    q("O03", "옛 표 + 새 표", "경기도 외국인 수, 2010년과 2025년 차이는?", [FOREIGN_2010, FOREIGN_2015],
      [("a", 0, "31", "2010"), ("b", 1, "31", "2025")], [comb("x", "b", "a", "diff")], lambda v: v["b"] - v["a"], 0, "trend"),
    # 같은 대상 두 출처: 월별 혼인 표의 연 값 vs 직업별 혼인 표의 계
    q("S01", "같은 대상 두 출처", "2024년 서울 혼인 건수가 월별 통계랑 직업별 통계에서 얼마나 차이 나?", [MARRIAGE, MARRIAGE_JOB],
      [("a", 0, "11", "2024"), ("b", 1, "11", "2024")], [comb("x", "a", "b", "diff")], lambda v: v["a"] - v["b"], 0),
    q("S02", "같은 대상 두 출처", "2023년 전국 이혼 건수, 월별 이혼 통계와 직업별 이혼 통계 값 차이는?", [DIVORCE, DIVORCE_JOB],
      [("a", 0, "00", "2023"), ("b", 1, "00", "2023")], [comb("x", "a", "b", "diff")], lambda v: v["a"] - v["b"], 0),
    # 세 표
    q("T01", "세 표", "2024년 서울 인구 1,000명당 혼인 건수와 이혼 건수의 차이는?", [MARRIAGE, DIVORCE, POP_MID],
      [("m", 0, "11", "2024"), ("d", 1, "11", "2024"), ("p", 2, "11", "2024")],
      [per("m", "p", sid="rm"), per("d", "p", sid="rd"), comb("g", "rm", "rd", "diff")],
      lambda v: round(v["m"] / v["p"] * 1000, 1) - round(v["d"] / v["p"] * 1000, 1), 1),
    q("T02", "세 표", "2024년 부산은 인구 1,000명당 출생아 수가 혼인 건수보다 얼마나 많아?", [BIRTH_SIDO, MARRIAGE, POP_MID],
      [("b", 0, "21", "2024"), ("m", 1, "21", "2024"), ("p", 2, "21", "2024")],
      [per("b", "p", sid="rb"), per("m", "p", sid="rm"), comb("g", "rb", "rm", "diff")],
      lambda v: round(v["b"] / v["p"] * 1000, 1) - round(v["m"] / v["p"] * 1000, 1), 1),
    # 거절: 한 표에 없는 시점, 원인 질문
    q("R01", "거절(시점)", "2015년 서울 인구 1,000명당 출생아 수는?", [BIRTH_SIDO, POP_MID],
      [("b", 0, "11", "2015"), ("p", 1, "11", "2015")], [per("b", "p")], None, 1, "value", "refuse", None),
    q("R02", "거절(원인)", "2024년 이혼보다 혼인이 많았던 이유가 뭐야?", [MARRIAGE, DIVORCE],
      [], [], None, 0, "compare", "refuse", "causal"),
    q("R03", "거절(시점)", "2005년 전국 외국인 수가 2010년보다 얼마나 적었어?", [FOREIGN_2010, FOREIGN_2015],
      [], [], None, 0, "trend", "refuse", None),
]


def build(data: Data) -> list[dict]:
    rows = []
    for spec in QUESTIONS:
        labels = "ABC"
        values = {}
        steps = []
        for step_id, n, arg, period in spec["cells"]:
            key = spec["tables"][n]
            steps.append(select(step_id, labels[n], key, arg, period, data))
            if spec["kind"] == "answer":
                values[step_id] = read(data, key, arg, period)
        row = {"qid": spec["qid"], "relation": spec["relation"], "question": spec["question"], "table_keys": spec["tables"],
               "kind": spec["kind"]}
        if spec["kind"] == "answer":
            final = round(spec["final"](values), spec["round"])
            row["expected"] = {"cells": {k: v for k, v in values.items()}, "value": int(final) if spec["round"] == 0 else final,
                               "round": spec["round"]}
            show = [s["id"] for s in steps] + [c["id"] for c in spec["combine"]]
            row["gold_plan"] = {"steps": steps + spec["combine"], "show": show, "present": spec["present"]}
        else:
            row["expected"] = {"outcome": "refused", **({"reason": spec["refusal"]} if spec["refusal"] else {})}
            if steps:  # the plan a model would write; computing it is refused
                show = [s["id"] for s in steps] + [c["id"] for c in spec["combine"]]
                row["gold_plan"] = {"steps": steps + spec["combine"], "show": show, "present": spec["present"]}
        rows.append(row)
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args(argv)
    data_dir = Path(os.environ.get("KOSIS_WORKSPACE", ROOT)) / "data"
    rows = build(Data(data_dir))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"{len(rows)} questions -> {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
