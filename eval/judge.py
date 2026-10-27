"""Natural-language quality of answers (T1-26): deterministic format checks plus an LLM judge for meaning.

    python -m eval.judge --run eval/results/<run_id>            # HCX-007 judge, 5 s between calls
    python -m eval.judge --run eval/results/<run_id> --labels eval/quality_labels.jsonl

Place in the flow: run_eval -> grade_run -> judge (optional) -> view_run -> regress. Numbers are not judged here
(eval/grade.py does that); the judge only reads the sentence. Format checks are code and deterministic; the judge
covers what rules cannot see (a question part left unanswered, an unsupported judgement, missing detail, awkward
wording). Score 0-2, capped at 1 when a format check fires. Refusals and clarifications use their own rubric.
The judge is not deterministic, so its score stays out of the regression gate; judged answers are cached by
(rubric version, model, question, answer) in data/cache/judge.jsonl.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import statistics
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from eval.scorer import read_jsonl, write_jsonl
from eval.view_raw import load_rows
from kosis_chat.config import load_env, load_settings, resolve_path
from kosis_chat.llm.model import STATUS_OK, LLMClient, build_client
from kosis_chat.llm.structured import ask_structured, structured_prompt

RUBRIC_VERSION = "quality-v2"
CACHE = resolve_path("data/cache/judge.jsonl")
Tag = Literal["ignored_part", "unsupported_judgement", "missing_detail", "awkward_label", "unclear_refusal", "other"]
TAGS = Tag.__args__
FORMAT_TAG = "format_error"

# ---------------------------------------------------------------- format checks (deterministic)

_SOURCE_LINE = re.compile(r"^\s*출처\s*:.*$", re.M)
_DUP_UNIT = re.compile(r"\d[\d,]*(?:\.\d+)?\s?(\S+?)\s+\1")  # "844명/㎢ 명/㎢"
_YEAR_MONTH = re.compile(r"(?<!\d)(?:19|20)\d{2}(?:0[1-9]|1[0-2])년")
_NUM_UNIT_JOSA = re.compile(r"\d[\d,]*(?:\.\d+)?\s?([가-힣]+?)(과|와|은|는|이|가|을|를)(?=[\s,.)]|$)")
_WITH_BATCHIM = {"와": "과", "는": "은", "가": "이", "를": "을"}  # josa written after a syllable without -> with a final
_TEMPLATE_JOSA = re.compile(r"와\(과\)|과\(와\)|은\(는\)|는\(은\)|이\(가\)|을\(를\)")


def _has_final(syllable: str) -> bool:
    return (ord(syllable) - 0xAC00) % 28 != 0


def format_issues(text: str) -> list[str]:
    """Format errors code can define exactly; the source line is left out (it quotes table names verbatim)."""
    body = _SOURCE_LINE.sub("", text)
    issues = [f"단위 중복 '{m.group(0)}'" for m in _DUP_UNIT.finditer(body)]
    issues += [f"연월 코드 노출 '{m.group(0)}'" for m in _YEAR_MONTH.finditer(body)]
    for m in _NUM_UNIT_JOSA.finditer(body):
        unit, josa = m.group(1), m.group(2)
        right = _WITH_BATCHIM.get(josa, josa) if _has_final(unit[-1]) else {v: k for k, v in _WITH_BATCHIM.items()}.get(josa, josa)
        if josa != right:
            issues.append(f"조사 '{unit}{josa}' → '{unit}{right}'")
    issues += [f"내부 라벨 '{w}'" for w in re.findall(r"[가-힣]+_계", body)]
    issues += [f"미완성 조사 '{m.group(0)}'" for m in _TEMPLATE_JOSA.finditer(body)]
    return issues


# ---------------------------------------------------------------- LLM judge (meaning)

class QualityJudgement(BaseModel):
    model_config = ConfigDict(extra="forbid")

    score: Literal[0, 1, 2]
    tags: list[Tag] = Field(default_factory=list)
    reason: str = Field(max_length=300)


SYSTEM_PROMPT = """당신은 통계 질의응답 서비스의 답변 문장을 평가하는 검토자입니다.
숫자가 맞는지는 이미 코드가 확인했습니다. 숫자 값은 판단하지 말고 문장의 전달력만 봅니다.

[답한 문항] 점수
- 2: 질문의 모든 부분에 바로 답하고, 읽기 쉽습니다.
- 1: 답은 되지만 어색한 표현, 표의 라벨을 그대로 옮긴 표현, 빠진 설명이 있습니다.
- 0: 질문이 명시적으로 요구한 것(원인, 합계, 평균, 두 기간 비교, 두 시점의 비중 등)을 답하지 않고 다른 값으로 대신했거나, 안내 없이 건너뛰었거나, 읽는 사람이 오해할 문장입니다.
[거절·되묻기 문항] 점수
- 2: 왜 답할 수 없는지(또는 무엇을 골라야 하는지)와 표로 확인할 수 있는 범위를 분명히 말합니다.
- 1: 사유는 있지만 어색하거나 불충분합니다.
- 0: 사유가 없거나 오해를 부릅니다. 정상적인 거절을 "답하지 않았다"는 이유로 0점 주지 않습니다.

문제 유형 태그(해당하는 것만)
- ignored_part: 질문이 명시적으로 요구한 것을 답하지 않음. 원인을 물었는데 "표로 알 수 없다"는 안내 없이 넘어간 경우도 여기에 해당하며 0점입니다.
- unsupported_judgement: 표에 없는 평가·원인·전망을 단정함 (예: "높은 편이다")
- missing_detail: 답은 했지만 보충할 정보가 빠짐 (예: 추이 질문에 처음·끝 값만 있고 중간 흐름(최고·최저 연도 등)이 없음)
- awkward_label: 표의 항목·분류 이름을 사람이 쓰는 말로 바꾸지 않고 그대로 써서 어색함 (예: "여자 1985 비율은", 지표 이름을 다른 지표처럼 줄여 씀)
- unclear_refusal: 거절·되묻기 사유가 불분명함
- other: 그 밖의 문장 문제
[코드가 확인한 사실]의 메모(예: "56개 중 가장 낮음")는 코드가 전체를 비교해 확인한 내용이므로 믿고 판단합니다.
reason은 한국어 한두 문장으로 씁니다."""


def user_prompt(row: dict) -> str:
    facts = "\n".join(f"- {f.get('period') or ''} {' / '.join((f.get('dims') or {}).values())} = "
                      f"{f.get('value') if f.get('value') is not None else f.get('raw')} {f.get('unit') or ''}"
                      f"{' (' + f['note'] + ')' if f.get('note') else ''}".strip()
                      for f in row.get("facts", [])[:12]) or "(없음)"
    kind = "거절·되묻기 문항" if row.get("outcome") in ("refused", "clarify") else "답한 문항"
    answer = _SOURCE_LINE.sub("", row["answer"]).strip()
    return f"[문항 종류]\n{kind}\n\n[질문]\n{row['question']}\n\n[답변]\n{answer}\n\n[코드가 확인한 사실]\n{facts}"


def _key(model: str, row: dict) -> str:
    return hashlib.sha256(json.dumps([RUBRIC_VERSION, model, row["question"], row["answer"]], ensure_ascii=False).encode()).hexdigest()


def judge_rows(rows: list[dict], client: LLMClient, model: str, interval: float = 0.0, cache: Path = CACHE) -> list[dict]:
    """rows: {id, question, answer, outcome, facts}. Returns one quality record per row (cached judgements reused)."""
    cached = {r["key"]: r["judgement"] for r in read_jsonl(cache)} if cache.exists() else {}
    out, new = [], []
    for row in rows:
        issues = format_issues(row["answer"]) if row["answer"] else []
        key = _key(model, row)
        judgement, status = cached.get(key), "cached"
        if judgement is None and row["answer"]:
            if new and interval:
                time.sleep(interval)
            result = ask_structured(client, SYSTEM_PROMPT, structured_prompt(user_prompt(row), QualityJudgement), QualityJudgement)
            status = result.status
            if result.status == STATUS_OK:
                judgement = result.value.model_dump()
                new.append({"key": key, "judgement": judgement})
        score = None if judgement is None else min(judgement["score"], 1) if issues else judgement["score"]
        out.append({"question_id": row["id"], "run_index": row.get("run_index", 0), "outcome": row.get("outcome", ""), "score": score,
                    "judge_score": None if judgement is None else judgement["score"],
                    "tags": sorted(set((judgement or {}).get("tags", [])) | ({FORMAT_TAG} if issues else set())),
                    "format_issues": issues, "reason": (judgement or {}).get("reason", ""), "judge_status": status})
    if new:
        cache.parent.mkdir(parents=True, exist_ok=True)
        with cache.open("a", encoding="utf-8") as f:
            f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in new)
    return out


# ---------------------------------------------------------------- run folder, labels, report

def run_rows(run: Path, run_index: int = 0) -> list[dict]:
    manifest, rows = load_rows(run, run_index)
    pipeline = manifest.get("pipeline") == "stat"
    out = []
    for row in rows:
        r = row["result"]
        stat = r.get("stat_answer") or {}
        out.append({"id": row["id"], "run_index": run_index, "question": row["question"],
                    "answer": (stat.get("text") if pipeline else None) or r.get("raw_answer") or "",
                    "outcome": stat.get("outcome", ""), "facts": (stat.get("result") or {}).get("facts", [])})
    return out


def detection(records: list[dict], labels: list[dict]) -> dict:
    """Share of labelled issues the checks found, split by who labelled them (human / candidate)."""
    by_id = {r["question_id"]: set(r["tags"]) for r in records}
    out: dict[str, dict] = {}
    for label in labels:
        if label["question_id"] not in by_id:
            continue
        c = out.setdefault(label["source"], {"hits": 0, "total": 0, "missed": []})
        c["total"] += 1
        if label["issue"] in by_id[label["question_id"]]:
            c["hits"] += 1
        else:
            c["missed"].append(f"{label['question_id']}:{label['issue']}")
    return out


def summary(records: list[dict]) -> dict:
    scored = [r["score"] for r in records if r["score"] is not None]
    return {
        "questions": len(records), "scored": len(scored),
        "mean": round(statistics.mean(scored), 2) if scored else None,
        "scores": dict(sorted(Counter(scored).items())),
        "zero": [r["question_id"] for r in records if r["score"] == 0],
        "format_issue_questions": sum(1 for r in records if r["format_issues"]),
        "tags": dict(Counter(t for r in records for t in r["tags"]).most_common()),
        "judge_failed": [r["question_id"] for r in records if r["score"] is None and r["judge_status"] != "cached"],
    }


def render(s: dict, meta: dict, found: dict | None) -> str:
    lines = [f"# 자연어 품질 — {meta['run']}", "",
             f"- judge: {meta['model']} · 루브릭 {RUBRIC_VERSION} · 답변 모델 {meta['answer_model']}", "",
             "| 지표 | 값 |", "| --- | --- |",
             f"| 품질 평균 (0~2) | {s['mean']} ({s['scored']}/{s['questions']}문항 채점) |",
             f"| 점수 분포 | {', '.join(f'{k}점 {v}' for k, v in s['scores'].items())} |",
             f"| 표기 오류가 있는 문항 (코드 검사) | {s['format_issue_questions']} |",
             f"| 0점 문항 | {', '.join(s['zero']) or '-'} |",
             f"| judge 실패 (점수 없음) | {', '.join(s['judge_failed']) or '-'} |",
             "", "문제 유형: " + (", ".join(f"{k} {v}" for k, v in s["tags"].items()) or "-")]
    if meta["model"] == meta["answer_model"]:
        lines.append("\n> 경고: judge와 답변 모델이 같아 자기 답 선호가 생길 수 있습니다.")
    if found:
        lines += ["", "| 보정 라벨 | 검출 | 놓친 것 |", "| --- | --- | --- |"]
        lines += [f"| {src} | {v['hits']}/{v['total']} | {', '.join(v['missed']) or '-'} |" for src, v in found.items()]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m eval.judge", description=__doc__.splitlines()[0])
    parser.add_argument("--run", required=True, help="run_eval folder")
    parser.add_argument("--model", default="HCX-007", help="judge model (default HCX-007, not the answer model)")
    parser.add_argument("--backend", choices=["hcx", "mock"], default="hcx")
    parser.add_argument("--interval", type=float, default=5.0, help="seconds between judge calls (HCX 429)")
    parser.add_argument("--labels", help="quality labels jsonl to measure detection against")
    parser.add_argument("--run-index", type=int, default=0)
    args = parser.parse_args(argv)
    run = Path(args.run)
    load_env()
    settings = load_settings(backend=args.backend, model_name=args.model, thinking_level="none", max_tokens=1024)
    records = judge_rows(run_rows(run, args.run_index), build_client(settings), args.model, args.interval)
    labels = [x for x in read_jsonl(Path(args.labels)) if x["run"] == run.name] if args.labels else []
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    meta = {"run": run.name, "model": args.model, "answer_model": (manifest.get("model") or {}).get("model_name")}
    s = summary(records)
    found = detection(records, labels) if labels else None
    write_jsonl(run / "quality.jsonl", records)
    (run / "quality.json").write_text(json.dumps(meta | {"summary": s, "detection": found}, ensure_ascii=False, indent=2), encoding="utf-8")
    (run / "quality.md").write_text(text := render(s, meta, found), encoding="utf-8")
    print(text)
    print(f"saved: {run / 'quality.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
