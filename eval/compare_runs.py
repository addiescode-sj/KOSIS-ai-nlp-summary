"""Side-by-side page of a raw run and a pipeline run on the same questions: the model alone vs the pipeline's answer.

    python -m eval.compare_runs --raw_path eval/results/<raw_run> --pipe_path eval/results/<pipeline_run>

First checks that both runs asked the same questions (golden.jsonl: same ids with the same question text) and stops
with the differences when they did not. Then writes <raw_path>/compare_<pipe run>.html with, for every question,
the raw reply as written next to the pipeline's answer (stat_answer.text, or raw_answer for text chatbot runs).
No grades or charts; the page needs no internet.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from eval.scorer import read_jsonl
from eval.view_raw import STYLE, load_rows, page_json, run_line

PAGE = """<!doctype html>
<html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
__STYLE__</head><body>
<header><h1>__TITLE__</h1><div class="meta" id="m1"></div><div class="meta" id="m2"></div>
<div class="filters">유형 <select id="ft"><option value="">전체</option></select> 파이프라인 결과 <select id="fo"><option value="">전체</option></select>
검색 <input id="fq" placeholder="문항 id·질문·답변"></div></header>
<main id="list"></main>
<script type="application/json" id="data">__DATA__</script>
<script>
const D = JSON.parse(document.getElementById('data').textContent);
const esc = s => String(s ?? '').replace(/[&<>"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
document.getElementById('m1').textContent = 'AI(raw): ' + D.raw_meta;
document.getElementById('m2').textContent = '파이프라인: ' + D.pipe_meta;
const fill = (id, vals) => { const el = document.getElementById(id); [...new Set(vals)].sort().forEach(v => { const o = document.createElement('option'); o.textContent = v; el.appendChild(o); }); };
fill('ft', D.rows.map(r => r.type)); fill('fo', D.rows.map(r => r.pipe.outcome));
function fmt(f) { if (f.value === null || f.value === undefined) return f.raw ? `'${f.raw}'` : ''; return Number(f.value).toLocaleString('ko-KR', {maximumFractionDigits: 4}) + (f.unit || ''); }
function side(s, label, cls) {
  const facts = (s.facts || []).map(f => `<tr><td>${esc(f.id)}</td><td>${esc(fmt(f))}</td><td>${esc(f.period || '')}</td>
    <td>${esc(Object.values(f.dims || {}).join(' / '))}</td><td>${esc(f.derivation || f.source_ref || '')}</td></tr>`).join('');
  return `<div><div class="head"><span class="label">${label}</span><span class="tag ${s.status === 'ok' ? '' : 'bad'}">${esc(s.status)}</span>
    ${s.outcome ? `<span class="tag">${esc(s.outcome)}</span>` : ''}${s.route ? `<span class="tag">${esc(s.route)}</span>` : ''}
    <span class="meta">${esc(s.stats)}</span></div><div class="a ${cls}">${esc(s.answer || s.error || '(답변 없음)')}</div>
    ${facts ? `<details><summary>사실 ${s.facts.length}개</summary><table><tr><th>id</th><th>값</th><th>시점</th><th>분류</th><th>원본 칸·계산식</th></tr>${facts}</table></details>` : ''}</div>`;
}
function card(r) {
  return `<section class="card"><div class="head"><span class="id">${esc(r.id)}</span><span class="tag">${esc(r.kind)}</span>
    <span class="tag">${esc(r.type)}</span></div>
    <div class="q">${esc(r.question)}</div><div class="cols">${side(r.raw, 'AI 답변 (raw)', '')}${side(r.pipe, '파이프라인 답변', 'pipe')}</div></section>`;
}
function draw() {
  const t = ft.value, o = fo.value, q = fq.value.trim();
  const rows = D.rows.filter(r => (!t || r.type === t) && (!o || r.pipe.outcome === o) && (!q || (r.id + r.question + r.raw.answer + r.pipe.answer).includes(q)));
  list.innerHTML = rows.map(card).join('') || '<p>해당 문항 없음</p>';
}
const ft = document.getElementById('ft'), fo = document.getElementById('fo'), fq = document.getElementById('fq');
const list = document.getElementById('list');
[ft, fo].forEach(el => el.addEventListener('change', draw)); fq.addEventListener('input', draw);
draw();
</script></body></html>
"""

MISSING = {"status": "실행 안 됨", "answer": "", "error": "", "stats": "", "facts": [], "outcome": "", "route": ""}


def question_differences(raw: Path, pipe: Path) -> list[str]:
    """Empty when both runs' golden.jsonl hold the same ids with the same question text."""
    a = {g["id"]: g["question"] for g in read_jsonl(raw / "golden.jsonl")}
    b = {g["id"]: g["question"] for g in read_jsonl(pipe / "golden.jsonl")}
    diffs = [f"{qid}: raw에만 있음" for qid in sorted(a.keys() - b.keys())]
    diffs += [f"{qid}: 파이프라인에만 있음" for qid in sorted(b.keys() - a.keys())]
    diffs += [f"{qid}: 질문 다름 (raw {a[qid]!r} / 파이프라인 {b[qid]!r})" for qid in sorted(a.keys() & b.keys())
              if a[qid].strip() != b[qid].strip()]
    return diffs


def _side(row: dict | None, pipeline: bool) -> dict:
    if row is None:
        return dict(MISSING)
    r = row["result"]
    stat = r.get("stat_answer") or {}
    meta = r.get("decision_metadata") or {}
    return {
        "status": row["status"], "error": row["error"], "stats": row["stats"],
        "answer": (stat.get("text") if pipeline else None) or r.get("raw_answer") or "",
        "facts": (stat.get("result") or {}).get("facts", []) if pipeline else [],
        "outcome": stat.get("outcome", "") if pipeline else "",
        "route": f"{meta.get('route', '-')}/{meta.get('compose', '-')}" if pipeline else "",
    }


def build(raw: Path, pipe: Path, run_index: int = 0, out: Path | None = None) -> Path:
    diffs = question_differences(raw, pipe)
    if diffs:
        raise ValueError("두 실행의 질문이 다릅니다:\n  " + "\n  ".join(diffs))
    raw_manifest, raw_rows = load_rows(raw, run_index)
    pipe_manifest, pipe_rows = load_rows(pipe, run_index)
    if raw_manifest.get("pipeline") != "raw":
        raise ValueError(f"--raw_path {raw.name} is a {raw_manifest.get('pipeline', 'chatbot')} run, not --pipeline raw")
    if pipe_manifest.get("pipeline") == "raw":
        raise ValueError(f"--pipe_path {pipe.name} is a raw run; give a chatbot or stat run")
    by_id = {r["id"]: r for r in pipe_rows}
    order = [r["id"] for r in raw_rows] + [r["id"] for r in pipe_rows if r["id"] not in {x["id"] for x in raw_rows}]
    raw_by_id = {r["id"]: r for r in raw_rows}
    rows = []
    for qid in order:
        a, b = raw_by_id.get(qid), by_id.get(qid)
        base = a or b
        rows.append({
            "id": qid, "kind": base["kind"], "type": base["type"], "question": base["question"],
            "raw": _side(a, pipeline=False), "pipe": _side(b, pipeline=True),
        })
    title = f"raw vs 파이프라인 — {raw_manifest.get('stage', raw.name)} / {pipe_manifest.get('stage', pipe.name)}"
    data = page_json({"raw_meta": run_line(raw, raw_manifest, len(raw_rows)),
                      "pipe_meta": run_line(pipe, pipe_manifest, len(pipe_rows)), "rows": rows})
    page = PAGE.replace("__TITLE__", title).replace("__STYLE__", STYLE).replace("__DATA__", data)
    out = out or raw / f"compare_{pipe.name}.html"
    out.write_text(page, encoding="utf-8")
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m eval.compare_runs", description=__doc__.splitlines()[0])
    parser.add_argument("--raw_path", required=True, help="run folder of eval.run_eval --pipeline raw (AI answers)")
    parser.add_argument("--pipe_path", required=True, help="run folder of a chatbot or stat run (pipeline answers)")
    parser.add_argument("--run-index", type=int, default=0)
    parser.add_argument("--out", help="output .html (default <raw_path>/compare_<pipe run>.html)")
    args = parser.parse_args(argv)
    try:
        out = build(Path(args.raw_path), Path(args.pipe_path), args.run_index, Path(args.out) if args.out else None)
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 2
    print(f"saved: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
