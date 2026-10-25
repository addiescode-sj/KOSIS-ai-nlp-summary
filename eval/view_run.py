"""One HTML page per run folder: every question with its answer, the facts behind it, the grade and the chart.

    python -m eval.view_run --run eval/results/<run_id>

Reads results.jsonl (and the latest grades/*/grades.jsonl from eval.grade_run when present) and writes
<run>/view.html. Open it in a browser; the charts load the Vega-Lite viewer from cdn.jsdelivr.net, so the page
needs internet. Data values stay inside the run folder (not committed).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from eval.scorer import read_jsonl

VEGA = ("https://cdn.jsdelivr.net/npm/vega@5.30.0", "https://cdn.jsdelivr.net/npm/vega-lite@5.21.0",
        "https://cdn.jsdelivr.net/npm/vega-embed@6.26.0")

PAGE = """<!doctype html>
<html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
__SCRIPTS__
<style>
body{font-family:system-ui,"Malgun Gothic",sans-serif;margin:0;background:#f6f7f9;color:#1d2330}
header{padding:16px 20px;background:#fff;border-bottom:1px solid #dde1e8;position:sticky;top:0;z-index:2}
h1{font-size:18px;margin:0 0 6px} .meta{font-size:12px;color:#5b6475}
.filters{margin-top:8px;display:flex;gap:8px;flex-wrap:wrap;font-size:13px} select,input{font-size:13px;padding:3px 6px}
main{padding:16px 20px;display:grid;gap:14px;max-width:1100px}
.card{background:#fff;border:1px solid #dde1e8;border-radius:8px;padding:14px 16px}
.head{display:flex;gap:8px;align-items:center;flex-wrap:wrap;font-size:13px}
.id{font-weight:700} .tag{padding:1px 7px;border-radius:10px;background:#eef1f6;font-size:12px}
.PASS{background:#dcf3e3;color:#17663a} .FAIL{background:#fbe0e0;color:#9b1c1c} .PARTIAL{background:#fff1cc;color:#7a5800}
.q{font-size:15px;margin:8px 0 6px;font-weight:600} .a{white-space:pre-wrap;font-size:14px;line-height:1.55;background:#f8f9fb;
border-left:3px solid #2f6db5;padding:8px 10px;margin:6px 0;overflow-wrap:anywhere}
.body{display:grid;grid-template-columns:minmax(0,1fr);gap:10px} .chart{overflow-x:auto}
table{border-collapse:collapse;font-size:12px;width:100%} th,td{border-bottom:1px solid #eceff4;padding:3px 6px;text-align:left;vertical-align:top}
th{color:#5b6475;font-weight:600} details{font-size:12px;margin-top:6px} pre{white-space:pre-wrap;font-size:11px;background:#f8f9fb;padding:6px}
.err{color:#9b1c1c;font-size:12px}
</style></head><body>
<header><h1>__TITLE__</h1><div class="meta" id="meta"></div>
<div class="filters">판정 <select id="fv"><option value="">전체</option><option>PASS</option><option>PARTIAL</option><option>FAIL</option><option>-</option></select>
유형 <select id="ft"><option value="">전체</option></select> 경로 <select id="fr"><option value="">전체</option></select>
검색 <input id="fq" placeholder="문항 id·질문"></div></header>
<main id="list"></main>
<script type="application/json" id="data">__DATA__</script>
<script>
const D = JSON.parse(document.getElementById('data').textContent);
const esc = s => String(s ?? '').replace(/[&<>"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
document.getElementById('meta').textContent = D.meta;
const fill = (id, vals) => { const el = document.getElementById(id); [...new Set(vals)].sort().forEach(v => { const o = document.createElement('option'); o.textContent = v; el.appendChild(o); }); };
fill('ft', D.rows.map(r => r.type)); fill('fr', D.rows.map(r => r.route));
function fmt(f) { if (f.value === null || f.value === undefined) return f.raw ? `'${f.raw}'` : ''; return Number(f.value).toLocaleString('ko-KR', {maximumFractionDigits: 4}) + (f.unit || ''); }
function card(r) {
  const facts = (r.facts || []).map(f => `<tr><td>${esc(f.id)}</td><td>${esc(fmt(f))}</td><td>${esc(f.period || '')}</td>
    <td>${esc(Object.values(f.dims || {}).join(' / '))}</td><td>${esc(f.derivation || f.source_ref || '')}</td><td>${esc(f.note || '')}</td></tr>`).join('');
  return `<section class="card" data-id="${esc(r.id)}"><div class="head"><span class="id">${esc(r.id)}</span>
    <span class="tag">${esc(r.kind)}</span><span class="tag">${esc(r.type)}</span><span class="tag ${esc(r.verdict)}">${esc(r.verdict)}</span>
    <span class="tag">${esc(r.route)}</span>${r.errors.length ? `<span class="err">${esc(r.errors.join(', '))}</span>` : ''}</div>
    <div class="q">${esc(r.question)}</div><div class="body"><div class="a">${esc(r.answer)}</div>
    <div class="chart" id="c-${esc(r.id)}">${r.chart ? '' : '<span class="meta">그래프 없음</span>'}</div><div>
    ${facts ? `<table><tr><th>id</th><th>값</th><th>시점</th><th>분류</th><th>원본 칸·계산식</th><th>메모</th></tr>${facts}</table>` : ''}
    <details><summary>상세 (채점 메모·모델 원문)</summary><pre>${esc(r.details.join('\\n'))}</pre><pre>${esc(r.raw.join('\\n---\\n'))}</pre></details></div></div></section>`;
}
function draw() {
  const v = fv.value, t = ft.value, rt = fr.value, q = fq.value.trim();
  const rows = D.rows.filter(r => (!v || r.verdict === v) && (!t || r.type === t) && (!rt || r.route === rt) && (!q || (r.id + r.question).includes(q)));
  list.innerHTML = rows.map(card).join('') || '<p>해당 문항 없음</p>';
  rows.filter(r => r.chart).forEach(r => vegaEmbed('#c-' + CSS.escape(r.id), r.chart, {actions: {export: true, source: true, compiled: false, editor: false}})
    .catch(e => { document.getElementById('c-' + r.id).innerHTML = '<span class="err">그래프 오류: ' + esc(e.message) + '</span>'; }));
}
const fv = document.getElementById('fv'), ft = document.getElementById('ft'), fr = document.getElementById('fr'), fq = document.getElementById('fq');
const list = document.getElementById('list');
[fv, ft, fr].forEach(el => el.addEventListener('change', draw)); fq.addEventListener('input', draw);
draw();
</script></body></html>
"""


def _latest_grades(run: Path) -> tuple[dict[str, dict], str]:
    folders = sorted((run / "grades").glob("*/grades.jsonl")) if (run / "grades").exists() else []
    if not folders:
        return {}, "채점 없음 (python -m eval.grade_run 실행 시 표시)"
    return {g["question_id"]: g for g in read_jsonl(folders[-1])}, f"채점: {folders[-1].parent.name}"


def build(run: Path, run_index: int = 0) -> Path:
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    golden = {g["id"]: g for g in read_jsonl(run / "golden.jsonl")} if (run / "golden.jsonl").exists() else {}
    grades, graded_note = _latest_grades(run)
    rows = []
    for r in read_jsonl(run / "results.jsonl"):
        if r.get("run_index", 0) != run_index:
            continue
        qid = r["question_id"]
        stat = r.get("stat_answer") or {}
        meta = r.get("decision_metadata") or {}
        grade = grades.get(qid, {})
        item = golden.get(qid, {})
        rows.append({
            "id": qid, "kind": r.get("kind", ""), "type": item.get("type") or "-", "question": item.get("question", ""),
            "answer": stat.get("text") or r.get("raw_answer") or r.get("call_error") or "",
            "facts": (stat.get("result") or {}).get("facts", []), "chart": r.get("chart"),
            "route": f"{meta.get('route', '-')}/{meta.get('compose', '-')}",
            "verdict": grade.get("verdict", "-"), "errors": grade.get("error_types", []),
            "details": grade.get("details", []), "raw": r.get("raw_responses") or [],
        })
    title = f"실행 결과 보기 — {manifest.get('stage', run.name)}"
    meta_line = (f"{run.name} · 파이프라인 {manifest.get('pipeline', 'chatbot')} · 프롬프트 {manifest.get('prompt_version')} · "
                 f"모델 {manifest.get('model', {}).get('model_name')} · {len(rows)}문항 · {graded_note}")
    data = json.dumps({"meta": meta_line, "rows": rows}, ensure_ascii=False).replace("</", "<\\/")
    scripts = "\n".join(f'<script src="{url}"></script>' for url in VEGA)
    page = PAGE.replace("__TITLE__", title).replace("__SCRIPTS__", scripts).replace("__DATA__", data)
    out = run / "view.html"
    out.write_text(page, encoding="utf-8")
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m eval.view_run", description=__doc__.splitlines()[0])
    parser.add_argument("--run", required=True, help="run folder written by eval.run_eval")
    parser.add_argument("--run-index", type=int, default=0)
    args = parser.parse_args(argv)
    out = build(Path(args.run), args.run_index)
    print(f"saved: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
