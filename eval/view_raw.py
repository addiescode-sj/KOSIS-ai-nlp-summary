"""One HTML page per raw run (--pipeline raw): every question with the model's reply as written, no grades or charts.

    python -m eval.view_raw --run eval/results/<run_id>

Reads results.jsonl, golden.jsonl and the model input from events.jsonl and writes <run>/raw_view.html. The page
needs no internet. Data values stay inside the run folder (not committed). load_rows is shared with eval.compare_runs.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from eval.scorer import read_jsonl

STYLE = """<style>
body{font-family:system-ui,"Malgun Gothic",sans-serif;margin:0;background:#f6f7f9;color:#1d2330}
header{padding:16px 20px;background:#fff;border-bottom:1px solid #dde1e8;position:sticky;top:0;z-index:2}
h1{font-size:18px;margin:0 0 6px} .meta{font-size:12px;color:#5b6475}
.filters{margin-top:8px;display:flex;gap:8px;flex-wrap:wrap;font-size:13px} select,input{font-size:13px;padding:3px 6px}
main{padding:16px 20px;display:grid;gap:14px;max-width:1200px}
.card{background:#fff;border:1px solid #dde1e8;border-radius:8px;padding:14px 16px}
.head{display:flex;gap:8px;align-items:center;flex-wrap:wrap;font-size:13px}
.id{font-weight:700} .tag{padding:1px 7px;border-radius:10px;background:#eef1f6;font-size:12px} .bad{background:#fbe0e0;color:#9b1c1c}
.q{font-size:15px;margin:8px 0 6px;font-weight:600} .a{white-space:pre-wrap;font-size:14px;line-height:1.55;background:#f8f9fb;
border-left:3px solid #2f6db5;padding:8px 10px;margin:6px 0;overflow-wrap:anywhere} .a.pipe{border-left-color:#3a8f5c}
.cols{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:12px} .label{font-size:12px;color:#5b6475;font-weight:600}
table{border-collapse:collapse;font-size:12px;width:100%} th,td{border-bottom:1px solid #eceff4;padding:3px 6px;text-align:left;vertical-align:top}
th{color:#5b6475;font-weight:600} details{font-size:12px;margin-top:6px} pre{white-space:pre-wrap;font-size:11px;background:#f8f9fb;padding:6px;overflow-x:auto}
</style>"""

PAGE = """<!doctype html>
<html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
__STYLE__</head><body>
<header><h1>__TITLE__</h1><div class="meta" id="meta"></div>
<div class="filters">유형 <select id="ft"><option value="">전체</option></select> 상태 <select id="fs"><option value="">전체</option></select>
검색 <input id="fq" placeholder="문항 id·질문·답변"></div></header>
<main id="list"></main>
<script type="application/json" id="data">__DATA__</script>
<script>
const D = JSON.parse(document.getElementById('data').textContent);
const esc = s => String(s ?? '').replace(/[&<>"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
document.getElementById('meta').textContent = D.meta;
const fill = (id, vals) => { const el = document.getElementById(id); [...new Set(vals)].sort().forEach(v => { const o = document.createElement('option'); o.textContent = v; el.appendChild(o); }); };
fill('ft', D.rows.map(r => r.type)); fill('fs', D.rows.map(r => r.status));
function card(r) {
  return `<section class="card"><div class="head"><span class="id">${esc(r.id)}</span><span class="tag">${esc(r.kind)}</span>
    <span class="tag">${esc(r.type)}</span><span class="tag ${r.status === 'ok' ? '' : 'bad'}">${esc(r.status)}</span>
    <span class="meta">${esc(r.stats)}</span></div>
    <div class="q">${esc(r.question)}</div><div class="a">${esc(r.answer || r.error || '(답변 없음)')}</div>
    <details><summary>모델에 보낸 입력 (표 + 질문)</summary><pre>${esc(r.input || '기록 없음')}</pre></details></section>`;
}
function draw() {
  const t = ft.value, st = fs.value, q = fq.value.trim();
  const rows = D.rows.filter(r => (!t || r.type === t) && (!st || r.status === st) && (!q || (r.id + r.question + r.answer).includes(q)));
  list.innerHTML = rows.map(card).join('') || '<p>해당 문항 없음</p>';
}
const ft = document.getElementById('ft'), fs = document.getElementById('fs'), fq = document.getElementById('fq');
const list = document.getElementById('list');
[ft, fs].forEach(el => el.addEventListener('change', draw)); fq.addEventListener('input', draw);
draw();
</script></body></html>
"""


def _inputs(run: Path) -> dict[str, str]:
    """User text of every model call by event id (generate_request events)."""
    path = run / "events.jsonl"
    if not path.exists():
        return {}
    return {e["event_id"]: e.get("user", "") for e in read_jsonl(path) if e.get("type") == "generate_request"}


def load_rows(run: Path, run_index: int = 0, with_input: bool = False) -> tuple[dict, list[dict]]:
    """(manifest, rows) for one repetition in golden order; each row has the golden question and the result row."""
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    golden = list(read_jsonl(run / "golden.jsonl")) if (run / "golden.jsonl").exists() else []
    results = {r["question_id"]: r for r in read_jsonl(run / "results.jsonl") if r.get("run_index", 0) == run_index}
    inputs = _inputs(run) if with_input else {}
    order = [g["id"] for g in golden] + [qid for qid in results if qid not in {g["id"] for g in golden}]
    items = {g["id"]: g for g in golden}
    rows = []
    for qid in order:
        if qid not in results:  # an interrupted run: questions not reached yet are left out
            continue
        r, item = results[qid], items.get(qid, {})
        usage = r.get("usage") or {}
        rows.append({
            "id": qid, "kind": r.get("kind", ""), "type": item.get("type") or "-", "question": item.get("question", ""),
            "status": r.get("call_status", "-"), "error": r.get("call_error") or "", "result": r,
            "stats": f"{r.get('latency_seconds', 0):.1f}s · 입력 {usage.get('promptTokens', '-')} / 출력 "
                     f"{usage.get('completionTokens', '-')} 토큰 · 표 {(r.get('scope') or {}).get('table_chars', '-')}자",
            "input": next((inputs[e] for e in r.get("event_ids", []) if e in inputs), ""),
        })
    return manifest, rows


def run_line(run: Path, manifest: dict, count: int) -> str:
    return (f"{run.name} · 파이프라인 {manifest.get('pipeline', 'chatbot')} · 프롬프트 {manifest.get('prompt_version')} · "
            f"모델 {manifest.get('model', {}).get('model_name')} · 표 범위 {manifest.get('scope_mode')} · "
            f"직렬화 {manifest.get('serialization')} · {count}문항 · 상태 {manifest.get('status')}")


def page_json(value: dict) -> str:
    return json.dumps(value, ensure_ascii=False).replace("</", "<\\/")


def build(run: Path, run_index: int = 0) -> Path:
    manifest, rows = load_rows(run, run_index, with_input=True)
    if manifest.get("pipeline") != "raw":
        raise ValueError(f"{run.name} is a {manifest.get('pipeline', 'chatbot')} run; use eval.view_run")
    for row in rows:
        row["answer"] = row.pop("result").get("raw_answer") or ""
    title = f"raw 답변 보기 — {manifest.get('stage', run.name)}"
    data = page_json({"meta": run_line(run, manifest, len(rows)), "rows": rows})
    page = PAGE.replace("__TITLE__", title).replace("__STYLE__", STYLE).replace("__DATA__", data)
    out = run / "raw_view.html"
    out.write_text(page, encoding="utf-8")
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m eval.view_raw", description=__doc__.splitlines()[0])
    parser.add_argument("--run", required=True, help="raw run folder written by eval.run_eval --pipeline raw")
    parser.add_argument("--run-index", type=int, default=0)
    args = parser.parse_args(argv)
    try:
        out = build(Path(args.run), args.run_index)
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 2
    print(f"saved: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
