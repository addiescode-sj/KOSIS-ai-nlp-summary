"""Terminal entry point: load one table, then answer questions with the stat module until the user quits.

T1-28 S6: the questions go to kosis_chat.api.StatModule (the C pipeline: rules, code-computed numbers, model
sentences), the same answers summarize_stat gives. The file is read here (adapters.parquet.read_parquet) and handed
to the module as a loaded table (StatInput); its meta is the tables.csv row next to the file or one folder up. The
B0 chatbot CLI (--scope, --form, --answer-format, --prompt-version) is `python -m kosis_chat.legacy.cli`.

--backend hcx needs HCX_API_KEY in the shell environment (.env is not read since T1-28 S2c); --backend mock needs no
key (tests and offline checks).
"""

from __future__ import annotations

import argparse
import os
import sys

from kosis_chat.adapters.parquet import catalog_row, fetched_at_of, find_catalog, read_parquet
from kosis_chat.api import StatModule
from kosis_chat.config import resolve_path
from kosis_chat.llm.model import BACKENDS
from kosis_chat.llm.providers import LLMKeyMissing
from kosis_chat.prepare.schemas import Problem, SchemaError
from kosis_chat.schemas.io import ParquetTable, StatInput, StatOptions
from kosis_chat.schemas.resources import DEFAULT_MODEL, MOCK_PROVIDER, LLMConfig

QUIT_WORDS = {"q", "quit", "exit", "종료"}
KEY_ENV = "HCX_API_KEY"  # the shell environment variable the hcx backend's key is read from


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m kosis_chat.cli", description="Ask questions about one KOSIS table.")
    parser.add_argument("--table", required=True, help="table file: data/values/<org>_<tbl>.parquet (meta from data/tables.csv)")
    parser.add_argument("--backend", choices=BACKENDS, default="hcx", help="model provider (mock: no key, tests only)")
    parser.add_argument("--model", default=None, help=f"HCX model name (default {DEFAULT_MODEL})")
    parser.add_argument(
        "--thinking", choices=["none", "low", "medium", "high"], default=None, help="thinking effort (HCX-007 only)"
    )
    parser.add_argument("--max-tokens", type=int, default=None, help="max output tokens (raise it when thinking is on)")
    parser.add_argument("--question", "-q", action="append", help="ask this question and exit (repeatable)")
    return parser


def llm_config(args: argparse.Namespace) -> LLMConfig:
    """The LLMConfig of the command line; the hcx key comes from the shell environment (None when unset: the first
    question that needs the model then reports LLMKeyMissing)."""
    fields = {"model": args.model or (MOCK_PROVIDER if args.backend == MOCK_PROVIDER else DEFAULT_MODEL),
              "provider": args.backend}
    if args.backend != MOCK_PROVIDER:
        fields["api_key"] = os.environ.get(KEY_ENV) or None
    if args.thinking is not None:
        fields["thinking_effort"] = args.thinking
    if args.max_tokens is not None:
        fields["max_tokens"] = args.max_tokens
    return LLMConfig(**fields)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    path = resolve_path(args.table)
    try:
        if not path.exists():
            raise SchemaError([Problem(str(path), "file not found")])
        if path.suffix.lower() != ".parquet":
            raise SchemaError([Problem(path.name, f"unsupported input format {path.suffix!r} (use data/values/*.parquet)")])
        row = catalog_row(path)
        table = ParquetTable(frame=read_parquet(path), fetched_at=fetched_at_of(path))
    except SchemaError as exc:
        print("입력 표를 읽을 수 없습니다:", file=sys.stderr)
        for problem in exc.problems:
            print(f"  - {problem}", file=sys.stderr)
        return 2
    except ValueError as exc:  # the frame breaks the column contract (ParquetTable)
        print(f"입력 표를 읽을 수 없습니다: {exc}", file=sys.stderr)
        return 2
    llm = llm_config(args)
    module = StatModule(llm, StatOptions(catalog_path=find_catalog(path)))
    print(f"[{row.get('table_name', '')}] backend={llm.provider} model={llm.model}")

    questions = args.question or None
    while True:
        if questions is not None:
            if not questions:
                return 0
            question = questions.pop(0)
            print(f"> {question}")
        else:
            try:
                question = input("> ").strip()
            except EOFError:
                return 0
            if question.lower() in QUIT_WORDS:
                return 0
            if not question:
                continue
        try:
            out = module.answer(StatInput(question=question, tables=[table]))
        except LLMKeyMissing as exc:  # the client is built when a question first needs the model
            print(f"설정 오류: {exc} (set {KEY_ENV} in the shell)", file=sys.stderr)
            return 2
        except SchemaError as exc:  # the table's rows or catalog row break the input contract
            print("입력 표를 읽을 수 없습니다:", file=sys.stderr)
            for problem in exc.problems:
                print(f"  - {problem}", file=sys.stderr)
            return 2
        except RuntimeError as exc:  # the model call failed (summarize_stat's contract): this question only
            print(f"[error] {exc}")
            continue
        print(out.answer.text)
        if out.chart is not None:
            print(f"[차트: {out.chart.kind}, 점 {sum(len(s.points) for s in out.chart.series)}개]")


if __name__ == "__main__":
    sys.exit(main())
