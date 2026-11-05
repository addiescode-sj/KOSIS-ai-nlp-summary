"""Terminal entry point: load one table, then answer questions until the user quits.

--backend hcx needs HCX_API_KEY in the shell environment (.env is not read since T1-28 S2c).
"""

from __future__ import annotations

import argparse
import sys

from kosis_chat.chatbot import Chatbot
from kosis_chat.config import load_settings, resolve_path
from kosis_chat.contract import org_id_from_table_path, to_stat_answer
from kosis_chat.data.loader import load_table
from kosis_chat.data.schema import SchemaError
from kosis_chat.llm import prompts
from kosis_chat.llm.model import BACKENDS
from kosis_chat.llm.providers import LLMKeyMissing

QUIT_WORDS = {"q", "quit", "exit", "종료"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m kosis_chat.cli", description="Ask questions about one KOSIS table.")
    parser.add_argument("--table", required=True, help="table file: data/values/<org>_<tbl>.parquet (meta from data/tables.csv)")
    parser.add_argument("--config", default=None, help="YAML settings file in the old layout (deprecated; default: built-in settings)")
    parser.add_argument("--backend", choices=BACKENDS, default=None, help="override the configured backend")
    parser.add_argument("--answer-format", choices=["text", "json"], default="text")
    parser.add_argument("--prompt-version", choices=sorted(prompts.SYSTEM_PROMPTS), default=None)
    parser.add_argument("--model", default=None, help="override the HCX model name, e.g. HCX-007")
    parser.add_argument(
        "--thinking", choices=["none", "low", "medium", "high"], default=None, help="override thinking effort (HCX-007 only)"
    )
    parser.add_argument("--max-tokens", type=int, default=None, help="override max output tokens (raise it when thinking is on)")
    parser.add_argument("--form", default=None, help="text file with answer-format instructions appended to each question")
    parser.add_argument("--serialization", choices=["long", "wide"], default=None, help="table text style (default: long)")
    parser.add_argument(
        "--scope", choices=["auto", "full"], default="auto", help="table part per question: auto (from the question) or full"
    )
    parser.add_argument("--question", "-q", action="append", help="ask this question and exit (repeatable)")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        version = (
            prompts.resolve_answer_prompt(args.answer_format, args.prompt_version)
            if args.prompt_version else None
        )
    except ValueError as exc:
        print(f"설정 오류: {exc}", file=sys.stderr)
        return 2
    settings = load_settings(
        args.config,
        backend=args.backend,
        prompt_version=version,
        model_name=args.model,
        thinking_level=args.thinking,
        max_tokens=args.max_tokens,
        serialization=args.serialization,
    )
    try:
        table = load_table(resolve_path(args.table))
    except SchemaError as exc:
        print("입력 표를 읽을 수 없습니다:", file=sys.stderr)
        for problem in exc.problems:
            print(f"  - {problem}", file=sys.stderr)
        return 2

    try:
        form = resolve_path(args.form).read_text(encoding="utf-8-sig") if args.form else None
    except OSError as exc:
        print(f"--form 파일을 읽을 수 없습니다: {exc}", file=sys.stderr)
        return 2

    try:
        bot = Chatbot(table, settings=settings, form=form, scope_mode=args.scope, answer_format=args.answer_format)
    except ValueError as exc:  # backend misconfigured, e.g. an unknown scope
        print(f"설정 오류: {exc}", file=sys.stderr)
        return 2
    print(f"[{table.meta.tbl_nm}] backend={settings.backend} model={settings.model_name} prompt={bot.prompt_version}")

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
            answer = bot.ask(question)
        except LLMKeyMissing as exc:  # the client is built when a question first needs the model
            print(f"설정 오류: {exc}", file=sys.stderr)
            return 2
        print(f"[범위 {answer.scope.get('cells', 0):,}칸, {answer.scope.get('table_chars', 0):,}자]")
        if answer.ok:
            try:
                print(to_stat_answer(question, answer, org_id=org_id_from_table_path(args.table, table.meta.tbl_id)).text)
            except ValueError as exc:
                print(f"[format_error] {exc}")
        else:
            print(f"[{answer.status}] {answer.error}")


if __name__ == "__main__":
    sys.exit(main())
