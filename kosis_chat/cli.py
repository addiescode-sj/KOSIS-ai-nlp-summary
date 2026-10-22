"""Terminal entry point: load one table, then answer questions until the user quits."""

from __future__ import annotations

import argparse
import sys

from kosis_chat.chatbot import Chatbot
from kosis_chat.config import load_env, load_settings, resolve_path
from kosis_chat.data.loader import load_table
from kosis_chat.data.schema import SchemaError
from kosis_chat.llm.model import BACKENDS

QUIT_WORDS = {"q", "quit", "exit", "종료"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m kosis_chat.cli", description="Ask questions about one KOSIS table.")
    parser.add_argument("--table", required=True, help="data file (.csv or KOSIS .json) with <name>.meta.json next to it")
    parser.add_argument("--config", default=None, help="config file (default: configs/model.yaml)")
    parser.add_argument("--backend", choices=BACKENDS, default=None, help="override the configured backend")
    parser.add_argument("--model", default=None, help="override the HCX model name, e.g. HCX-007")
    parser.add_argument(
        "--thinking", choices=["none", "low", "medium", "high"], default=None, help="override thinking effort (HCX-007 only)"
    )
    parser.add_argument("--max-tokens", type=int, default=None, help="override max output tokens (raise it when thinking is on)")
    parser.add_argument("--form", default=None, help="text file with answer-format instructions appended to each question")
    parser.add_argument("--question", "-q", action="append", help="ask this question and exit (repeatable)")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    load_env()
    settings = load_settings(
        args.config, backend=args.backend, model_name=args.model, thinking_level=args.thinking, max_tokens=args.max_tokens
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
        bot = Chatbot(table, settings=settings, form=form)
    except ValueError as exc:  # backend misconfigured, e.g. HCX_API_KEY missing
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
        answer = bot.ask(question)
        if answer.ok:
            print(answer.text)
        else:
            print(f"[{answer.status}] {answer.error}")


if __name__ == "__main__":
    sys.exit(main())
