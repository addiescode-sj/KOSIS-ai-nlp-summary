#!/usr/bin/env bash
# Project verification entry point. The Stop hook runs this before the agent
# may declare completion. Offline only: no network, no API keys, no HCX calls.
set -euo pipefail
cd "${CLAUDE_PROJECT_DIR:-$(dirname "$0")/..}"

# Harness self-check: typecheck the hook scripts (skipped until npm install)
if [ -d node_modules ]; then
  npx tsc --noEmit
fi

# Python: prefer the project venv, then python3, then python (Windows Git Bash).
PY="${PYTHON:-}"
if [ -z "$PY" ]; then
  for candidate in .venv/Scripts/python.exe .venv/bin/python python3 python; do
    if command -v "$candidate" >/dev/null 2>&1 || [ -x "$candidate" ]; then
      PY="$candidate"
      break
    fi
  done
fi
if [ -z "$PY" ]; then
  echo "verify: no Python found (install Python 3.12 or set PYTHON=...)" >&2
  exit 1
fi
if ! "$PY" -c "import pytest, yaml" >/dev/null 2>&1; then
  echo "verify: missing packages; run: $PY -m pip install -r requirements.txt" >&2
  exit 1
fi

# Offline tests (MockClient only).
"$PY" -m pytest

# Real data, golden sets and eval results must never be tracked by git.
if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  for path in data/samples/x eval/golden/x eval/results/x .env; do
    if ! git check-ignore -q "$path"; then
      echo "verify: $path is not gitignored" >&2
      exit 1
    fi
  done
  tracked="$(git ls-files data/samples eval/golden eval/results .env)"
  if [ -n "$tracked" ]; then
    echo "verify: real-data paths are tracked by git:" >&2
    echo "$tracked" >&2
    exit 1
  fi
fi

echo "verify: OK"
exit 0
