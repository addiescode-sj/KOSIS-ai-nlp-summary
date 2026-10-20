#!/usr/bin/env bash
# Project verification entry point. The Stop hook runs this before the agent
# may declare completion. Add your project's checks below; keep it fast.
set -euo pipefail
cd "${CLAUDE_PROJECT_DIR:-$(dirname "$0")/..}"

# Harness self-check: typecheck the hook scripts (skipped until npm install)
if [ -d node_modules ]; then
  npx tsc --noEmit
fi

# --- Add project checks here, e.g.: ---
# npm test --silent
# npm run lint

echo "verify: OK"
exit 0
