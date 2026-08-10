#!/usr/bin/env bash
# qa_graph_journeys.sh -- self-locating wrapper for tools/qa_graph_journeys.js.
#
# WHY THIS EXISTS: the plain `node tools/qa_graph_journeys.js` fails on this
# workstation with MODULE_NOT_FOUND for "playwright" -- the default `node`
# on PATH is v12.22.9 (/usr/bin/node), Playwright needs Node >=20 and isn't
# installed where v12 looks for it anyway. Found live (2026-08-10): the
# master hit exactly this running the harness for the first time, which is
# what "permanent" actually means -- a script only *I* know how to invoke
# is not a deliverable. This wrapper finds a working Node 20+ and the
# isolated Playwright install for you; you should never need to type the
# raw NODE_PATH/node-binary invocation from the .js file's header again.
#
# Usage:
#   tools/qa_graph_journeys.sh [url]
#   (url defaults to the live console graph page, same as the .js file)
#
# What it does, in order:
#   1. Uses $QA_NODE_BIN if you set it (escape hatch for a different node).
#   2. Else looks for a Node >=20 binary in the usual nvm install location
#      on this workstation (~/.nvm/versions/node/*), picking the newest.
#   3. Else falls back to whatever `node` is on PATH, and REFUSES to run if
#      its version is <20 (a clear error, not Playwright's own cryptic
#      MODULE_NOT_FOUND/version check further down the line).
#   4. Sets NODE_PATH to the isolated Playwright install this machine's
#      other QA tooling already depends on (the xgpage skill's qa_widths.js
#      documents the same path: ~/.claude/skills/xgpage/SKILL.md, "Reusable
#      script" section) unless $QA_NODE_PATH overrides it.
set -euo pipefail
cd "$(dirname "$0")/.."

NODE_BIN="${QA_NODE_BIN:-}"
if [ -z "$NODE_BIN" ]; then
  # Newest nvm-installed node >=20, if any (sorted by version, last wins).
  for d in $(ls -d "$HOME"/.nvm/versions/node/v* 2>/dev/null | sort -V); do
    v="$("$d/bin/node" --version 2>/dev/null | sed 's/^v//;s/\..*//')"
    if [ -n "$v" ] && [ "$v" -ge 20 ] 2>/dev/null; then
      NODE_BIN="$d/bin/node"
    fi
  done
fi
if [ -z "$NODE_BIN" ]; then
  if command -v node >/dev/null 2>&1; then
    v="$(node --version | sed 's/^v//;s/\..*//')"
    if [ "$v" -ge 20 ] 2>/dev/null; then
      NODE_BIN="$(command -v node)"
    fi
  fi
fi
if [ -z "$NODE_BIN" ]; then
  echo "qa_graph_journeys.sh: no Node >=20 found (checked \$QA_NODE_BIN, ~/.nvm/versions/node/*, and PATH)." >&2
  echo "  Playwright requires Node 20+; the default /usr/bin/node on this workstation is v12." >&2
  echo "  Set QA_NODE_BIN=/path/to/node20 and rerun, or install one (e.g. via nvm)." >&2
  exit 1
fi

NODE_PATH_VAL="${QA_NODE_PATH:-/localhome/xya120/.npm/_npx/9833c18b2d85bc59/node_modules}"
if [ ! -d "$NODE_PATH_VAL/playwright" ]; then
  echo "qa_graph_journeys.sh: no playwright package found at $NODE_PATH_VAL" >&2
  echo "  Set QA_NODE_PATH to wherever this workstation's isolated Playwright install lives" >&2
  echo "  (see ~/.claude/skills/xgpage/SKILL.md, \"Reusable script\" section, for the current path)." >&2
  exit 1
fi

echo "qa_graph_journeys.sh: using node=$NODE_BIN playwright=$NODE_PATH_VAL" >&2
exec env NODE_PATH="$NODE_PATH_VAL" "$NODE_BIN" tools/qa_graph_journeys.js "$@"
