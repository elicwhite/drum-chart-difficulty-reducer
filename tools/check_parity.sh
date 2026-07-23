#!/usr/bin/env bash
#
# The repo's single "prove Python == JavaScript" entrypoint.
#
# Runs BOTH language suites and prints one PASS/FAIL summary. Each suite
# independently diffs its own reduce() implementation against the SAME
# frozen fixtures under data/fixtures/ (parity_fixture.json for the model
# backend, scorecard_parity_fixture.json for the scorecard backend) --
# this script does not re-run a cross-language diff itself, it just proves
# both languages land on the identical fixture, which is transitively
# "Python == JS" (both equal the same third thing, note-for-note).
#
# Coverage per language x backend (4 cells, all required to pass):
#   Python / model      python/tests/test_parity.py
#   Python / scorecard   python/tests/test_parity_scorecard.py
#   JS     / model       javascript/test/parity.test.js ("fixture songs match note-for-note")
#   JS     / scorecard   javascript/test/parity.test.js ("scorecard fixture songs match note-for-note")
# Plus supporting suites both sides run as part of the same command:
#   portable_exp bit-exactness, decode.py/decode.js tie-break unit tests.
#
# Usage (from the repo root):
#   tools/check_parity.sh
#
# Requires: a venv with sklearn 1.8.0 (a different sklearn version can't
# unpickle the pinned model) for the Python side; a working `node` (no npm
# installs needed) for the JS side.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
REF_ROOT="$REPO_ROOT"
# Python interpreter: uses $PYTHON if set, else python3 from PATH. It must have the
# runtime deps installed (pip install -r requirements.txt). No repo-internal venv is
# assumed.
PY="${PYTHON:-python3}"

PY_LOG="$(mktemp -t check_parity_py.XXXXXX)"
JS_LOG="$(mktemp -t check_parity_js.XXXXXX)"
trap 'rm -f "$PY_LOG" "$JS_LOG"' EXIT

echo "== drum-reducer-reference: cross-language parity check =="
echo "   repo root:    $REPO_ROOT"
echo "   python:       $PY"
echo "   node:         $(command -v node) ($(node --version 2>/dev/null))"
echo

echo "-- [1/2] Python suite (pytest python/tests/) --"
if ! command -v "$PY" >/dev/null 2>&1; then
  echo "   SKIPPED: python interpreter '$PY' not found -- install deps (pip install -r"
  echo "            requirements.txt) or set PYTHON=/path/to/python."
  PY_STATUS=127
else
  ( cd "$REPO_ROOT" && OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}" "$PY" -m pytest "$REF_ROOT/python/tests/" -q ) \
    > "$PY_LOG" 2>&1
  PY_STATUS=$?
  tail -n 20 "$PY_LOG"
fi
echo

echo "-- [2/2] JavaScript suite (node --test javascript/test/) --"
( cd "$REF_ROOT/javascript" && npm test ) > "$JS_LOG" 2>&1
JS_STATUS=$?
tail -n 20 "$JS_LOG"
echo

echo "== SUMMARY =="
if [ "$PY_STATUS" -eq 0 ]; then
  echo "  Python  (model + scorecard, both fixtures): PASS"
else
  echo "  Python  (model + scorecard, both fixtures): FAIL (exit $PY_STATUS) -- see $PY_LOG"
fi
if [ "$JS_STATUS" -eq 0 ]; then
  echo "  JS      (model + scorecard, both fixtures): PASS"
else
  echo "  JS      (model + scorecard, both fixtures): FAIL (exit $JS_STATUS) -- see $JS_LOG"
fi

if [ "$PY_STATUS" -eq 0 ] && [ "$JS_STATUS" -eq 0 ]; then
  echo
  echo "PY == JS: both languages reproduce both fixtures (model backend 0.1703 rb4_test edit_rate; scorecard backend note-for-note). PASS."
  exit 0
else
  echo
  echo "PY == JS: NOT PROVEN -- at least one suite failed. FAIL."
  exit 1
fi
