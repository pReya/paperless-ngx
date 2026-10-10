#!/usr/bin/env bash
# Runs the taxonomy-candidates demo suite twice: against `dev` (the problem)
# and against the current checkout (the fix). Uses this repo's own .venv for
# both, so run `uv sync --frozen` once first.
#
#   src/paperless_ai/tests/e2e/run_before_after.sh [before-ref]
#
# Expected: 3 failed / 2 passed on `dev`, 5 passed on the feature branch.
set -uo pipefail

ROOT=$(git rev-parse --show-toplevel)
SUITE=src/paperless_ai/tests/e2e
BEFORE_REF=${1:-dev}
PYTEST=("$ROOT/.venv/bin/python" -m pytest "$SUITE" -p no:randomly -n0 -q --no-cov)

echo "=============================================================="
echo " BEFORE: $BEFORE_REF  (expected: the three 'offered' tests fail)"
echo "=============================================================="
WORKTREE=$(mktemp -d)/paperless-ngx-before
git -C "$ROOT" worktree add -q "$WORKTREE" "$BEFORE_REF"
mkdir -p "$WORKTREE/$(dirname "$SUITE")"
cp -r "$ROOT/$SUITE" "$WORKTREE/$SUITE"
(cd "$WORKTREE" && "${PYTEST[@]}" 2>&1 | grep -E "^E   |^FAILED|^PASSED|passed|failed")
git -C "$ROOT" worktree remove --force "$WORKTREE"
git -C "$ROOT" worktree prune

echo
echo "=============================================================="
echo " AFTER: $(git -C "$ROOT" rev-parse --abbrev-ref HEAD)  (expected: all pass)"
echo "=============================================================="
(cd "$ROOT" && "${PYTEST[@]}" 2>&1 | grep -E "^E   |^FAILED|^PASSED|passed|failed")
