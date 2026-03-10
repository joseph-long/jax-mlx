#!/usr/bin/env bash
# Run JAX's upstream test suite against the jax-mlx backend.
#
# Usage:
#   bash scripts/jax_tests.sh [pytest-args...]
#
# The JAX source checkout lives in .agent-context/jax/.  This script
# checks out the pinned JAX version (matching the installed jaxlib), then
# runs the four test files that cover core op coverage:
#
#   tests/lax_numpy_test.py         jnp.* ops
#   tests/lax_autodiff_test.py      autodiff rules
#   tests/lax_control_flow_test.py  while/cond/scan
#   tests/lax_numpy_indexing_test.py  advanced indexing
#
# Extra pytest args are forwarded verbatim.
#
# Environment:
#   JAX_TAG          Git tag / ref to check out (default: jax-v0.9.0).
#                    Must match the jaxlib version installed in .venv.
#   JAX_TEST_FILES   Space-separated list of test files to run (relative
#                    to the JAX repo root).  Defaults to the four files
#                    above.
#   NO_PULL          Set to 1 to skip the fetch/checkout step.
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
JAX_DIR="$REPO_ROOT/.agent-context/jax"

# Version must match the jaxlib installed in .venv (see uv.lock / pyproject.toml).
JAX_TAG="${JAX_TAG:-jax-v0.9.0}"

# --- locate the freshest built dylib ---
# Search both the cmake direct output (build/lib/) and the Python wheel output (build/*/lib/)
LATEST_DYLIB="$(ls -t \
    "$REPO_ROOT/build/lib/libpjrt_plugin_mlx.dylib" \
    "$REPO_ROOT/build"/*/lib/libpjrt_plugin_mlx.dylib \
    2>/dev/null | head -n 1)"
if [[ -z "$LATEST_DYLIB" ]]; then
  echo "ERROR: no libpjrt_plugin_mlx.dylib found under build/.  Run: uv pip install -e ." >&2
  exit 1
fi

# --- check out the pinned JAX version (unless skipped) ---
if [[ "${NO_PULL:-0}" != "1" ]]; then
  echo "=== Checking out JAX $JAX_TAG ($JAX_DIR) ===" >&2
  git -C "$JAX_DIR" fetch --depth=1 origin "refs/tags/$JAX_TAG" 2>&1 | tail -3 >&2
  git -C "$JAX_DIR" checkout FETCH_HEAD 2>&1 | tail -1 >&2
fi
JAX_HEAD="$(git -C "$JAX_DIR" rev-parse --short HEAD)"
echo "=== JAX @ $JAX_HEAD ($JAX_TAG) ===" >&2

# --- choose test files ---
DEFAULT_FILES="tests/lax_numpy_test.py tests/lax_autodiff_test.py tests/lax_control_flow_test.py tests/lax_numpy_indexing_test.py"
TEST_FILES="${JAX_TEST_FILES:-$DEFAULT_FILES}"

# --- run ---
echo "=== Running JAX tests with jax-mlx backend ===" >&2
echo "    dylib: $LATEST_DYLIB" >&2
echo "    files: $TEST_FILES" >&2

cd "$JAX_DIR"
exec env \
  JAX_PLATFORMS=mlx \
  JAX_MLX_LIBRARY_PATH="$LATEST_DYLIB" \
  COLUMNS=200 \
  "$REPO_ROOT/.venv/bin/python" -m pytest \
  $TEST_FILES \
  -p no:warnings \
  --override-ini="addopts=" \
  "$@"
