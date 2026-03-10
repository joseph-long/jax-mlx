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
# Structured output:
#   Each run writes per-file JUnit XML + logs under:
#     .benchmarks/jax_tests_<timestamp>_<jax_head>/
#   After all files finish, a failure summary with selectors is generated via
#   scripts/summarize_jax_tests.py.
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
IFS=' ' read -r -a TEST_FILES_ARR <<< "$TEST_FILES"

# --- prepare structured output directory ---
BENCHMARK_DIR="$REPO_ROOT/.benchmarks"
mkdir -p "$BENCHMARK_DIR"
TIMESTAMP="$(date +"%Y-%m-%dT%H-%M-%S")"
RUN_DIR="$BENCHMARK_DIR/jax_tests_${TIMESTAMP}_${JAX_HEAD}"
mkdir -p "$RUN_DIR"
echo "=== Structured outputs ===" >&2
echo "    dir: $RUN_DIR" >&2

# --- run ---
echo "=== Running JAX tests with jax-mlx backend ===" >&2
echo "    dylib: $LATEST_DYLIB" >&2
echo "    files: $TEST_FILES" >&2

cd "$JAX_DIR"
overall_rc=0
for test_file in "${TEST_FILES_ARR[@]}"; do
  safe_name="$(echo "$test_file" | tr '/.' '__')"
  junit_xml="$RUN_DIR/${safe_name}.junit.xml"
  log_file="$RUN_DIR/${safe_name}.log"

  echo "=== pytest $test_file ===" >&2
  set +e
  env \
    JAX_PLATFORMS=mlx \
    JAX_MLX_LIBRARY_PATH="$LATEST_DYLIB" \
    COLUMNS=200 \
    uv run --project "$REPO_ROOT" python -m pytest \
    "$test_file" \
    -p no:warnings \
    --override-ini="addopts=" \
    --junitxml="$junit_xml" \
    "$@" 2>&1 | tee "$log_file"
  rc=${PIPESTATUS[0]}
  set -e

  echo -e "${test_file}\t${rc}" >> "$RUN_DIR/exit_codes.tsv"
  if [[ $rc -ne 0 ]]; then
    overall_rc=$rc
  fi
done

echo "=== Summary ===" >&2
uv run --project "$REPO_ROOT" python "$REPO_ROOT/scripts/summarize_jax_tests.py" "$RUN_DIR" \
  | tee "$RUN_DIR/summary.txt"

exit $overall_rc
