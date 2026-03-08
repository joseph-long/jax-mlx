#!/usr/bin/env bash
# Run benchmarks and save results to .benchmarks/<timestamp>_<githash>[_dirty].json
# Usage: scripts/benchmark.sh [extra pytest-benchmark args...]
#
# Results are saved to .benchmarks/ with filenames like:
#   2026-03-08T14-30-00_ae54cf9.json
#   2026-03-08T14-30-00_ae54cf9_dirty.json  (if working tree has uncommitted changes)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# Locate the freshest built dylib.
LATEST_DYLIB="$(ls -t build/*/lib/libpjrt_plugin_mlx.dylib 2>/dev/null | head -n 1)"
if [[ -z "$LATEST_DYLIB" ]]; then
    echo "error: no libpjrt_plugin_mlx.dylib found under build/; run 'uv pip install -e .' first" >&2
    exit 1
fi
export JAX_MLX_LIBRARY_PATH="$LATEST_DYLIB"

# Build the output filename.
TIMESTAMP="$(date -u '+%Y-%m-%dT%H-%M-%S')"
GIT_HASH="$(git rev-parse --short HEAD)"
if ! git diff --quiet || ! git diff --cached --quiet; then
    SUFFIX="_dirty"
else
    SUFFIX=""
fi
mkdir -p .benchmarks
OUTFILE=".benchmarks/${TIMESTAMP}_${GIT_HASH}${SUFFIX}.json"

echo "Running benchmarks against: $LATEST_DYLIB"
echo "Saving results to: $OUTFILE"

uv run pytest \
    -m benchmark \
    --benchmark-only \
    --benchmark-json="$OUTFILE" \
    --benchmark-columns=min,mean,stddev,rounds \
    "$@"

echo ""
echo "Saved: $OUTFILE"
