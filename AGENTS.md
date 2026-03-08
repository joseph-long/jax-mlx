# Guidelines

- You may NEVER skip or xfail tests without my explicit approval.
- You MUST use `uv` to manage dependencies.
- You MUST use `uv run ...` to execute commands.
- You may NEVER use `--no-verify` for git commits.
- You may NEVER push to `main` unless explicitly requested.
- You may NEVER delete operations or tests without my explicit approval.
- The only target we care about for MLX is Apple Silicon, which is a unified-memory architecture. Don't copy memory "to device" or "to host": it's already there!

## Striding Rules (MLX)

- MLX arrays are often views with non-trivial strides. Any direct pointer access via `array.data<T>()` MUST either:
  - prove `array.flags().row_contiguous`, or
  - iterate with shape+stride metadata (logical index order), not linear pointer arithmetic.
- `stablehlo.bitcast_convert` and `ToHostBuffer` are byte-boundary operations. They must pack bytes in logical element order using strides; never assume contiguous bytes unless `row_contiguous` is true.
- Do not add eager materialization (for example `zeros + x`) just to make data contiguous unless the API contract strictly requires contiguous output bytes.
- Normalize stride units carefully at boundaries: treat MLX strides as element strides unless proven otherwise, and convert only when required.
- Be careful with negative strides: never cast signed indices to `size_t` before applying bounds/offset logic.

# CI is Always Green

Tests, linting, and compilation on the `main` branch and in Continuous Integration testing ALWAYS pass. This means that any failures MUST be related to changes we made. You may NEVER claim that failures are "known issues," "unrelated to our changes," or similar.

# Adding New Ops

1. Identify the op to implement and find its StableHLO op name (e.g., `stablehlo.cosine`). The simplest approach is to implement a test for the op and look for failures (the error message includes the StableHLO op name).
2. Find the matching MLX free function (e.g., `mlx::core::cos`) by examining the installed `mlx` Python package, or (if present) a checked-out copy in `.agent-context/mlx/`.
3. Add a handler in `src/pjrt_plugin/mlx_executable.cc`:
   - For simple unary ops: add an `else if` branch in the interpreter calling the MLX free function directly.
   - For ops requiring type inspection, attribute access, or multi-step lowering: write a static helper function above the interpreter.
4. Rebuild with `uv pip install --python .venv -e .` and run `uv run pytest` to confirm the XFAILs become PASSes.

# Build and Test

```bash
uv sync --all-groups
uv pip install --python .venv -e .
uv run pytest
```

## Always Test Latest Built Artifact

- There may be multiple `libpjrt_plugin_mlx.dylib` files on disk (for example in UV caches). Always pin tests to the freshly built local artifact.
- After each rebuild, run tests with `JAX_MLX_LIBRARY_PATH` set to the newest build output:

```bash
LATEST_DYLIB="$(ls -t build/*/lib/libpjrt_plugin_mlx.dylib | head -n 1)"
JAX_MLX_LIBRARY_PATH="$LATEST_DYLIB" uv run pytest
```

- Use this same env var for focused test runs too (not just full-suite runs).

# Benchmarks

Benchmarks are excluded from normal test runs.

## Running and saving results

Use the benchmark script to run benchmarks and save a timestamped JSON file to
`.benchmarks/`:

```bash
bash scripts/benchmark.sh
# Saves to .benchmarks/<ISO8601>_<githash>[_dirty].json
```

Files with `_dirty` in the name were produced from an uncommitted working tree and
should not be used as baselines. Commit your changes before benchmarking to get a
clean baseline.

## Comparing results

After a refactor, run benchmarks again and compare to the previous clean baseline:

```bash
# Compare most recent file against the most recent clean baseline automatically:
uv run scripts/benchmark_compare.py

# Or name files explicitly:
uv run scripts/benchmark_compare.py .benchmarks/new.json .benchmarks/old.json

# Adjust significance threshold (default 1σ):
uv run scripts/benchmark_compare.py .benchmarks/new.json --threshold 2.0
```

The comparison reports which benchmarks changed by more than `threshold` standard
deviations of the baseline mean. A performance refactor is only confirmed to be
beneficial if benchmarks it was designed to improve show up in the FASTER column.
Anything in SLOWER is a regression and must be investigated.

## Running benchmarks directly (without saving)

```bash
# Run benchmarks (compares CPU vs MLX performance)
LATEST_DYLIB="$(ls -t build/*/lib/libpjrt_plugin_mlx.dylib | head -n 1)"
JAX_MLX_LIBRARY_PATH="$LATEST_DYLIB" uv run pytest -m benchmark --benchmark-only
```
