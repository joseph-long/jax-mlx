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

# Benchmarks

Benchmarks are excluded from normal test runs. To run them:

```bash
# Run benchmarks (compares CPU vs MLX performance)
uv run pytest -m benchmark --benchmark-only
```
