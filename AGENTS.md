# Guidelines

- You may NEVER skip or xfail tests without my explicit approval.
- You MUST use `uv` to manage dependencies.
- You MUST use `uv run ...` to execute commands.
- You MUST check the comprehensive list of MLX operations in `mps_ops/` before implementing a custom operation.
- You may NEVER use `--no-verify` for git commits.
- You may NEVER push to `main` unless explicitly requested.
- You may NEVER delete operations or tests without my explicit approval.
- For each op, you MUST register an `OperationTestConfig` for tests in `tests/test_ops.py`. See `tests/configs/unary.py` for an example and `tests/configs/util.py` for the signature of `OperationTestConfig`.
- The only target we care about for MLX is Apple Silicon, which is a unified-memory architecture. Don't copy memory "to device" or "to host": it's already there!

# CI is Always Green

Tests, linting, and compilation on the `main` branch and in Continuous Integration testing ALWAYS pass. This means that any failures MUST be related to changes we made. You may NEVER claim that failures are "known issues," "unrelated to our changes," or similar.

# Naming Conventions

- Handler functions MUST use PascalCase: `HandleSort`, `HandleTopK`, `HandleDotGeneral`

# Adding New Ops

1. Identify the op to implement and find its StableHLO op name (e.g., `stablehlo.cosine`). The simplest approach is to implement a test for the op and look for failures (the error message includes the StableHLO op name).
2. Find the matching MLX free function (e.g., `mlx::core::cos`). The `mps_ops/` directory lists legacy graph methods and is useful for discovering what operations are available and their semantics.
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
