# Contributing

This guide walks through the development setup and the workflow for adding new operations.

## Setup

You need macOS on Apple Silicon, Python 3.13, and [uv](https://docs.astral.sh/uv/). Start by building the LLVM/MLIR and StableHLO dependencies. This is a one-time step and takes about 30 minutes.

```bash
brew install cmake ninja
./scripts/setup_deps.sh
```

> If you previously built deps for the old `jax-mps` project (at `~/.local/jax-mps-deps`), you can symlink them rather than rebuilding: `ln -sf ~/.local/jax-mps-deps ~/.local/jax-mlx-deps`

Then install the Python dependencies, build the plugin, and set up pre-commit hooks:

```bash
uv sync --all-groups
uv pip install --python .venv -e .
pre-commit install
```

> If you have a conda/mamba environment active, the plain `uv pip install -e .` command will install into that environment instead of the project's `.venv`. Using `--python .venv` avoids this. Alternatively, deactivate the environment first.

Pre-commit hooks run clang-format, ruff, pyright, a rebuild, and the full test suite on every commit. Apple GPU execution is not available in GitHub Actions, so the pre-commit hooks are the primary line of defence — please do not skip them. This may seem pedantic (apologies), but agents need strong guardrails in the form of validation so they don't ... go off the rails.

## Adding a new operation

jax-mlx implements a StableHLO → MLX interpreter in `src/pjrt_plugin/mlx_executable.cc`. Each StableHLO op is dispatched to a corresponding `mlx::core` free function.

### Finding the op name

The easiest way to find an op's name is to write a test and look at the error message:

```
Unsupported op: stablehlo.cosine
```

### Finding the MLX implementation

Check `mlx::core` free functions in the MLX headers. The `mps_ops/` directory lists MPS Graph methods and is useful for discovering what operations are available and their semantics.

### Adding the handler

Open `src/pjrt_plugin/mlx_executable.cc` and find the interpreter's op dispatch block. For a simple unary op:

```cpp
else if (opName == "stablehlo.cosine") set(0, mx::cos(operand(0)));
```

For ops that need attribute access or multi-step lowering, write a static helper above `runFunction`:

```cpp
static mx::array myHelper(mx::array a, ...) { ... }
// then in the dispatch block:
else if (opName == "stablehlo.my_op") {
    auto myOp = mlir::cast<mlir::stablehlo::MyOp>(op);
    set(0, myHelper(operand(0), myOp.getSomeAttr()));
}
```

### Registering a test

Add an `OperationTestConfig` in the appropriate file under `tests/configs/`. See `tests/configs/unary.py` for examples and `tests/configs/util.py` for the `OperationTestConfig` signature.

### Rebuild and verify

```bash
uv pip install -e .
uv run pytest
```

The test for the new op should move from XFAIL to PASS.

## Pull requests

Please open PRs against `main`. The pre-commit hooks ensure formatting, type checking, and tests all pass before a commit is created.
