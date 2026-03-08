# Contributing

This guide walks through the development setup and the workflow for adding new operations.

## Setup

You need macOS on Apple Silicon, Python 3.13, and [uv](https://docs.astral.sh/uv/). Start by building the LLVM/MLIR and StableHLO dependencies. This is a one-time step and takes about 30 minutes.

```bash
brew install cmake ninja
./scripts/setup_deps.sh
```

Then install the Python dependencies, build the plugin, and set up pre-commit hooks:

```bash
uv sync --all-groups
uv pip install -e .
pre-commit install
```

Pre-commit hooks run clang-format, ruff, pyright, a rebuild, and the full test suite on every commit. Apple GPU execution is not available in GitHub Actions, so the pre-commit hooks are the primary line of defence — please do not skip them. This may seem pedantic (apologies), but agents need strong guardrails in the form of validation so they don't ... go off the rails.

## Adding a new operation

TODO: Revise for MLX

## Pull requests

Please open PRs against `main`. The pre-commit hooks ensure formatting, type checking, and tests all pass before a commit is created.
