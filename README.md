# jax-mlx 

<!-- [![GitHub Action Badge](https://github.com/tillahoffmann/jax-mlx/actions/workflows/build.yml/badge.svg)](https://github.com/tillahoffmann/jax-mlx/actions/workflows/build.yml) [![PyPI](https://img.shields.io/pypi/v/jax-mlx)](https://pypi.org/project/jax-mlx/) -->

A JAX backend piggybacking on implementations for Metal/Apple Silicon in Apple MLX, enabling GPU-accelerated JAX computations on Apple Silicon.

## Example

jax-mlx achieves a modest 3x speed-up (TODO: revise) over the CPU backend when training a simple ResNet18 model on CIFAR-10 using an M4 MacBook Air.

```bash
% JAX_PLATFORMS=cpu uv run examples/resnet/main.py --steps=30
JAX devices: [CpuDevice(id=0), CpuDevice(id=1), CpuDevice(id=2), CpuDevice(id=3), CpuDevice(id=4), CpuDevice(id=5), CpuDevice(id=6), CpuDevice(id=7), CpuDevice(id=8), CpuDevice(id=9)]
Loading CIFAR-10...
Loaded 50,000 training samples
Preparing 195 batches on device...
Starting training for 30 steps ...
loss = 0.029: 100%|████████████████████████████████████████████████████████████████████████████████████████| 30/30 [01:55<00:00,  3.86s/it]
Final training loss: 0.029
Time per step (second half): 4.457

% JAX_PLATFORMS=mlx uv run examples/resnet/main.py --steps=30
JAX devices: [MlxDevice(id=0)]
Loading CIFAR-10...
Loaded 50,000 training samples
Preparing 195 batches on device...
Starting training for 30 steps ...
loss = 0.018: 100%|████████████████████████████████████████████████████████████████████████████████████████| 30/30 [00:40<00:00,  1.35s/it]
Final training loss: 0.018
Time per step (second half): 1.336

```

## Installation

jax-mlx requires macOS on Apple Silicon and Python 3.13. Install it with pip:

```bash
pip install -e .
```

The plugin registers itself with JAX automatically and is enabled by default. Set `JAX_PLATFORMS=mlx` to select the MLX backend explicitly.

jax-mlx is built against the StableHLO bytecode format matching jaxlib 0.9.x. Using a different jaxlib version will likely cause deserialization failures at JIT compile time. See [Version Pinning](#version-pinning) for details.

## Architecture

This project implements a [PJRT plugin](https://openxla.org/xla/pjrt) to dispatch StableHLO operations to the C++ MLX implementations.

## Building

1. Install build tools and build and install LLVM/MLIR & StableHLO. This is a one-time setup and takes about 30 minutes. See the `setup_deps.sh` script for further options, such as forced re-installation, installation location, etc. The script pins LLVM and StableHLO to specific commits matching jaxlib 0.9.0 for bytecode compatibility (see the section on [Version Pinning](#version-pinning)) for details.

```bash
$ brew install cmake ninja
$ ./scripts/setup_deps.sh
```

2. Build the plugin and install it as a Python package. This step should be fast, and MUST be repeated for all changes to C++ files.

```bash
$ uv pip install -e .
```

### Version Pinning

The script pins LLVM and StableHLO to specific commits matching jaxlib 0.9.0 for bytecode compatibility. To update these versions for a different jaxlib release, trace the dependency chain:

```bash
# 1. Find XLA commit used by jaxlib
curl -s https://raw.githubusercontent.com/jax-ml/jax/jax-v0.9.0/third_party/xla/revision.bzl
# → XLA_COMMIT = "bb760b04..."

# 2. Find LLVM commit used by that XLA version
curl -s https://raw.githubusercontent.com/openxla/xla/<XLA_COMMIT>/third_party/llvm/workspace.bzl
# → LLVM_COMMIT = "f6d0a512..."

# 3. Find StableHLO commit used by that XLA version
curl -s https://raw.githubusercontent.com/openxla/xla/<XLA_COMMIT>/third_party/stablehlo/workspace.bzl
# → STABLEHLO_COMMIT = "127d2f23..."
```

Then update the `STABLEHLO_COMMIT` and `LLVM_COMMIT_OVERRIDE` variables in `setup_deps.sh`.

## Project Structure

```
jax-mlx/
├── CMakeLists.txt
├── src/
│   ├── jax_plugins/mlx/         # Python JAX plugin
│   ├── pjrt_plugin/             # C++ PJRT implementation
│   │   ├── pjrt_api.cc          # PJRT C API entry point
│   │   ├── mlx_client.h/mm      # Metal client management
│   │   ├── mlx_executable.h/mm  # StableHLO compilation & execution
│   │   └── ops/                 # Operation implementations
│   └── proto/                   # Protobuf definitions
└── tests/
```

## How It Works

### PJRT Plugin

PJRT (Portable JAX Runtime) is JAX's abstraction for hardware backends. The plugin implements:

- `PJRT_Client_Create` - Initialize Metal device
- `PJRT_Client_Compile` - Parse HLO and prepare MLXGraph
- `PJRT_Client_BufferFromHostBuffer` - Transfer data to GPU
- `PJRT_LoadedExecutable_Execute` - Run computation on GPU

## Testing

### In-tree tests

```bash
uv run pytest          # fast correctness suite (~1430 tests)
```

### JAX upstream test suite

JAX's own test suite provides much broader op coverage.  The JAX source is
checked out at `.agent-context/jax/`.  To run it:

```bash
# Pull latest JAX main and run the four core test files:
bash scripts/jax_tests.sh -q --tb=short

# Run only one file (faster):
JAX_TEST_FILES="tests/lax_numpy_test.py" bash scripts/jax_tests.sh -q --tb=no

# Skip the git pull:
NO_PULL=1 bash scripts/jax_tests.sh -q --tb=no
```

The script automatically sets `JAX_PLATFORMS=mlx,cpu` and uses the freshest built
dylib.  See `AGENTS.md` for the current pass-rate breakdown and top failure
categories.

## Benchmarking

Microbenchmarks compare CPU and MLX performance across a range of ops and sizes.

### Running benchmarks

```bash
# Run and save a timestamped result file to .benchmarks/:
bash scripts/benchmark.sh

# Run without saving (quick look):
LATEST_DYLIB="$(ls -t build/*/lib/libpjrt_plugin_mlx.dylib | head -n 1)"
JAX_MLX_LIBRARY_PATH="$LATEST_DYLIB" uv run pytest -m benchmark --benchmark-only
```

To amortize per-call overhead during benchmark timing, set `JAX_BENCH_ITERS`
(default: `16`). This repeats each benchmarked operation `N` times per timed
call on both CPU and MLX:

```bash
# Example: use 8 amortized iterations per timed benchmark call
JAX_BENCH_ITERS=8 bash scripts/benchmark.sh
```

Benchmark comparison normalizes results back to per-iteration units using the
recorded amortization factor, so files with the same `JAX_BENCH_ITERS` setting
remain directly comparable.

### Comparing before/after a change

```bash
# Save a clean baseline before your change, then after:
git stash           # or commit first
bash scripts/benchmark.sh                        # saves .benchmarks/<ts>_<hash>.json
git stash pop       # apply your change
uv pip install -e . # rebuild
bash scripts/benchmark.sh                        # saves a new result

# Compare the two (auto-selects most recent vs oldest clean baseline):
uv run scripts/benchmark_compare.py
```

By default, comparison:
- excludes CPU benchmark entries from baseline delta reporting,
- uses a `2σ` significance threshold,
- and includes a final section showing MLX-vs-CPU speedup from the same run.
