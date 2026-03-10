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

To bias the suite toward larger, throughput-oriented shapes, set
`JAX_BENCH_PROFILE=throughput` (default: `default`):

```bash
# Example: larger shapes to make compute throughput dominate launch overhead
JAX_BENCH_PROFILE=throughput JAX_BENCH_ITERS=8 bash scripts/benchmark.sh
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

### Example benchmark output

```
% uv run python scripts/benchmark_compare.py /Users/jlong/devel/jax-mlx/.benchmarks/2026-03-10T17-50-23_2d92833.json .benchmarks/2026-03-10T18-29-40_27f5309_dirty.json
New:      2026-03-10T17-50-23_2d92833.json
Baseline: 2026-03-10T18-29-40_27f5309_dirty.json
Threshold: 2.0σ of baseline
Filter:   excluding CPU benchmark entries

────────────────────────────────────────────────────────────────────────
  FASTER (60 benchmarks)
────────────────────────────────────────────────────────────────────────
  test_benchmark_grad[mlx-benchmark.matmul_1000_grad0]
    ↓ 99.8%  (-142.2σ)              99.3 µs vs 51.3 ms baseline
  test_benchmark_grad[mlx-benchmark.conv2d_128ch_grad1]
    ↓ 100.0%  (-115.7σ)             796.7 ns vs 4.8 ms baseline
  test_benchmark_grad[mlx-benchmark.softmax_1000_grad0]
    ↓ 99.7%  (-85.3σ)               46.6 µs vs 15.9 ms baseline
  test_benchmark_grad[mlx-benchmark.conv2d_64ch_grad1]
    ↓ 100.0%  (-59.3σ)              473.4 ns vs 1.7 ms baseline
  test_benchmark_grad[mlx-benchmark.layernorm_1024_grad0]
    ↓ 99.8%  (-54.1σ)               27.9 µs vs 12.5 ms baseline
  test_benchmark_value[mlx-benchmark.layernorm_512]
    ↓ 99.7%  (-52.6σ)               7.3 µs vs 2.3 ms baseline
  test_benchmark_grad[mlx-benchmark.conv2d_32ch_grad1]
    ↓ 100.0%  (-48.9σ)              278.5 ns vs 1.5 ms baseline
  test_benchmark_grad[mlx-benchmark.matmul_1000_grad1]
    ↓ 99.8%  (-48.0σ)               89.6 µs vs 46.8 ms baseline
  test_benchmark_value[mlx-benchmark.layernorm_256]
    ↓ 99.7%  (-30.9σ)               4.4 µs vs 1.2 ms baseline
  test_benchmark_grad[mlx-benchmark.layernorm_512_grad0]
    ↓ 99.8%  (-29.5σ)               10.1 µs vs 6.6 ms baseline
  test_benchmark_grad[mlx-benchmark.add_1_grad1]
    ↓ 85.1%  (-28.9σ)               626.0 ns vs 4.2 µs baseline
  test_benchmark_grad[mlx-benchmark.conv2d_32ch_grad0]
    ↓ 99.6%  (-24.6σ)               1.3 µs vs 321.6 µs baseline
  test_benchmark_grad[mlx-benchmark.matmul_100_grad1]
    ↓ 99.7%  (-23.5σ)               758.9 ns vs 291.1 µs baseline
  test_benchmark_grad[mlx-benchmark.layernorm_256_grad0]
    ↓ 99.9%  (-23.3σ)               4.6 µs vs 3.2 ms baseline
  test_benchmark_value[mlx-benchmark.conv2d_128ch]
    ↓ 99.7%  (-23.1σ)               4.2 µs vs 1.2 ms baseline
  test_benchmark_value[mlx-benchmark.matmul_1000]
    ↓ 99.9%  (-21.9σ)               56.6 µs vs 47.6 ms baseline
  test_benchmark_grad[mlx-benchmark.conv2d_64ch_grad0]
    ↓ 99.6%  (-21.7σ)               2.2 µs vs 501.2 µs baseline
  test_benchmark_grad[mlx-benchmark.matmul_100_grad0]
    ↓ 99.7%  (-20.6σ)               955.8 ns vs 303.7 µs baseline
  test_benchmark_grad[mlx-benchmark.exp_100_grad0]
    ↓ 99.2%  (-17.6σ)               4.2 µs vs 536.8 µs baseline
  test_benchmark_grad[mlx-benchmark.matmul_batched_128_grad0]
    ↓ 99.2%  (-17.3σ)               2.2 µs vs 291.8 µs baseline
  test_benchmark_grad[mlx-benchmark.matmul_batched_32_grad1]
    ↓ 99.6%  (-16.0σ)               828.2 ns vs 230.0 µs baseline
  test_benchmark_grad[mlx-benchmark.matmul_batched_32_grad0]
    ↓ 99.7%  (-15.8σ)               749.1 ns vs 245.2 µs baseline
  test_benchmark_value[mlx-benchmark.softmax_1000]
    ↓ 99.3%  (-14.4σ)               47.2 µs vs 7.2 ms baseline
  test_benchmark_grad[mlx-benchmark.matmul_batched_8_grad1]
    ↓ 99.8%  (-14.1σ)               405.3 ns vs 220.4 µs baseline
  test_benchmark_grad[mlx-benchmark.matmul_1_grad0]
    ↓ 99.9%  (-14.0σ)               204.6 ns vs 198.2 µs baseline
  test_benchmark_grad[mlx-benchmark.matmul_10_grad0]
    ↓ 99.9%  (-13.8σ)               199.5 ns vs 197.2 µs baseline
  test_benchmark_grad[mlx-benchmark.exp_10_grad0]
    ↓ 99.7%  (-13.8σ)               600.8 ns vs 205.8 µs baseline
  test_benchmark_grad[mlx-benchmark.matmul_batched_8_grad0]
    ↓ 99.8%  (-12.8σ)               419.9 ns vs 211.3 µs baseline
  test_benchmark_grad[mlx-benchmark.matmul_1_grad1]
    ↓ 99.9%  (-12.7σ)               204.0 ns vs 199.5 µs baseline
  test_benchmark_value[mlx-benchmark.layernorm_1024]
    ↓ 99.6%  (-12.6σ)               16.1 µs vs 4.4 ms baseline
  test_benchmark_grad[mlx-benchmark.matmul_10_grad1]
    ↓ 99.9%  (-12.2σ)               199.4 ns vs 207.0 µs baseline
  test_benchmark_grad[mlx-benchmark.conv2d_128ch_grad0]
    ↓ 99.7%  (-11.2σ)               4.8 µs vs 1.4 ms baseline
  test_benchmark_grad[mlx-benchmark.exp_1_grad0]
    ↓ 99.8%  (-10.3σ)               270.8 ns vs 173.2 µs baseline
  test_benchmark_grad[mlx-benchmark.softmax_1_grad0]
    ↓ 99.9%  (-9.4σ)                254.9 ns vs 272.6 µs baseline
  test_benchmark_grad[mlx-benchmark.softmax_100_grad0]
    ↓ 99.7%  (-8.9σ)                4.2 µs vs 1.6 ms baseline
  test_benchmark_grad[mlx-benchmark.exp_1000_grad0]
    ↓ 98.7%  (-8.5σ)                46.3 µs vs 3.6 ms baseline
  test_benchmark_grad[mlx-benchmark.add_1_grad0]
    ↓ 85.3%  (-7.5σ)                631.0 ns vs 4.3 µs baseline
  test_benchmark_grad[mlx-benchmark.softmax_10_grad0]
    ↓ 99.8%  (-7.3σ)                597.5 ns vs 337.8 µs baseline
  test_benchmark_grad[mlx-benchmark.matmul_batched_128_grad1]
    ↓ 99.1%  (-7.2σ)                2.6 µs vs 301.8 µs baseline
  test_benchmark_value[mlx-benchmark.add_1000]
    ↓ 97.1%  (-7.0σ)                46.3 µs vs 1.6 ms baseline
  test_benchmark_value[mlx-benchmark.softmax_1]
    ↓ 99.9%  (-5.9σ)                208.6 ns vs 248.2 µs baseline
  test_benchmark_grad[mlx-benchmark.add_10_grad0]
    ↓ 55.3%  (-5.8σ)                3.7 µs vs 8.4 µs baseline
  test_benchmark_grad[mlx-benchmark.sum_10_grad0]
    ↓ 50.4%  (-5.2σ)                3.7 µs vs 7.5 µs baseline
  test_benchmark_value[mlx-benchmark.sum_1000]
    ↓ 100.0%  (-5.1σ)               138.8 ns vs 724.8 µs baseline
  test_benchmark_value[mlx-benchmark.softmax_100]
    ↓ 99.5%  (-4.8σ)                4.0 µs vs 846.5 µs baseline
  test_benchmark_grad[mlx-benchmark.sum_100_grad0]
    ↓ 11.4%  (-4.7σ)                56.0 µs vs 63.2 µs baseline
  test_benchmark_grad[mlx-benchmark.add_10_grad1]
    ↓ 52.1%  (-4.4σ)                3.8 µs vs 7.9 µs baseline
  test_benchmark_value[mlx-benchmark.matmul_100]
    ↓ 99.8%  (-3.8σ)                580.1 ns vs 305.6 µs baseline
  test_benchmark_value[mlx-benchmark.sum_1]
    ↓ 99.9%  (-3.7σ)                132.4 ns vs 171.3 µs baseline
  test_benchmark_value[mlx-benchmark.exp_1000]
    ↓ 96.0%  (-3.7σ)                47.6 µs vs 1.2 ms baseline
  test_benchmark_value[mlx-benchmark.add_1]
    ↓ 99.8%  (-3.1σ)                205.5 ns vs 130.3 µs baseline
  test_benchmark_value[mlx-benchmark.sum_100]
    ↓ 99.9%  (-3.0σ)                129.4 ns vs 254.2 µs baseline
  test_benchmark_value[mlx-benchmark.matmul_1]
    ↓ 99.9%  (-2.9σ)                134.5 ns vs 185.1 µs baseline
  test_benchmark_value[mlx-benchmark.matmul_10]
    ↓ 99.9%  (-2.8σ)                134.6 ns vs 190.9 µs baseline
  test_benchmark_value[mlx-benchmark.add_100]
    ↓ 98.9%  (-2.7σ)                4.1 µs vs 367.8 µs baseline
  test_benchmark_value[mlx-benchmark.exp_100]
    ↓ 98.6%  (-2.6σ)                4.2 µs vs 295.0 µs baseline
  test_benchmark_value[mlx-benchmark.conv2d_64ch]
    ↓ 99.5%  (-2.5σ)                2.2 µs vs 474.5 µs baseline
  test_benchmark_value[mlx-benchmark.exp_10]
    ↓ 99.7%  (-2.5σ)                534.8 ns vs 175.3 µs baseline
  test_benchmark_grad[mlx-benchmark.add_100_grad1]
    ↓ 7.0%  (-2.1σ)                 58.6 µs vs 63.1 µs baseline
  test_benchmark_value[mlx-benchmark.softmax_10]
    ↓ 99.8%  (-2.0σ)                528.9 ns vs 276.5 µs baseline

────────────────────────────────────────────────────────────────────────
  SLOWER (1 benchmark)
────────────────────────────────────────────────────────────────────────
  test_benchmark_grad[mlx-benchmark.add_100_grad0]
    ↑ 13.0%  (+3.0σ)                69.0 µs vs 61.1 µs baseline

────────────────────────────────────────────────────────────────────────
  72 benchmarks compared  |  60 faster  |  1 slower  |  11 unchanged

────────────────────────────────────────────────────────────────────────
  MLX VS CPU (same run; speedup = CPU mean / MLX mean) (72 benchmarks)
────────────────────────────────────────────────────────────────────────
  test_benchmark_value[mlx-benchmark.add_1000]
    1.32x faster        MLX 46.3 µs  |  CPU 61.3 µs
  test_benchmark_value[mlx-benchmark.sum_100]
    1.13x faster        MLX 129.4 ns  |  CPU 146.0 ns
  test_benchmark_value[mlx-benchmark.add_100]
    1.11x faster        MLX 4.1 µs  |  CPU 4.6 µs
  test_benchmark_value[mlx-benchmark.layernorm_512]
    1.07x faster        MLX 7.3 µs  |  CPU 7.8 µs
  test_benchmark_value[mlx-benchmark.exp_1000]
    1.07x faster        MLX 47.6 µs  |  CPU 50.7 µs
  test_benchmark_grad[mlx-benchmark.sum_10_grad0]
    1.06x faster        MLX 3.7 µs  |  CPU 4.0 µs
  test_benchmark_value[mlx-benchmark.layernorm_1024]
    1.05x faster        MLX 16.1 µs  |  CPU 16.9 µs
  test_benchmark_grad[mlx-benchmark.sum_100_grad0]
    1.04x faster        MLX 56.0 µs  |  CPU 58.4 µs
  test_benchmark_value[mlx-benchmark.matmul_100]
    1.03x faster        MLX 580.1 ns  |  CPU 598.8 ns
  test_benchmark_grad[mlx-benchmark.matmul_1000_grad0]
    1.03x faster        MLX 99.3 µs  |  CPU 102.4 µs
  test_benchmark_grad[mlx-benchmark.softmax_10_grad0]
    1.03x faster        MLX 597.5 ns  |  CPU 614.5 ns
  test_benchmark_value[mlx-benchmark.conv2d_128ch]
    1.02x faster        MLX 4.2 µs  |  CPU 4.3 µs
  test_benchmark_grad[mlx-benchmark.matmul_100_grad1]
    1.02x faster        MLX 758.9 ns  |  CPU 772.6 ns
  test_benchmark_value[mlx-benchmark.exp_100]
    1.02x faster        MLX 4.2 µs  |  CPU 4.3 µs
  test_benchmark_value[mlx-benchmark.matmul_10]
    1.02x faster        MLX 134.6 ns  |  CPU 136.7 ns
  test_benchmark_grad[mlx-benchmark.matmul_10_grad1]
    1.02x faster        MLX 199.4 ns  |  CPU 202.5 ns
  test_benchmark_grad[mlx-benchmark.add_10_grad0]
    1.01x faster        MLX 3.7 µs  |  CPU 3.8 µs
  test_benchmark_value[mlx-benchmark.matmul_batched_128]
    1.01x faster        MLX 2.1 µs  |  CPU 2.1 µs
  test_benchmark_value[mlx-benchmark.matmul_batched_8]
    1.01x faster        MLX 292.5 ns  |  CPU 294.9 ns
  test_benchmark_grad[mlx-benchmark.exp_1000_grad0]
    1.01x faster        MLX 46.3 µs  |  CPU 46.6 µs
  test_benchmark_grad[mlx-benchmark.matmul_10_grad0]
    1.01x faster        MLX 199.5 ns  |  CPU 200.8 ns
  test_benchmark_value[mlx-benchmark.layernorm_256]
    1.00x faster        MLX 4.4 µs  |  CPU 4.4 µs
  test_benchmark_grad[mlx-benchmark.exp_100_grad0]
    1.00x faster        MLX 4.2 µs  |  CPU 4.2 µs
  test_benchmark_value[mlx-benchmark.softmax_100]
    1.00x faster        MLX 4.0 µs  |  CPU 4.1 µs
  test_benchmark_grad[mlx-benchmark.add_1000_grad0]
    1.00x faster        MLX 868.7 µs  |  CPU 871.0 µs
  test_benchmark_value[mlx-benchmark.softmax_1000]
    1.00x faster        MLX 47.2 µs  |  CPU 47.3 µs
  test_benchmark_grad[mlx-benchmark.softmax_100_grad0]
    1.00x faster        MLX 4.2 µs  |  CPU 4.2 µs
  test_benchmark_value[mlx-benchmark.add_10]
    1.00x slower        MLX 526.6 ns  |  CPU 526.1 ns
  test_benchmark_value[mlx-benchmark.exp_10]
    1.00x slower        MLX 534.8 ns  |  CPU 533.5 ns
  test_benchmark_grad[mlx-benchmark.matmul_1_grad0]
    1.00x slower        MLX 204.6 ns  |  CPU 204.0 ns
  test_benchmark_grad[mlx-benchmark.matmul_batched_128_grad0]
    1.00x slower        MLX 2.2 µs  |  CPU 2.2 µs
  test_benchmark_value[mlx-benchmark.exp_1]
    1.01x slower        MLX 204.2 ns  |  CPU 203.0 ns
  test_benchmark_grad[mlx-benchmark.exp_10_grad0]
    1.01x slower        MLX 600.8 ns  |  CPU 596.7 ns
  test_benchmark_grad[mlx-benchmark.softmax_1000_grad0]
    1.01x slower        MLX 46.6 µs  |  CPU 46.2 µs
  test_benchmark_grad[mlx-benchmark.softmax_1_grad0]
    1.01x slower        MLX 254.9 ns  |  CPU 252.6 ns
  test_benchmark_value[mlx-benchmark.matmul_1000]
    1.01x slower        MLX 56.6 µs  |  CPU 56.0 µs
  test_benchmark_value[mlx-benchmark.sum_1]
    1.01x slower        MLX 132.4 ns  |  CPU 130.6 ns
  test_benchmark_value[mlx-benchmark.conv2d_32ch]
    1.01x slower        MLX 1.1 µs  |  CPU 1.1 µs
  test_benchmark_grad[mlx-benchmark.matmul_100_grad0]
    1.02x slower        MLX 955.8 ns  |  CPU 940.1 ns
  test_benchmark_grad[mlx-benchmark.add_1_grad1]
    1.02x slower        MLX 626.0 ns  |  CPU 614.3 ns
  test_benchmark_grad[mlx-benchmark.matmul_batched_8_grad0]
    1.02x slower        MLX 419.9 ns  |  CPU 411.8 ns
  test_benchmark_value[mlx-benchmark.sum_10]
    1.02x slower        MLX 130.2 ns  |  CPU 127.2 ns
  test_benchmark_grad[mlx-benchmark.add_1_grad0]
    1.02x slower        MLX 631.0 ns  |  CPU 616.4 ns
  test_benchmark_value[mlx-benchmark.conv2d_64ch]
    1.02x slower        MLX 2.2 µs  |  CPU 2.1 µs
  test_benchmark_value[mlx-benchmark.matmul_1]
    1.03x slower        MLX 134.5 ns  |  CPU 130.8 ns
  test_benchmark_value[mlx-benchmark.softmax_1]
    1.03x slower        MLX 208.6 ns  |  CPU 202.0 ns
  test_benchmark_grad[mlx-benchmark.conv2d_64ch_grad0]
    1.04x slower        MLX 2.2 µs  |  CPU 2.2 µs
  test_benchmark_grad[mlx-benchmark.sum_1_grad0]
    1.04x slower        MLX 626.7 ns  |  CPU 605.0 ns
  test_benchmark_grad[mlx-benchmark.matmul_batched_32_grad0]
    1.04x slower        MLX 749.1 ns  |  CPU 721.9 ns
  test_benchmark_value[mlx-benchmark.matmul_batched_32]
    1.04x slower        MLX 639.2 ns  |  CPU 614.4 ns
  test_benchmark_value[mlx-benchmark.softmax_10]
    1.04x slower        MLX 528.9 ns  |  CPU 508.1 ns
  test_benchmark_grad[mlx-benchmark.exp_1_grad0]
    1.04x slower        MLX 270.8 ns  |  CPU 259.7 ns
  test_benchmark_grad[mlx-benchmark.add_100_grad1]
    1.05x slower        MLX 58.6 µs  |  CPU 56.0 µs
  test_benchmark_value[mlx-benchmark.add_1]
    1.05x slower        MLX 205.5 ns  |  CPU 195.6 ns
  test_benchmark_grad[mlx-benchmark.conv2d_32ch_grad0]
    1.06x slower        MLX 1.3 µs  |  CPU 1.2 µs
  test_benchmark_grad[mlx-benchmark.matmul_batched_8_grad1]
    1.06x slower        MLX 405.3 ns  |  CPU 381.9 ns
  test_benchmark_grad[mlx-benchmark.conv2d_128ch_grad1]
    1.06x slower        MLX 796.7 ns  |  CPU 750.3 ns
  test_benchmark_grad[mlx-benchmark.layernorm_256_grad0]
    1.07x slower        MLX 4.6 µs  |  CPU 4.3 µs
  test_benchmark_grad[mlx-benchmark.add_10_grad1]
    1.07x slower        MLX 3.8 µs  |  CPU 3.6 µs
  test_benchmark_value[mlx-benchmark.sum_1000]
    1.07x slower        MLX 138.8 ns  |  CPU 129.6 ns
  test_benchmark_grad[mlx-benchmark.conv2d_32ch_grad1]
    1.07x slower        MLX 278.5 ns  |  CPU 259.5 ns
  test_benchmark_grad[mlx-benchmark.matmul_1_grad1]
    1.09x slower        MLX 204.0 ns  |  CPU 186.8 ns
  test_benchmark_grad[mlx-benchmark.conv2d_128ch_grad0]
    1.13x slower        MLX 4.8 µs  |  CPU 4.3 µs
  test_benchmark_grad[mlx-benchmark.matmul_batched_32_grad1]
    1.16x slower        MLX 828.2 ns  |  CPU 713.5 ns
  test_benchmark_grad[mlx-benchmark.sum_1000_grad0]
    1.17x slower        MLX 851.0 µs  |  CPU 725.9 µs
  test_benchmark_grad[mlx-benchmark.matmul_batched_128_grad1]
    1.19x slower        MLX 2.6 µs  |  CPU 2.2 µs
  test_benchmark_grad[mlx-benchmark.conv2d_64ch_grad1]
    1.23x slower        MLX 473.4 ns  |  CPU 384.7 ns
  test_benchmark_grad[mlx-benchmark.matmul_1000_grad1]
    1.26x slower        MLX 89.6 µs  |  CPU 71.0 µs
  test_benchmark_grad[mlx-benchmark.add_100_grad0]
    1.31x slower        MLX 69.0 µs  |  CPU 52.7 µs
  test_benchmark_grad[mlx-benchmark.layernorm_512_grad0]
    1.37x slower        MLX 10.1 µs  |  CPU 7.3 µs
  test_benchmark_grad[mlx-benchmark.layernorm_1024_grad0]
    1.44x slower        MLX 27.9 µs  |  CPU 19.3 µs
  test_benchmark_grad[mlx-benchmark.add_1000_grad1]
    1.49x slower        MLX 874.9 µs  |  CPU 586.0 µs
```
