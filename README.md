# jax-mlx 

<!-- [![GitHub Action Badge](https://github.com/tillahoffmann/jax-mlx/actions/workflows/build.yml/badge.svg)](https://github.com/tillahoffmann/jax-mlx/actions/workflows/build.yml) [![PyPI](https://img.shields.io/pypi/v/jax-mlx)](https://pypi.org/project/jax-mlx/) -->

A JAX backend piggybacking on implementations for Metal/Apple Silicon in Apple MLX, enabling GPU-accelerated JAX computations on Apple Silicon.

## Example

jax-mlx achieves a modest 3x speed-up (TODO: revise) over the CPU backend when training a simple ResNet18 model on CIFAR-10 using an M4 MacBook Air.

```bash
$ JAX_PLATFORMS=cpu uv run examples/resnet/main.py --steps=30
loss = 0.029: 100%|██████████| 30/30 [01:29<00:00,  2.99s/it]
Final training loss: 0.029
Time per step (second half): 3.041

$ JAX_PLATFORMS=mlx uv run examples/resnet/main.py --steps=30
WARNING:2026-01-26 17:32:53,989:jax._src.xla_bridge:905: Platform 'mlx' is experimental and not all JAX functionality may be correctly supported!
loss = 0.028: 100%|██████████| 30/30 [00:30<00:00,  1.03s/it]
Final training loss: 0.028
Time per step (second half): 0.991
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
