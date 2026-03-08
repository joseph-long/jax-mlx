# jax-mlx: PJRT Plugin Backed by MLX

## Motivation

The current MPS Graph approach has fundamental brittleness:

- Apple's internal `WhileOpHandler` segfaults on `case → while → nested while` patterns
- `MPSGraph` rejects `complex<f32>` as a non-native type in `gather_nd` and other ops
- Native ops (Cholesky, triangular solve) require custom Metal shaders that bypass the
  graph compiler anyway, and the segmented execution system that interleaves them with
  graph segments is fragile and hard to maintain
- Linalg ops inlined by JAX (LU, etc.) involve complex gather/scatter patterns that
  trigger graph-mode limitations

MLX is Apple's own array framework (used internally for production ML training), ships
as a pip package with a C++ API, a shared library, and CMake config files. It targets
the same Metal GPU cores via its own optimized kernel library but sidesteps MPSGraph's
compiler bugs entirely. It also natively supports `complex64`, full linalg, and
arbitrary control flow.

## Architecture

```
JAX (Python)
    ↓ PJRT C API
PJRT Plugin (C++)   ← same API surface, no changes to JAX-facing code
    ↓ StableHLO MLIR
StableHLO → MLX Lowering   ← new layer
    ↓ mlx::core ops (lazy, batched)
MLX Runtime
    ↓ Metal / CPU (unified memory)
Apple Silicon
```

## Layer 1: PJRT C API (~80% reuse)

`pjrt_api.cc`, `pjrt_client.cc`, `pjrt_device.cc`, `pjrt_memory.cc`,
`pjrt_buffer.cc`, `pjrt_executable.cc`, `pjrt_event.cc`, `pjrt_topology.cc`

These files are mechanical PJRT wiring. The only changes are:
- Replace `jax_mps::MpsClient` → `jax_mlx::MlxClient` in `pjrt_types.h`
- Replace `jax_mps::MpsBuffer` → `jax_mlx::MlxBuffer`
- Replace `jax_mps::MpsExecutable` → `jax_mlx::MlxExecutable`
- Remove `metal_device()` guard (MLX handles GPU availability internally)
- All public function signatures and PJRT structs are unchanged

## Layer 2: Buffer (`mlx_buffer.h/.cc`)

Replaces `mps_buffer.h/.mm`. Wraps `mlx::core::array` instead of `id<MTLBuffer>`.

Key properties:
- **Unified memory**: no separate GPU/CPU copies, no explicit transfers
- Input: allocate a copy of the PJRT data, construct `mlx::core::array(ptr, shape, dtype, deleter)`
- Output: after `array_.eval()`, `array_.data<uint8_t>()` is a directly readable CPU pointer
- No ObjC — pure C++17

`MlxBuffer` exposes the same interface as `MpsBuffer`:
- `dtype()`, `dimensions()`, `byte_size()`, `element_count()`
- `ToHostBuffer(void* dst, std::function<void()> on_done)`
- `IsDeleted()`, `Delete()`

## Layer 3: Client (`mlx_client.h/.cc`)

Replaces `mps_client.h/.mm`. Initializes `mlx::core::Device::gpu` (or CPU fallback).
Creates one `MlxDevice` (id=0). Exposes `BufferFromHostBuffer` and `CompileStableHLO`.
No ObjC, no Metal framework link (MLX's dylib handles that internally).

## Layer 4: Executable (`mlx_executable.h/.cc`)

Replaces `mps_executable.h/.mm`. This is the core execution engine.

**Current model**: Build `MPSGraph`, hand to Apple's compiler, execute. Native ops
punch holes in the graph and are interleaved via `ExecutionPlan`. Apple's compiler
has bugs with nested control flow.

**New model**:
```cpp
// ValueMap: mlir::Value → mlx::core::array (lazy, not yet evaluated)
using ValueMap = std::unordered_map<void*, mlx::core::array>;

// Walk MLIR op-by-op, build up lazy MLX array graph
for (auto& op : entry_func_.getBody().front())
    dispatch(op, value_map);

// Single eval() dispatches everything to Metal via MLX
mlx::core::eval(output_arrays);
```

The `ExecutionPlan` with GRAPH/NATIVE segment interleaving is deleted.
Everything is a single-pass MLIR walk that accumulates lazy MLX arrays.

Wrap with `mlx::core::compile(fn)` for Metal kernel caching across calls.

## Layer 5: Op Registry (`mlx_registry.h`)

Replaces `ops/registry.h`. The `ValueMap` type changes:

```cpp
// OLD
using ValueMap = std::unordered_map<void*, MPSGraphTensor*>;
struct HandlerContext { MPSGraph* graph; mlir::Operation* op; ValueMap& values; };

// NEW
using ValueMap = std::unordered_map<void*, mlx::core::array>;
struct HandlerContext { mlir::Operation* op; ValueMap& values; };
```

The GRAPH/NATIVE handler distinction is deleted — there is only one handler kind.

Unary/binary macros become free-function calls:
```cpp
// OLD: REGISTER_MLIR_UNARY_OP("stablehlo.sine", "sinWithTensor:", Sin)
// NEW: REGISTER_MLIR_UNARY_OP("stablehlo.sine", mlx::core::sin, Sin)
```

## Layer 6: Op Handlers (`ops/*.cc`)

Replace `ops/*.mm`. Each file becomes pure C++. Most op bodies shrink to 1–3 lines.

Map of StableHLO → MLX equivalents (representative):

| StableHLO | MLX |
|---|---|
| `stablehlo.add` | `mlx::core::add` |
| `stablehlo.multiply` | `mlx::core::multiply` |
| `stablehlo.dot_general` | `mlx::core::matmul` / `tensordot` |
| `stablehlo.reduce` | `mlx::core::sum` / `max` / `min` / `prod` |
| `stablehlo.convolution` | `mlx::core::conv_general` |
| `stablehlo.sort` | `mlx::core::argsort` + `take` |
| `stablehlo.fft` | `mlx::core::fft::fft` |
| `stablehlo.cholesky` | `mlx::core::linalg::cholesky` |
| `stablehlo.triangular_solve` | `mlx::core::linalg::triangular_solve` |
| `stablehlo.gather` | `mlx::core::gather` |
| `stablehlo.scatter` | `mlx::core::scatter` |

Complex types (`complex64`) work natively — no custom shaders needed.

## Layer 7: Control Flow (big win)

Replaces `ops/control_flow_ops.h/.mm`.

```cpp
// stablehlo.while → literal C++ while loop
ProcessResult HandleWhile(HandlerContext& ctx) {
    // init carried values from operands
    while (true) {
        run_region(cond_region, carried_values);
        mlx::core::eval(cond_out);
        if (!cond_out.item<bool>()) break;
        run_region(body_region, carried_values);
    }
}

// stablehlo.if → C++ if
ProcessResult HandleIf(HandlerContext& ctx) {
    mlx::core::eval(pred);
    run_region(pred.item<bool>() ? true_region : false_region, operands);
}
```

No dependency on `MPSGraph`'s graph compiler for control flow. The nested
`while-inside-cond` segfault (issue #57) cannot occur. MLX's lazy eval still
batches and fuses ops within each loop body iteration.

## Build System

- Remove `OBJCXX` from `project(LANGUAGES ...)`
- Remove `Metal`, `MetalPerformanceShaders`, `MetalPerformanceShadersGraph`,
  `Foundation` framework links
- Add MLX cmake path from Python package:
  ```cmake
  execute_process(
      COMMAND ${Python3_EXECUTABLE} -c
          "import mlx.core, os; print(os.path.dirname(os.path.abspath(mlx.core.__file__)))"
      OUTPUT_VARIABLE MLX_PYTHON_DIR ...)
  list(APPEND CMAKE_PREFIX_PATH "${MLX_PYTHON_DIR}")
  find_package(MLX REQUIRED CONFIG)
  ```
- Link `mlx` instead of Apple frameworks
- All `.mm` sources become `.cc` (or new `.cc` files)

## What Transfers Unchanged

| Component | Reuse |
|---|---|
| PJRT C API boilerplate | ~80% |
| StableHLO MLIR parsing (`stablehlo_parser`) | ~95% |
| Type utilities (dtype mapping) | ~60% |
| Op registration pattern | ~90% |
| Test suite | 100% |
| CI / pre-commit / benchmarks | 100% |

## What Gets Rewritten

| Component | Notes |
|---|---|
| `mps_buffer` → `mlx_buffer` | Simpler — unified memory, no transfers |
| `mps_client` → `mlx_client` | Pure C++, no ObjC |
| `mps_executable` → `mlx_executable` | Drop ExecutionPlan; add compile() |
| `ops/registry.h` | Remove MPSGraph*, NATIVE kind |
| `ops/*.mm` → `ops/*.cc` | Mechanical; MLX API replaces MPSGraph ObjC |
| `control_flow_ops` | Much simpler; C++ loops replace graph-mode ops |
| `type_utils` | MLX dtypes instead of MPSDataType |
| `CMakeLists.txt` | Add MLX, remove ObjC/frameworks |

## Advantages Over Current Design

- **Complex types** work everywhere (`complex64` is native in MLX)
- **Control flow** works reliably — no Apple graph compiler bugs
- **Full linalg** via `mlx::core::linalg` (solve, inv, cholesky, svd, eig, etc.)
- **No ObjC** in the plugin source
- **Automatic kernel fusion** via `mlx::core::compile()`
- **Linux/CUDA** path for free once MLX's CUDA backend matures

---

## Rough Draft

Goal: get the plugin to compile, link, and load in JAX so that iteration on the
op layer can begin. Execution returns "not yet implemented" errors at this stage.

### Step 1: Add MLX to dependencies

Add `mlx>=0.31.0` to `pyproject.toml` `[project.dependencies]`.
MLX ships its C++ headers and `libmlx.dylib` inside the Python package, along with
a CMake config at `{site-packages}/mlx/share/cmake/MLX/MLXConfig.cmake`.

### Step 2: Create new MLX skeleton files

Create the following pure-C++ files with the minimum interface needed to
satisfy all PJRT boilerplate compilation:

- `src/pjrt_plugin/mlx_device.h` — `MlxDevice` with `id()`, `local_hardware_id()`, `debug_string()`
- `src/pjrt_plugin/mlx_client.h/.cc` — `MlxClient` wrapping `mlx::core::Device::gpu`;
  `BufferFromHostBuffer` copies data into an MLX array;
  `CompileStableHLO` stores the `ParsedModule`
- `src/pjrt_plugin/mlx_buffer.h/.cc` — `MlxBuffer` wrapping `mlx::core::array`;
  `ToHostBuffer` calls `eval()` then `memcpy`
- `src/pjrt_plugin/mlx_executable.h/.cc` — `MlxExecutable` stub;
  `IsValid()` true, `Execute()` returns `ExecutionResult::Error("not yet implemented")`
- `src/pjrt_plugin/mlx_type_utils.h/.cc` — `PjrtDtypeToMlx`, `MlxDtypeToPjrt`,
  `MlirTypeToPjrtDtype` (last one reused unchanged from old `type_utils`)

### Step 3: Create `stablehlo_parser.cc`

Copy `stablehlo_parser.mm` to `stablehlo_parser.cc`. Remove the
`#include "pjrt_plugin/ops/registry.h"` and make `checkUnsupportedOps` always
return `{}` — op support will be discovered at execution time, not compile time.

### Step 4: Update `pjrt_types.h`

Replace `mps_buffer.h / mps_client.h / mps_device.h / mps_executable.h` includes
with `mlx_buffer.h / mlx_client.h / mlx_device.h / mlx_executable.h`. Update the
`PJRT_Client`, `PJRT_Buffer`, and `PJRT_Executable` structs to use `jax_mlx::` types.

### Step 5: Patch `pjrt_client.cc` and `pjrt_executable.cc`

- Remove the two `metal_device()` guard checks in `pjrt_client.cc`
  (replace with simple `!client->client` null checks)
- Update `jax_mps::MpsBuffer*` and `jax_mps::MpsDevice*` references in
  `pjrt_executable.cc` to `jax_mlx::` equivalents
- Update the zero-sized tensor error message to remove MPS-specific wording

### Step 6: Update `CMakeLists.txt`

- Remove `OBJCXX` from `project(LANGUAGES ...)`
- Remove `CMAKE_OBJCXX_STANDARD` settings
- Remove `find_library` calls for Metal/MPS/MPSGraph/Foundation frameworks
- Add Python-based MLX prefix path discovery and `find_package(MLX REQUIRED CONFIG)`
- Replace all `.mm` source files in `PJRT_SOURCES` with the new `.cc` files
- Replace framework links with `mlx`

### Step 7: Build and test

```bash
uv pip install -e .
uv run pytest tests/ -x -q 2>&1 | head -60
```

Expected outcome: plugin loads, `jax.devices('mps')` returns a device, all tests
fail with "not yet implemented" rather than crashing. This is the baseline from
which op-by-op implementation begins.
