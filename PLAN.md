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

---

## Status

**Branch**: `claude-mps-to-mlx`

### Done

- [x] All MLX skeleton files: `mlx_buffer`, `mlx_client`, `mlx_device`, `mlx_executable`, `mlx_type_utils`
- [x] `stablehlo_parser.cc` (pure C++, op support discovered at execution time)
- [x] `pjrt_types.h` updated to use `jax_mlx::` types
- [x] All PJRT boilerplate updated (`pjrt_client.cc`, `pjrt_executable.cc`, etc.)
- [x] Build system: MLX from Python virtualenv, pure C++, no ObjC frameworks
- [x] Deps renamed: `jax-mps-deps` → `jax-mlx-deps` (setup_deps.sh, CMakeLists.txt, CI)
- [x] Namespace: `jax_mps` → `jax_mlx` in all actively compiled files
- [x] `issue_url.h` namespace updated to `jax_mlx`
- [x] StableHLO interpreter in `mlx_executable.cc` (~60 ops):
  - Elementwise: add, subtract, multiply, divide, power, remainder, max, min
  - Unary math: abs, neg, sign, exp, expm1, sqrt, rsqrt, cbrt, log, log1p, logistic, sin, cos, tan, tanh, floor, ceil, round, is_finite, real, imag, popcnt, count_leading_zeros, not
  - CHLO: asin, acos, atan, asinh, acosh, atanh, sinh, cosh, erf, erf_inv
  - Comparisons: eq, ne, lt, le, gt, ge
  - Shape: reshape, broadcast_in_dim, convert, transpose, iota, copy, concatenate, slice, dynamic_slice, dynamic_update_slice, pad
  - Reductions: sum, max, min, prod, any, all
  - Linear algebra: dot_general (via permute+reshape+matmul)
  - Bitwise: and, or, xor, shift_left, shift_right_arithmetic, shift_right_logical
  - Other: select, atan2, complex
  - Module: func.call (recursive)
- [x] `uv run pytest`: 96 tests passing, 226 skipped

### Remaining failures (1341 total)

| Error | Count | Notes |
|---|---|---|
| `stablehlo.custom_call` | ~1189 | PRNG, sorting keys, other JAX internals |
| `stablehlo.scatter` | ~56 | Index-update ops, scatter accumulation |
| `stablehlo.while` | unknown | Control flow (loops) |
| `stablehlo.if` | unknown | Control flow (conditionals) |
| Accuracy | ~15 | Numerical tolerance failures |

---

## Next Steps

### 1. `stablehlo.custom_call` (highest impact: ~1189 failures)

JAX dispatches PRNG (`threefry2x32`), sort-with-keys, and other ops as
`stablehlo.custom_call` with a `call_target_name` attribute.

Approach: inspect `call_target_name` and dispatch to MLX:

```cpp
else if (opName == "stablehlo.custom_call") {
    auto ccOp = mlir::cast<mlir::stablehlo::CustomCallOp>(op);
    auto target = ccOp.getCallTargetName();
    if (target == "mhlo.threefry2x32") {
        // Implement Threefry PRNG ...
    } else {
        return InterpResult::Error(jax_mlx::UnsupportedOpsMessage({target.str()}));
    }
}
```

Key targets to identify by running: `JAX_TRACEBACK_FILTERING=off uv run pytest tests/ -q 2>&1 | grep "custom_call" | sort -u`

### 2. `stablehlo.scatter` (~56 failures)

MLX has `mlx::core::scatter` but with a different signature than StableHLO scatter.
StableHLO scatter has a scatter body region (can be add, multiply, max, etc.).

Approach:
1. Inspect the scatter body region to determine the scatter kind (add, update, etc.)
2. Map to the appropriate `mlx::core::scatter` / `mlx::core::scatter_add` etc.

### 3. Control flow: `stablehlo.while` and `stablehlo.if`

These require running sub-regions of the MLIR. Implement by factoring out a
`runRegion(region, carried_values)` helper that calls `interpretFunction` on the
region's entry block:

```cpp
// stablehlo.while
else if (opName == "stablehlo.while") {
    auto whileOp = mlir::cast<mlir::stablehlo::WhileOp>(op);
    // carried_values initialized from operands
    while (true) {
        auto cond = interpretRegion(whileOp.getCond(), module, carried_values);
        mx::eval(cond.outputs);
        if (!cond.outputs[0].item<bool>()) break;
        auto body = interpretRegion(whileOp.getBody(), module, carried_values);
        carried_values = body.outputs;
    }
    // map while results to carried_values
}

// stablehlo.if
else if (opName == "stablehlo.if") {
    auto ifOp = mlir::cast<mlir::stablehlo::IfOp>(op);
    mx::eval({operand(0)});
    auto& branch = operand(0).item<bool>() ? ifOp.getTrueBranch()
                                            : ifOp.getFalseBranch();
    auto result = interpretRegion(branch, module, {});
    // map results
}
```

The key difference from `func.call`: regions share the parent block's `ValueMap`
(or operate on explicitly-passed `carried_values`), whereas `func.call` creates a
fresh scope.

### 4. Remove old MPS code

Once all tests pass on the MLX backend, the following files can be deleted:

- `src/pjrt_plugin/mps_buffer.h/.mm`
- `src/pjrt_plugin/mps_client.h/.mm`
- `src/pjrt_plugin/mps_device.h/.mm`
- `src/pjrt_plugin/mps_executable.h/.mm`
- `src/pjrt_plugin/ops/` (entire directory — all `.mm` files)

### 5. Performance: `mlx::core::compile()`

Wrap `interpretFunction` with `mlx::core::compile()` to cache Metal kernel
compilation across calls. This is the primary performance optimization — identical
to how `jax.jit` caches XLA compilation, but inside the MLX runtime layer.

```cpp
// In MlxExecutable::Execute, on first call:
compiled_fn_ = mlx::core::compile([this, device](const std::vector<mx::array>& inputs) {
    return interpretFunction(entry_func_, *module_, inputs).outputs;
});
```

### 6. Accuracy fixes

A small number of tests fail with numerical tolerance errors rather than missing ops.
These should be investigated after the above items are complete.
