# jax-mlx: PJRT Plugin Backed by MLX

## Motivation

The previous graph-compiler approach had fundamental brittleness:

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
**Last updated**: March 8, 2026

### Done

- [x] All MLX skeleton files: `mlx_buffer`, `mlx_client`, `mlx_device`, `mlx_executable`, `mlx_type_utils`
- [x] `stablehlo_parser.cc` (pure C++, op support discovered at execution time)
- [x] `pjrt_types.h` updated to use `jax_mlx::` types
- [x] All PJRT boilerplate updated (`pjrt_client.cc`, `pjrt_executable.cc`, etc.)
- [x] Build system: MLX from Python virtualenv, pure C++, no ObjC frameworks
- [x] Deps path: `~/.local/jax-mlx-deps` (setup_deps.sh, CMakeLists.txt, CI)
- [x] Namespace: `jax_mps` → `jax_mlx` in all actively compiled files
- [x] `issue_url.h` namespace updated to `jax_mlx`
- [x] StableHLO interpreter in `mlx_executable.cc` substantially expanded:
  - Elementwise: add, subtract, multiply, divide, power, remainder, max, min
  - Unary math: abs, neg, sign, exp, expm1, sqrt, rsqrt, cbrt, log, log1p, logistic, sin, cos, tan, tanh, floor, ceil, round, is_finite, real, imag, popcnt, count_leading_zeros, not
  - CHLO: asin, acos, atan, asinh, acosh, atanh, sinh, cosh, erf, erf_inv
  - Comparisons: eq, ne, lt, le, gt, ge
  - Shape: reshape, broadcast_in_dim, convert, transpose, iota, copy, concatenate, slice, dynamic_slice, dynamic_update_slice (via `slice_update`), pad
  - Reductions: sum, max, min, prod, any, all, argmax/argmin tuple-reduce pattern
  - Linear algebra: dot_general (via permute+reshape+matmul)
  - Indexing: expanded gather/scatter support for batched and multi-axis patterns used by numpyro + slice update tests
  - Sorting: `stablehlo.sort` lowering for value and key/value sorts
  - Bitwise: and, or, xor, shift_left, shift_right_arithmetic, shift_right_logical
  - Other: select, atan2, complex
  - Custom calls: `mhlo.erf`, inverse trig/hyperbolic trig family, `mhlo.topk` (value + index)
  - Module: func.call (recursive)
- [x] Non-`nextafter` op test sweep is green:
  - `JAX_MLX_LIBRARY_PATH=... uv run pytest tests/test_ops.py -k 'not nextafter' --maxfail=1 -q`
  - Result: `1422 passed, 224 skipped, 8 deselected, 12 xfailed`
- [x] Benchmarks run successfully:
  - `JAX_MLX_LIBRARY_PATH=... uv run pytest -m benchmark --benchmark-only`
  - Result: `144 passed, 1666 deselected`
- [x] ResNet example runs end-to-end on MLX:
  - `JAX_PLATFORMS=mlx JAX_MLX_LIBRARY_PATH=... uv run examples/resnet/main.py --steps=5`
  - Result: completes successfully (final loss observed: `1.849`)

---

## Next Steps

### 1. Close remaining failures outside the `not nextafter` slice

The broad non-`nextafter` sweep is green. Remaining work is in:
1. `nextafter` (explicitly excluded in the current green run)
2. Any benchmark/example regressions found under broader stress or larger runs
3. Any remaining full-suite failures once `nextafter` is re-enabled for strict comparison

### 2. Control flow hardening: `stablehlo.while` and `stablehlo.if`

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

### 3. Remove old legacy MPSGraph code

Once all tests pass on the MLX backend, the following files can be deleted:

- `src/pjrt_plugin/mps_buffer.h/.mm`
- `src/pjrt_plugin/mps_client.h/.mm`
- `src/pjrt_plugin/mps_device.h/.mm`
- `src/pjrt_plugin/mps_executable.h/.mm`
- `src/pjrt_plugin/ops/` (entire directory — all `.mm` files)

### 4. Performance: `mlx::core::compile()`

Wrap `interpretFunction` with `mlx::core::compile()` to cache Metal kernel
compilation across calls. This is the primary performance optimization — identical
to how `jax.jit` caches XLA compilation, but inside the MLX runtime layer.

```cpp
// In MlxExecutable::Execute, on first call:
compiled_fn_ = mlx::core::compile([this, device](const std::vector<mx::array>& inputs) {
    return interpretFunction(entry_func_, *module_, inputs).outputs;
});
```

### 5. Accuracy and parity fixes

A small number of tests fail with numerical tolerance errors rather than missing ops.
These should be investigated after the above items are complete.

### 6. Remove avoidable `next_after` materialization

Current `next_after` fallback uses a contiguous-realization path before host-side
`std::nextafter`. This is acceptable short-term for correctness, but should be
replaced with stride-aware logical iteration that avoids forcing contiguity.
