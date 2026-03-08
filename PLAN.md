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
  - Convolution: `stablehlo.convolution` with full dim_number remapping, feature groups, **batch groups** (used for grouped/depthwise weight gradients)
  - Indexing: expanded gather/scatter support for batched and multi-axis patterns used by numpyro + slice update tests
  - Sorting: `stablehlo.sort` lowering for value and key/value sorts
  - Bitwise: and, or, xor, shift_left, shift_right_arithmetic, shift_right_logical
  - Control flow: `stablehlo.while`, `stablehlo.if`, `stablehlo.case`, `func.call` (recursive)
  - Other: select, atan2, complex, next_after (stride-aware)
  - Custom calls: `mhlo.erf`, inverse trig/hyperbolic trig family, `mhlo.topk` (value + index)
  - Linalg: Cholesky, triangular solve, QR, eigh, eig, svd, det, slogdet, trace, lgamma
  - FFT: fft, ifft, rfft, irfft (1D/2D/3D via custom_call)
- [x] Full test suite is green:
  - `uv run pytest` → `1430 passed, 224 skipped, 144 deselected, 12 xfailed`
  - The 12 xfails are zero-sized tensor tests (MLX/Metal platform limitation)
  - The 224 skips are gradient tests for non-differentiable ops (argmax, bitwise, etc.) — expected
- [x] Benchmarks run successfully (144 passed); MLX 2–4× faster than CPU on conv2d ≥64ch and matmul ≥1000
- [x] ResNet example runs end-to-end on MLX

---

## Next Steps

### 1. Performance: `mlx::core::compile()`

Wrap the interpreter execution with `mlx::core::compile()` to cache Metal kernel
compilation across repeated calls with the same shapes. This is the primary performance
optimization — analogous to `jax.jit` caching XLA compilation.

Without it, every call to `Execute` re-traces the MLX graph and recompiles Metal kernels,
which is why layernorm (5–10×) and softmax (2–4×) are still slower than CPU at medium
sizes. With `mlx::core::compile()` the compilation cost is amortized.

```cpp
// In MlxExecutable, cache a compiled function keyed on input shapes:
compiled_fn_ = mlx::core::compile([this](const std::vector<mx::array>& inputs) {
    return interpretFunction(entry_func_, *module_, inputs).outputs;
});
// Then in Execute: outputs = compiled_fn_(input_arrays);
```

Key constraints:
- The compiled function must be pure: same inputs → same outputs, no side effects.
- Shape/dtype changes require a new compiled function (or the cache key must include shapes).
- `mx::eval(outputs)` must still be called after `compiled_fn_`.

### 2. Remove old legacy MPSGraph code

Once confidence is high that the MLX backend is stable, delete:

- `src/pjrt_plugin/mps_buffer.h/.mm`
- `src/pjrt_plugin/mps_client.h/.mm`
- `src/pjrt_plugin/mps_device.h/.mm`
- `src/pjrt_plugin/mps_executable.h/.mm`
- `src/pjrt_plugin/ops/` (entire directory — all `.mm` files)

These are compiled but never linked in the MLX build, so this is pure cleanup.

### 3. Remaining test gaps

- **Grouped/depthwise conv weight grads** — now fixed via `batch_group_count` loop implementation
- **`jnp.pad` gradient** — now fixed (was a stale MPS-era FIXME)
- **`nnx.Embed(2d)` gradient** — now fixed (was a stale MPS-era FIXME)
- **Zero-sized tensor xfails** (3 tests: cholesky, triangular_solve, batched matmul) — MLX/Metal
  platform limitation, not actionable without upstream MLX changes

### 4. Accuracy and parity investigation

The 12 xfailed tests are all zero-sized tensor cases. No numerical tolerance failures
remain in the current sweep. If regressions appear after `mlx::core::compile()` is added,
investigate then.
