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

### 1. Performance: `mlx::core::compile()` — Incremental Refactor Plan

#### Why naive wrapping fails

`mx::compile()` traces a function with placeholder arrays and records MLX operations
without executing them. Any `eval()` / `item<T>()` / `data<T>()` call on an intermediate
array during tracing throws:

```
[eval] Attempting to eval an array during function transformations like compile or vmap
```

The interpreter currently has **seven eval sites** that prevent compilation.  They fall
into three categories:

| Category | Ops | Eval site | Replaceable? |
|----------|-----|-----------|-------------|
| **Index extraction** | `dynamic_slice`, `dynamic_update_slice` | `item<>()` to get integer start indices for `mx::slice` | **Yes — Phase A** |
| **Bit reinterpretation** | `bitcast_convert` | `data<uint8_t>()` to copy bytes | **Yes — Phase A** |
| **Host-side math** | `next_after` | `data<T>()` for `std::nextafter` loop | **Yes — Phase B** |
| **Host-side math** | `lgamma` | `data<T>()` for `std::lgamma` loop | No — MLX has no native lgamma |
| **CPU linear algebra** | `cholesky`, `triangular_solve` | `data<T>()` for manual decomposition | No — MLX has no native Cholesky/trsm |

After Phases A and B, the only remaining eval barriers are `lgamma`, `cholesky`, and
`triangular_solve` — none of which appear in typical ML models (ResNet, transformers,
diffusion, etc.). Those models become fully compilable.

---

#### Phase A — Eliminate index and bitcast barriers

**A1. `stablehlo.dynamic_slice`**: Replace `item<>()` index extraction with
`arange + add + take` per dimension. Verified to work inside `mx::compile`:

```cpp
// Before (breaks compile):
int start = ScalarToInt64(startArr);
mx::slice(operand, {start, ...}, {start + size, ...});

// After (compile-safe):
// For each spatial dimension d with static size S[d] and lazy start S_d:
auto idx_d = mx::add(mx::arange(S[d], mx::int32),
                     mx::astype(start_d, mx::int32));
// Apply all dimensions via sequential mx::take calls:
auto result = operand;
for each dim d: result = mx::take(result, idx_d, d);
```

**A2. `stablehlo.dynamic_update_slice`**: Same approach — construct index arrays per
dimension and use `mx::scatter` / `mx::put_along_axis` rather than
`mx::slice_update(operand, update, {materialized_starts})`.

**A3. `stablehlo.bitcast_convert`**: Replace the `eval + copyStridedToLinearBytes +
malloc/memcpy` path with `mx::view(operand, target_dtype)`.  `mx::view` is already
verified to work inside `mx::compile` and handles the reinterpret-cast semantics
correctly (changes shape along last axis when element sizes differ).

**Verification gate**: After A1–A3, all slice/scatter/bitcast tests must still pass.
Run `uv run pytest -k "slice or scatter or bitcast"`.

---

#### Phase B — Port `next_after` to pure MLX bit manipulation

`std::nextafter` can be expressed entirely in MLX as integer bit manipulation (the same
logic already implemented in `binary_ops.mm` for the MPSGraph backend).  Verified
compile-safe:

```cpp
// next_after(x, y) for float32 — no eval needed:
auto x_bits   = mx::view(x, mx::int32);
auto y_bits   = mx::view(y, mx::int32);
auto zero_i   = mx::zeros_like(x_bits);
auto one_i    = mx::ones_like(x_bits);
auto x_eq_y   = mx::equal(x, y);
auto x_is_0   = mx::equal(x, mx::zeros_like(x));
auto x_pos    = mx::greater_equal(x_bits, zero_i);
auto towards_y = mx::greater(y, x);
auto dir      = mx::where(towards_y, one_i, mx::full(x_bits.shape(), -1, mx::int32));
// same-sign branch: add direction; different-sign branch: subtract direction
auto y_pos    = mx::greater_equal(y_bits, zero_i);
auto same_sign = mx::equal(x_pos, y_pos);
auto nz_bits  = mx::where(same_sign, mx::add(x_bits, dir), mx::subtract(x_bits, dir));
// zero input: return smallest subnormal with sign of y
auto min_sub  = mx::where(y_pos,
                    mx::full(x_bits.shape(), 1,           mx::int32),
                    mx::full(x_bits.shape(), 0x80000001u, mx::int32));
auto res_bits = mx::where(x_is_0, min_sub, nz_bits);
res_bits      = mx::where(x_eq_y, y_bits, res_bits);
return mx::view(res_bits, mx::float32);
// For float64: same structure with int64 / uint64 bit types.
```

The current host-side `HandleNextAfter` in `mlx_executable.cc` already implements this
logic with `std::nextafter`; Phase B rewrites it using the MLX ops above instead.

**Verification gate**: `uv run pytest -k "nextafter"` still passes.

---

#### Phase C — Compile gate in `MlxExecutable::Execute`

After Phases A and B, add a pre-pass that walks the MLIR function once to detect
remaining hard barriers:

```cpp
static bool HasHardEvalBarrier(mlir::func::FuncOp func) {
    // Any op whose handler still calls eval() internally
    static const std::unordered_set<std::string_view> kBarriers = {
        "stablehlo.cholesky",
        "stablehlo.triangular_solve",
        "chlo.lgamma",
    };
    bool found = false;
    func.walk([&](mlir::Operation* op) {
        if (kBarriers.count(op->getName().getStringRef())) {
            found = true;
            return mlir::WalkResult::interrupt();
        }
        return mlir::WalkResult::advance();
    });
    return found;
}
```

In `MlxExecutable::Execute`, lazily compile on first call when no barrier is detected:

```cpp
if (!compiled_fn_.has_value()) {
    if (HasHardEvalBarrier(entry_func_)) {
        compiled_fn_ = CompiledFn{};  // sentinel: fall back to interpreted
    } else {
        compiled_fn_ = mx::compile(CompiledFn{
            [this](const std::vector<mx::array>& in) {
                auto r = interpretFunction(entry_func_, *module_, in);
                if (!r.ok()) throw std::runtime_error(r.error);
                return r.outputs;
            }});
    }
}
```

Note: `stablehlo.while`, `stablehlo.if`, and `stablehlo.case` do NOT need to be in
`kBarriers` — those ops call `mx::eval()` on the condition/predicate scalars to decide
which branch to take, but they do so as part of interpreter dispatch logic, not as
intermediate-array materialization.  Wrapping a function that contains `while` with
`mx::compile()` would unroll the loop for the first execution's iteration count, which
is wrong for variable-trip-count loops.  Therefore, these must remain in the barrier set
even though they don't call `eval()` on data arrays.  Update `kBarriers` accordingly.

**Verification gate**: Full test suite (`uv run pytest`) still 1430 passed.

---

#### Phase D (future) — Segment compilation for linalg models

For functions that mix compilable regions with hard barriers (`cholesky`,
`triangular_solve`, `lgamma`), a segment-compilation pass could split the function at
each barrier, compile each inter-barrier segment independently, and execute them as:

```
compiled_prefix → host_barrier_1 → compiled_middle → host_barrier_2 → compiled_suffix
```

This requires a pre-pass that identifies segments, extracts them as sub-functions, and
compiles each independently.  Implementation is more complex and the benefit is only felt
on linalg-heavy models — defer until Phase C is validated.

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
