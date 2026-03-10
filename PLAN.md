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
**Last updated**: March 10, 2026 (upstream JAX control-flow triage ongoing)

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
  - Shape: reshape, broadcast_in_dim, convert, transpose, iota, copy, concatenate, slice, dynamic_slice (compile-safe via `arange+take`), dynamic_update_slice (compile-safe via broadcast linear index), pad
  - Reductions: sum, max, min, prod, any, all, argmax/argmin tuple-reduce pattern
  - Linear algebra: dot_general (via permute+reshape+matmul)
  - Convolution: `stablehlo.convolution` with full dim_number remapping, feature groups, **batch groups** (used for grouped/depthwise weight gradients)
  - Indexing: expanded gather/scatter support for batched and multi-axis patterns used by numpyro + slice update tests
  - Sorting: `stablehlo.sort` lowering for value and key/value sorts
  - Bitwise: and, or, xor, shift_left, shift_right_arithmetic, shift_right_logical
  - Control flow: `stablehlo.while`, `stablehlo.if`, `stablehlo.case`, `func.call` (recursive)
  - Other: select, atan2, complex, next_after (compile-safe via MLX bit manipulation), bitcast_convert (compile-safe via `mlxc::view`)
  - Custom calls: `mhlo.erf`, inverse trig/hyperbolic trig family, `mhlo.topk` (value + index)
  - Linalg: Cholesky, triangular solve, QR, eigh, eig, svd, det, slogdet, trace, lgamma
  - FFT: fft, ifft, rfft, irfft (1D/2D/3D via custom_call)
- [x] Full test suite is green:
  - `uv run pytest` → `1430 passed, 224 skipped, 144 deselected, 12 xfailed`
  - The 12 xfails are zero-sized tensor tests (MLX/Metal platform limitation)
  - The 224 skips are gradient tests for non-differentiable ops (argmax, bitwise, etc.) — expected
- [x] Benchmarks run successfully (144 passed); MLX 2–4× faster than CPU on conv2d ≥64ch and matmul ≥1000
- [x] ResNet example runs end-to-end on MLX
- [x] Legacy MPS-era files removed; `MpsDevice` → `MlxDevice` rename complete
- [x] `mlx::core::compile()` Phases A–C: all eval barriers eliminated for common ML ops;
  `Execute()` lazily wraps compilable functions; graceful fallback on Metal kernel failures
- [x] NaN/inf constant fix (`makeConstant`): float constants containing NaN or ±inf are now
  created via integer bit-pattern + `mlxc::view` so `mlxc::compile()` never embeds `nan`/`inf`
  as Metal float literals (undefined in Metal shading language). This unblocked `mlxc::compile()`
  for **layernorm** (which contains a `select+NaN` pattern) giving 2–3× speedup. Softmax
  regresses slightly at toy sizes (< 100 elements) due to compile overhead; not impactful
  for realistic transformer workloads.
- [x] Upstream JAX test runs now always save structured artifacts under `.benchmarks/jax_tests_*`
  and produce summarized failed-test selectors via `scripts/summarize_jax_tests.py`.
- [x] Fixed PJRT output metadata for zero-result executables (removed forced `num_outputs_=1`);
  this unblocked `lax_control_flow` grad/checkpoint tests that return no values.
- [x] Implemented `stablehlo.all_reduce` for single-device MLX semantics (identity over one
  replica), clearing prior `Unsupported operation(s): stablehlo.all_reduce` failures.
- [x] Updated upstream harness (`scripts/jax_tests.sh`) to use `JAX_PLATFORM_NAME=mlx`
  instead of `JAX_PLATFORMS=mlx`, so MLX remains the default backend while explicit
  `backend='cpu'` requests inside upstream tests continue to work.

---

## Next Steps

### Current Upstream Failure Profile (`tests/lax_control_flow_test.py`)

Latest run (`.benchmarks/jax_tests_2026-03-10T09-25-40_2de5b8b07`):
- `487 passed / 119 failed / 483 skipped`
- Previous runs:
  - `.benchmarks/jax_tests_2026-03-10T07-27-04_2de5b8b07`: `483 / 123 / 483`
  - Baseline before these fixes: `481 / 125 / 483`
- Net progress: 6 failures removed (all_reduce category eliminated; cpu-backend harness issue fixed; associative-scan solving regressions fixed)

Remaining high-impact categories:
1. `stablehlo.gather` / `stablehlo.scatter` generalization (remaining forms)
   - `testAssociativeScanSolvingRegressionTest_{2,43,100}` is now fixed.
   - Remaining gather/scatter work is broader indexing/scan combinations, not this specific regression.
2. Large scan numerics/assertion cluster (`impl=unroll0` dominant)
   - Root cause still unresolved; likely semantic mismatch in control-flow/indexing path used by scan lowering.
3. Broad scan/associative-scan assertion mismatches (`impl=unroll0`, many vmap combinations)
   - Requires deeper semantic debugging in control-flow/indexing lowering paths beyond backend/harness fixes.

### Notes From Current Iteration (March 10, 2026)

Tried and observed:
1. Added scalar index-vector gather handling for:
   - `operand_rank=3`
   - `start_indices_rank=1`
   - `index_vector_dim=0`
   - `start_index_map=[1,2]`
   - `collapsed_slice_dims=[]`
   - `slice_sizes=[1,2,1]`
   - Result: gather no longer fails in associative-scan solve path.
2. Added batched gather rank normalization for one-axis gathers:
   - allow `indices.ndim == operand.ndim() - 1` by appending a trailing singleton dim
   - validate/broadcast non-axis dims before `take_along_axis`
3. Added rank-3 axis-scatter fallback:
   - normalize axis to a common layout
   - apply per-slice 2D `scatter_add_axis` / `put_along_axis`
   - transpose back
   - Result: `testAssociativeScanSolvingRegressionTest_{2,43,100}` now passes.

Conclusion from iteration:
- This path confirms the correctness-first strategy works: add narrow fast paths where possible, then normalize shape/layout before MLX primitive calls.
- Next attempt should continue generalizing gather/scatter through explicit normalization/fallbacks for the remaining associative-scan unstructured and scan-vmap assertion clusters.

### 1. Performance: `mlx::core::compile()` — Incremental Refactor Plan

#### Why naive wrapping fails

`mlxc::compile()` traces a function with placeholder arrays and records MLX operations
without executing them. Any `eval()` / `item<T>()` / `data<T>()` call on an intermediate
array during tracing throws:

```
[eval] Attempting to eval an array during function transformations like compile or vmap
```

The interpreter currently has **seven eval sites** that prevent compilation.  They fall
into three categories:

| Category | Ops | Eval site | Replaceable? |
|----------|-----|-----------|-------------|
| **Index extraction** | `dynamic_slice`, `dynamic_update_slice` | `item<>()` to get integer start indices for `mlxc::slice` | **Yes — Phase A** |
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
`arange + add + take` per dimension. Verified to work inside `mlxc::compile`:

```cpp
// Before (breaks compile):
int start = ScalarToInt64(startArr);
mlxc::slice(operand, {start, ...}, {start + size, ...});

// After (compile-safe):
// For each spatial dimension d with static size S[d] and lazy start S_d:
auto idx_d = mlxc::add(mlxc::arange(S[d], mlxc::int32),
                     mlxc::astype(start_d, mlxc::int32));
// Apply all dimensions via sequential mlxc::take calls:
auto result = operand;
for each dim d: result = mlxc::take(result, idx_d, d);
```

**A2. `stablehlo.dynamic_update_slice`**: Same approach — construct index arrays per
dimension and use `mlxc::scatter` / `mlxc::put_along_axis` rather than
`mlxc::slice_update(operand, update, {materialized_starts})`.

**A3. `stablehlo.bitcast_convert`**: Replace the `eval + copyStridedToLinearBytes +
malloc/memcpy` path with `mlxc::view(operand, target_dtype)`.  `mlxc::view` is already
verified to work inside `mlxc::compile` and handles the reinterpret-cast semantics
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
auto x_bits   = mlxc::view(x, mlxc::int32);
auto y_bits   = mlxc::view(y, mlxc::int32);
auto zero_i   = mlxc::zeros_like(x_bits);
auto one_i    = mlxc::ones_like(x_bits);
auto x_eq_y   = mlxc::equal(x, y);
auto x_is_0   = mlxc::equal(x, mlxc::zeros_like(x));
auto x_pos    = mlxc::greater_equal(x_bits, zero_i);
auto towards_y = mlxc::greater(y, x);
auto dir      = mlxc::where(towards_y, one_i, mlxc::full(x_bits.shape(), -1, mlxc::int32));
// same-sign branch: add direction; different-sign branch: subtract direction
auto y_pos    = mlxc::greater_equal(y_bits, zero_i);
auto same_sign = mlxc::equal(x_pos, y_pos);
auto nz_bits  = mlxc::where(same_sign, mlxc::add(x_bits, dir), mlxc::subtract(x_bits, dir));
// zero input: return smallest subnormal with sign of y
auto min_sub  = mlxc::where(y_pos,
                    mlxc::full(x_bits.shape(), 1,           mlxc::int32),
                    mlxc::full(x_bits.shape(), 0x80000001u, mlxc::int32));
auto res_bits = mlxc::where(x_is_0, min_sub, nz_bits);
res_bits      = mlxc::where(x_eq_y, y_bits, res_bits);
return mlxc::view(res_bits, mlxc::float32);
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
        compiled_fn_ = mlxc::compile(CompiledFn{
            [this](const std::vector<mlxc::array>& in) {
                auto r = interpretFunction(entry_func_, *module_, in);
                if (!r.ok()) throw std::runtime_error(r.error);
                return r.outputs;
            }});
    }
}
```

Note: `stablehlo.while`, `stablehlo.if`, and `stablehlo.case` do NOT need to be in
`kBarriers` — those ops call `mlxc::eval()` on the condition/predicate scalars to decide
which branch to take, but they do so as part of interpreter dispatch logic, not as
intermediate-array materialization.  Wrapping a function that contains `while` with
`mlxc::compile()` would unroll the loop for the first execution's iteration count, which
is wrong for variable-trip-count loops.  Therefore, these must remain in the barrier set
even though they don't call `eval()` on data arrays.  Update `kBarriers` accordingly.

**Verification gate**: Full test suite (`uv run pytest`) still 1430 passed.
Run `bash scripts/benchmark.sh` before and after Phase C; run
`uv run scripts/benchmark_compare.py` to confirm the ops targeted by compilation
appear in the FASTER column with no regressions in the SLOWER column.

---

### 2. Targeted performance improvements

The goal is to find ops that are slower on MLX than CPU for typical ML workloads,
not just raw microbenchmarks, and prioritize investigation accordingly.

#### Approach

1. **Count ops in ResNet's StableHLO**: JAX code cannot be profiled with
   `cProfile` or `py-spy` at the JAX op level. Instead, use `jax.jit` +
   `lower()` to get the StableHLO bytecode for the main `train_step` in
   `examples/resnet/main.py`, then write a small script that parses the MLIR
   text and counts occurrences of each op name (e.g. `stablehlo.convolution`,
   `stablehlo.dot_general`, `stablehlo.broadcast_in_dim`). This identifies
   the top-10 most-called operations by frequency, which are the best
   candidates for per-call optimization. Compare their per-call time on MLX
   vs CPU using the existing microbenchmarks or new ones.

2. **Cross-reference benchmarks**: Run `bash scripts/benchmark.sh` and look at the
   SLOWER column relative to CPU — any op that is slower on MLX than CPU for typical
   sizes warrants investigation.

3. **Known candidates from current benchmarks**:
   - `layernorm` — **FIXED** via NaN/inf constant fix. `mlxc::compile()` now
     successfully fuses the full normalization pass into a Metal kernel, giving 2–3×
     speedup. Remaining gap vs CPU (~2.5× for hidden=1024) is fundamental: reductions
     cannot be fused with elementwise ops in Metal.
   - `softmax` at small sizes (< 100 elements) — regressed slightly due to NaN fix
     enabling compile for what was previously interpreted. Not impactful for realistic
     workloads where softmax processes attention matrices with many more elements.
   - Small matmuls / element-wise ops below crossover — MLX has higher kernel launch
     latency than CPU for tiny inputs. This is a platform characteristic. `mlxc::compile()`
     helps by reducing dispatch count, but the crossover for benefit depends on input size
     (> ~1000 total elements is a rough guideline).

4. **For each flagged op**: write a targeted microbenchmark, profile, and either
   implement a better lowering or document why MLX is fundamentally slower for that
   op/size regime.

### 3. JAX upstream test suite

**Goal**: Keep upstream JAX failures measurable even when one file segfaults, then
drive pass rate upward by fixing the highest-leverage failure buckets first.

**Runner/output flow**:
- `scripts/jax_tests.sh` now runs each upstream file independently and writes structured output to
  `.benchmarks/jax_tests_<timestamp>_<jax_head>/`.
- Each file writes `*.junit.xml` + `*.log`; return codes are written to `exit_codes.tsv`.
- `scripts/summarize_jax_tests.py` parses those artifacts, writes `summary.json`, and prints failed selectors.
- If a file crashes before XML is emitted, the summarizer now records a crash entry with the inferred selector from the log.

**Most recent upstream sweep** (2026-03-09, JAX tag `jax-v0.9.0`, run dir `.benchmarks/jax_tests_2026-03-09T23-21-11_2de5b8b07`):
- Parsed totals (files with JUnit XML): `3549 passed, 583 failed, 517 skipped, 4649 total`
- Per-file failures: `lax_numpy_test.py` 257, `lax_control_flow_test.py` 161, `lax_numpy_indexing_test.py` 165
- Crash-only file: `tests/lax_autodiff_test.py::testScatterGradSymbolicZeroUpdate` (exit code 139 / segfault)

**Current top failure buckets from `summary.json` messages**:

| Count | Bucket | Likely implementation area |
|------:|--------|----------------------------|
| 119 | `stablehlo.scatter` lowering failures | Extend scatter beyond single-axis/rank-2 point assumptions; support complex64 where valid |
| 103 | `stablehlo.gather` lowering failures | Add multi-batch / two-axis / mixed advanced indexing gather patterns |
| 30 | `Memory kinds and dtypes have different sizes` | Investigate `batched_device_put`/buffer metadata path and zero-size handling |
| 26 | missing Metal `*complex64*` kernels | Add complex fallback policy (cast/real-imag decomposition where semantically correct) |
| 19 | `stablehlo.pad with interior padding` unsupported | Implement interior padding path |
| 15 | `stablehlo.optimization_barrier` unsupported | Add no-op passthrough lowering |
| 10 | primitive `eigh` translation missing for mlx | Add MLIR lowering/registration for eigendecomposition path used by polyfit |
| 6 | missing Metal sort kernels | Add sort dtype fallback (cast to supported dtype, sort, cast back when safe) |

Plus many assertion-only mismatches (`~229`) that should be revisited after gather/scatter/pad fixes reduce cascading errors.

**Next implementation plan (ordered)**:
1. Stabilize crash path first: isolate and fix `testScatterGradSymbolicZeroUpdate` segfault in `batched_device_put`/sharding path.
2. Land `stablehlo.optimization_barrier` passthrough (small unblocker that removes many control-flow gradient failures).
3. Expand `stablehlo.gather` for multi-axis/multi-batch indexing patterns used by `lax_numpy_indexing_test.py`.
4. Expand `stablehlo.scatter` for multi-axis/segment-reduce/boolean update patterns.
5. Implement interior-padding support in `stablehlo.pad`.
6. Add robust dtype fallbacks for complex/sort kernels and close out remaining kernel-load failures.
7. Re-run `bash scripts/jax_tests.sh -q --tb=no` and refresh `summary.json`; then triage remaining assertion mismatches.

**Execution workflow per bucket**:
1. Use failed selectors directly from `summary.json` for focused reruns (`bash scripts/jax_tests.sh -q --tb=short -k <selector-fragment>`).
2. Dump StableHLO for representative failing cases from `.agent-context/jax`.
3. Implement lowering changes in `src/pjrt_plugin/mlx_executable.cc`.
4. Rebuild with `uv pip install --python .venv -e .` and re-run the focused selectors, then a full upstream sweep.

### 4. Remaining test gaps

- **Grouped/depthwise conv weight grads** — now fixed via `batch_group_count` loop implementation
- **`jnp.pad` gradient** — now fixed (was a stale MPS-era FIXME)
- **`nnx.Embed(2d)` gradient** — now fixed (was a stale MPS-era FIXME)
- **Zero-sized tensor xfails** (3 tests: cholesky, triangular_solve, batched matmul) — MLX/Metal
  platform limitation, not actionable without upstream MLX changes

### 5. Accuracy and parity investigation

The 12 xfailed tests are all zero-sized tensor cases. No numerical tolerance failures
remain in the current sweep. If regressions appear after `mlx::core::compile()` is added,
investigate then.

### 6. Segment compilation for linalg models (future)

For functions that mix compilable regions with hard barriers (`cholesky`,
`triangular_solve`, `lgamma`), a segment-compilation pass could split the function at
each barrier, compile each inter-barrier segment independently, and execute them as:

```
compiled_prefix → host_barrier_1 → compiled_middle → host_barrier_2 → compiled_suffix
```

This requires a pre-pass that identifies segments, extracts them as sub-functions, and
compiles each independently.  Implementation is more complex and the benefit is only felt
on linalg-heavy models — defer until Phase C is validated.
