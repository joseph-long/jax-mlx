import math
import os

import jax
import pytest
from jax import random
from pytest_benchmark.fixture import BenchmarkFixture

from .configs import OperationTestConfig, make_benchmark_op_configs

pytestmark = pytest.mark.benchmark

BENCH_AMORTIZED_ITERS = int(os.environ.get("JAX_BENCH_ITERS", "16"))
if BENCH_AMORTIZED_ITERS < 1:
    raise ValueError("JAX_BENCH_ITERS must be >= 1")
BENCH_PROFILE = os.environ.get("JAX_BENCH_PROFILE", "default").lower()
if BENCH_PROFILE not in {"default", "throughput"}:
    raise ValueError(
        f"Invalid JAX_BENCH_PROFILE={BENCH_PROFILE!r}; expected 'default' or 'throughput'."
    )


def _dynamic_rounds(benchmark: BenchmarkFixture, amortized_iters: int) -> int:
    min_rounds = getattr(benchmark, "_min_rounds", 5)
    return max(math.ceil(min_rounds / amortized_iters), 1)


def _find_first_inexact_arg_index(args):
    for i, arg in enumerate(args):
        if hasattr(arg, "dtype") and jax.numpy.issubdtype(arg.dtype, jax.numpy.inexact):
            return i
    return None


def _find_inexact_arg_index_preferring_not(args, excluded_index):
    for i, arg in enumerate(args):
        if i == excluded_index:
            continue
        if hasattr(arg, "dtype") and jax.numpy.issubdtype(arg.dtype, jax.numpy.inexact):
            return i
    return _find_first_inexact_arg_index(args)


def _batched_variants(arg, iters):
    offsets = jax.numpy.arange(iters, dtype=arg.dtype).reshape(
        (iters,) + (1,) * arg.ndim
    ) * jax.numpy.array(1e-6, dtype=arg.dtype)
    return jax.numpy.expand_dims(arg, 0) + offsets


def _replace_arg(args, index, value):
    return args[:index] + (value,) + args[index + 1 :]


def _result_fingerprint(result):
    leaves = jax.tree.leaves(result)
    if not leaves:
        return jax.numpy.array(0.0, dtype=jax.numpy.float32)
    leaf = leaves[0]
    if leaf.size == 0:
        return jax.numpy.array(0.0, dtype=jax.numpy.float32)
    value = jax.numpy.reshape(leaf, (-1,))[0]
    if jax.numpy.issubdtype(value.dtype, jax.numpy.complexfloating):
        value = jax.numpy.real(value)
    return jax.lax.convert_element_type(value, jax.numpy.float32)


OPERATION_TEST_CONFIGS = list(make_benchmark_op_configs())
GRAD_TEST_CONFIGS = []
for op_config in OPERATION_TEST_CONFIGS:
    with jax.default_device("cpu"):
        differentiable_argnums = op_config.get_differentiable_argnums()
        GRAD_TEST_CONFIGS.extend(
            (op_config, argnum) for argnum in differentiable_argnums
        )

DEVICES = []
for platform in ["cpu", "mlx"]:
    try:
        DEVICES.append(jax.devices(platform)[0])
    except RuntimeError as ex:
        if "Unknown backend" not in str(ex):
            raise


@pytest.fixture(params=DEVICES, ids=lambda x: x.platform)
def device(request: pytest.FixtureRequest):
    return request.param


@pytest.mark.parametrize(
    "op_config", OPERATION_TEST_CONFIGS, ids=lambda op_config: op_config.name
)
def test_benchmark_value(
    op_config: OperationTestConfig, device, benchmark: BenchmarkFixture
) -> None:
    # Get the args and move them to the right device.
    key = random.key(op_config.seed)
    args_key, kwargs_key = random.split(key)
    args = tuple(
        jax.tree.map(
            lambda x: jax.device_put(x, device).block_until_ready(),
            op_config.get_args(args_key),
        )
    )
    kwargs = jax.tree.map(
        lambda x: jax.device_put(x, device).block_until_ready(),
        op_config.get_kwargs(kwargs_key),
    )
    func = jax.jit(op_config.func, static_argnums=op_config.static_argnums)

    def run_once():
        return func(*args, **kwargs)

    amortized_iters = max(BENCH_AMORTIZED_ITERS, 1)
    varied_arg_index = _find_first_inexact_arg_index(args)
    varied_arg_batch = (
        _batched_variants(args[varied_arg_index], amortized_iters)
        if amortized_iters > 1 and varied_arg_index is not None
        else None
    )

    @jax.jit
    def run_amortized(call_args, call_kwargs, batched_arg):
        def body(_carry, per_iter_arg):
            iter_args = _replace_arg(call_args, varied_arg_index, per_iter_arg)
            out = func(*iter_args, **call_kwargs)
            return None, _result_fingerprint(out)

        _, outputs = jax.lax.scan(body, None, batched_arg)
        return outputs

    def run():
        if amortized_iters == 1 or varied_arg_index is None:
            return run_once().block_until_ready()
        result = run_amortized(args, kwargs, varied_arg_batch)
        jax.tree.map(lambda x: x.block_until_ready(), result)
        return result

    benchmark.extra_info["amortized_iterations"] = amortized_iters
    benchmark.extra_info["profile"] = BENCH_PROFILE

    # One warmup pass to pay JIT compile cost before timing.
    run()

    rounds = _dynamic_rounds(benchmark, amortized_iters)
    benchmark.pedantic(run, rounds=rounds, iterations=1)


@pytest.mark.parametrize(
    "op_config,argnum",
    GRAD_TEST_CONFIGS,
    ids=[f"{cfg.name}_grad{argnum}" for cfg, argnum in GRAD_TEST_CONFIGS],
)
def test_benchmark_grad(
    op_config: OperationTestConfig, argnum: int, device, benchmark: BenchmarkFixture
) -> None:
    # Get the args and move them to the right device.
    key = random.key(op_config.seed)
    args_key, kwargs_key = random.split(key)
    args = tuple(
        jax.tree.map(
            lambda x: jax.device_put(x, device).block_until_ready(),
            op_config.get_args(args_key),
        )
    )
    kwargs = jax.tree.map(
        lambda x: jax.device_put(x, device).block_until_ready(),
        op_config.get_kwargs(kwargs_key),
    )

    # Build scalar loss function (grad requires scalar output).
    func = op_config.func

    def scalar_output(*args, **kwargs):
        result = func(*args, **kwargs)
        if isinstance(result, (tuple, list)):
            result = result[0]
        # Handle complex outputs (same as OperationTestConfig.evaluate_grad).
        if jax.numpy.issubdtype(result.dtype, jax.numpy.complexfloating):
            result = jax.numpy.abs(result)
        if result.shape != ():
            result = result.mean()
        return result

    grad_func = jax.jit(
        op_config.grad_transform(scalar_output, argnums=argnum),
        static_argnums=op_config.static_argnums,
    )

    def run_once():
        return grad_func(*args, **kwargs)

    amortized_iters = max(BENCH_AMORTIZED_ITERS, 1)
    varied_arg_index = _find_inexact_arg_index_preferring_not(args, argnum)
    varied_arg_batch = (
        _batched_variants(args[varied_arg_index], amortized_iters)
        if amortized_iters > 1 and varied_arg_index is not None
        else None
    )

    def run():
        if amortized_iters == 1 or varied_arg_index is None:
            result = run_once()
            jax.tree.map(lambda x: x.block_until_ready(), result)
            return result
        result = None
        for i in range(amortized_iters):
            iter_args = _replace_arg(args, varied_arg_index, varied_arg_batch[i])
            result = grad_func(*iter_args, **kwargs)
        jax.tree.map(lambda x: x.block_until_ready(), result)
        return result

    benchmark.extra_info["amortized_iterations"] = amortized_iters
    benchmark.extra_info["profile"] = BENCH_PROFILE

    # One warmup pass to pay JIT compile cost before timing.
    run()

    rounds = _dynamic_rounds(benchmark, amortized_iters)
    benchmark.pedantic(run, rounds=rounds, iterations=1)
