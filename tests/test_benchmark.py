import os
import math

import jax
import pytest
from jax import random
from pytest_benchmark.fixture import BenchmarkFixture

from .configs import OperationTestConfig, make_benchmark_op_configs

pytestmark = pytest.mark.benchmark

BENCH_AMORTIZED_ITERS = int(
    os.environ.get("JAX_BENCH_ITERS", "16")
)
if BENCH_AMORTIZED_ITERS < 1:
    raise ValueError("JAX_BENCH_ITERS must be >= 1")
BENCH_PROFILE = os.environ.get("JAX_BENCH_PROFILE", "default").lower()
if BENCH_PROFILE not in {"default", "throughput"}:
    raise ValueError(
        f"Invalid JAX_BENCH_PROFILE={BENCH_PROFILE!r}; expected 'default' or 'throughput'."
    )


def _zeros_from_shape_dtype(struct):
    return jax.tree.map(lambda x: jax.numpy.zeros(x.shape, x.dtype), struct)


def _dynamic_rounds(benchmark: BenchmarkFixture, amortized_iters: int) -> int:
    min_rounds = getattr(benchmark, "_min_rounds", 5)
    return max(math.ceil(min_rounds / amortized_iters), 1)

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
    args = jax.tree.map(
        lambda x: jax.device_put(x, device).block_until_ready(),
        op_config.get_args(args_key),
    )
    kwargs = jax.tree.map(
        lambda x: jax.device_put(x, device).block_until_ready(),
        op_config.get_kwargs(kwargs_key),
    )
    func = jax.jit(op_config.func, static_argnums=op_config.static_argnums)

    def run_once():
        return func(*args, **kwargs)

    if BENCH_AMORTIZED_ITERS > 1:
        amortized_iters = BENCH_AMORTIZED_ITERS

        @jax.jit
        def run_amortized():
            init = _zeros_from_shape_dtype(jax.eval_shape(run_once))

            def body(_i, _carry):
                return run_once()

            return jax.lax.fori_loop(0, amortized_iters, body, init)

        def run():
            return run_amortized().block_until_ready()
    else:
        amortized_iters = 1

        def run():
            return run_once().block_until_ready()

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
    args = jax.tree.map(
        lambda x: jax.device_put(x, device).block_until_ready(),
        op_config.get_args(args_key),
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

    if BENCH_AMORTIZED_ITERS > 1:
        amortized_iters = BENCH_AMORTIZED_ITERS

        @jax.jit
        def run_amortized():
            init = _zeros_from_shape_dtype(jax.eval_shape(run_once))

            def body(_i, _carry):
                return run_once()

            return jax.lax.fori_loop(0, amortized_iters, body, init)

        def run():
            result = run_amortized()
            jax.tree.map(lambda x: x.block_until_ready(), result)
            return result
    else:
        amortized_iters = 1

        def run():
            result = run_once()
            jax.tree.map(lambda x: x.block_until_ready(), result)
            return result

    benchmark.extra_info["amortized_iterations"] = amortized_iters
    benchmark.extra_info["profile"] = BENCH_PROFILE

    # One warmup pass to pay JIT compile cost before timing.
    run()

    rounds = _dynamic_rounds(benchmark, amortized_iters)
    benchmark.pedantic(run, rounds=rounds, iterations=1)
