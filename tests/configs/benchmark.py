import os
from typing import Generator

import jax
from jax import numpy as jnp
from jax import random

from .util import OperationTestConfig


def make_benchmark_op_configs() -> Generator[OperationTestConfig]:
    profile = os.environ.get("JAX_BENCH_PROFILE", "default").lower()
    if profile not in {"default", "throughput"}:
        raise ValueError(
            f"Invalid JAX_BENCH_PROFILE={profile!r}; expected 'default' or 'throughput'."
        )

    if profile == "throughput":
        element_scales = [100, 1000, 2000]
        matmul_scales = [100, 1000, 2000]
        batched_sizes = [32, 128, 256]
        conv_channels = [64, 128, 256]
        layernorm_hidden = [512, 1024, 2048]
    else:
        element_scales = [1, 10, 100, 1000]
        matmul_scales = [1, 10, 100, 1000]
        batched_sizes = [8, 32, 128]
        conv_channels = [32, 64, 128]
        layernorm_hidden = [256, 512, 1024]

    with OperationTestConfig.module_name("benchmark"):
        # Elementwise ops: use 1D arrays to avoid quadratic memory growth.
        # scale -> total elements: 1->10K, 10->100K, 100->1M, 1000->10M
        for scale in element_scales:
            n = scale * 10_000  # Total element count

            # Unary elementwise (dispatch overhead + compute).
            yield OperationTestConfig(
                jnp.exp,
                lambda key, n=n: random.normal(key, (n,)),
                name=f"exp_{scale}",
            )

            # Binary elementwise (memory bandwidth bound).
            yield OperationTestConfig(
                jnp.add,
                lambda key, n=n: random.normal(key, (n,)),
                lambda key, n=n: random.normal(key, (n,)),
                name=f"add_{scale}",
            )

            # Reduction (cross-axis operations).
            yield OperationTestConfig(
                jnp.sum,
                lambda key, n=n: random.normal(key, (n,)),
                name=f"sum_{scale}",
            )

            # Softmax (exp + reduce + div, common ML pattern).
            # Use 2D with reasonable inner dim for softmax axis.
            yield OperationTestConfig(
                lambda x: jax.nn.softmax(x, axis=-1),
                lambda key, n=n: random.normal(key, (n // 1000, 1000)),
                name=f"softmax_{scale}",
            )

        # Matmul: scale controls matrix dimensions.
        # scale -> shape: 1->(4,5)@(5,3), 10->(40,50)@(50,30), etc.
        for scale in matmul_scales:
            yield OperationTestConfig(
                jnp.matmul,
                lambda key, s=scale: random.normal(key, (s * 4, s * 5)),
                lambda key, s=scale: random.normal(key, (s * 5, s * 3)),
                name=f"matmul_{scale}",
            )

        # Batched matmul (transformer-style).
        for batch in batched_sizes:
            yield OperationTestConfig(
                jnp.matmul,
                lambda key, b=batch: random.normal(key, (b, 64, 64)),
                lambda key, b=batch: random.normal(key, (b, 64, 64)),
                name=f"matmul_batched_{batch}",
            )

        # Conv2D: vision model workloads.
        # Shape: (batch, height, width, channels) with NHWC layout.
        for channels in conv_channels:
            yield OperationTestConfig(
                lambda x, w: jax.lax.conv_general_dilated(
                    x,
                    w,
                    window_strides=(1, 1),
                    padding="SAME",
                    dimension_numbers=("NHWC", "HWIO", "NHWC"),
                ),
                lambda key, c=channels: random.normal(key, (8, 32, 32, c)),
                lambda key, c=channels: random.normal(key, (3, 3, c, c)),
                name=f"conv2d_{channels}ch",
            )

        # LayerNorm: transformer normalization.
        def layer_norm(x):
            mean = jnp.mean(x, axis=-1, keepdims=True)
            var = jnp.var(x, axis=-1, keepdims=True)
            return (x - mean) / jnp.sqrt(var + 1e-5)

        for hidden in layernorm_hidden:
            yield OperationTestConfig(
                layer_norm,
                lambda key, h=hidden: random.normal(key, (32, 128, h)),
                name=f"layernorm_{hidden}",
            )
