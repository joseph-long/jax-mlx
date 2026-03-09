#!/usr/bin/env python3
"""Count StableHLO operation frequencies in a JAX computation.

Usage:
    uv run scripts/count_ops.py [--top N] [--filter PATTERN]

Lowers the ResNet18 train_step to StableHLO and reports the most-frequent
operations — useful for identifying which ops to prioritize for optimization.

Compare per-op MLX vs CPU performance using the existing microbenchmarks or
by adding targeted benchmarks for under-performing ops.
"""

import argparse
import os
import re
import sys
from collections import Counter

# Run on CPU for lowering (shape inference only, no actual execution).
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
from jax import random


def get_resnet_stablehlo() -> str:
    """Lower the ResNet18 train_step to StableHLO text."""
    # Add examples/resnet to sys.path so we can import the model.
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(script_dir)
    resnet_dir = os.path.join(repo_root, "examples", "resnet")
    if resnet_dir not in sys.path:
        sys.path.insert(0, resnet_dir)

    import optax
    from flax import nnx
    from model import ResNet18  # type: ignore[import]

    # Build model and optimizer.
    model = ResNet18(num_classes=10, rngs=nnx.Rngs(0))
    optimizer = nnx.Optimizer(model, optax.adam(1e-3), wrt=nnx.Param)

    # Representative inputs: batch=2 (small enough to lower quickly).
    key = random.key(0)
    inputs = random.normal(key, (2, 32, 32, 3))
    labels_onehot = jnp.eye(10)[:2]

    def loss_fn(model, inputs, labels_onehot):
        logits = model(inputs)
        return optax.softmax_cross_entropy(logits, labels_onehot).mean()

    # Lower using nnx.jit via make_jaxpr → then jit().lower().
    # We use a split-state approach to get a pure function that jax.jit can lower.
    graphdef, state = nnx.split(model)
    opt_graphdef, opt_state = nnx.split(optimizer)

    def pure_train_step(state, opt_state, inputs, labels_onehot):
        model = nnx.merge(graphdef, state)
        optimizer = nnx.merge(opt_graphdef, opt_state)

        def loss_fn_inner(model, inputs, labels_onehot):
            logits = model(inputs)
            return optax.softmax_cross_entropy(logits, labels_onehot).mean()

        loss, grads = nnx.value_and_grad(loss_fn_inner)(model, inputs, labels_onehot)
        optimizer.update(model, grads)
        _, new_state = nnx.split(model)
        _, new_opt_state = nnx.split(optimizer)
        return loss, new_state, new_opt_state

    lowered = jax.jit(pure_train_step).lower(
        state, opt_state, inputs, labels_onehot
    )
    return lowered.as_text()


def count_ops(stablehlo_text: str) -> Counter:
    """Count op occurrences by matching op-name tokens."""
    ops = re.findall(r'\b(?:stablehlo|chlo|mhlo)\.\w+', stablehlo_text)
    return Counter(ops)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--top", type=int, default=20,
                        help="Number of top ops to display (default: 20)")
    parser.add_argument("--filter", type=str, default=None,
                        help="Only show ops matching this substring")
    parser.add_argument("--save", type=str, default=None,
                        help="Save full StableHLO text to this file")
    args = parser.parse_args()

    print("Lowering ResNet18 train_step to StableHLO...", flush=True)
    text = get_resnet_stablehlo()

    if args.save:
        with open(args.save, "w") as f:
            f.write(text)
        print(f"Saved StableHLO to {args.save}")

    counts = count_ops(text)

    ops = [(op, n) for op, n in counts.most_common() if
           args.filter is None or args.filter in op]

    total = sum(n for _, n in ops)
    top = ops[:args.top]

    print(f"\nTop {len(top)} StableHLO ops in ResNet18 train_step "
          f"({'filtered: ' + args.filter if args.filter else 'all'}, "
          f"{sum(counts.values())} total ops):\n")
    print(f"  {'Count':>7}  {'% of total':>10}  Op")
    print(f"  {'-----':>7}  {'----------':>10}  --")
    for op, n in top:
        pct = 100.0 * n / sum(counts.values())
        print(f"  {n:>7}  {pct:>9.1f}%  {op}")

    print(f"\n  Total ops shown: {sum(n for _, n in top)} / {sum(counts.values())}")


if __name__ == "__main__":
    main()
