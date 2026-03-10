#!/usr/bin/env python3
"""Compare two pytest-benchmark JSON result files.

Usage:
    # Compare a specific file against the oldest clean baseline:
    uv run scripts/benchmark_compare.py path/to/new.json

    # Compare two specific files:
    uv run scripts/benchmark_compare.py path/to/new.json path/to/baseline.json

    # Set the significance threshold (default: 2.0 std-dev of baseline):
    uv run scripts/benchmark_compare.py new.json --threshold 1.5

Output: benchmarks that changed by more than `threshold` standard deviations of the
baseline mean, split into "faster" and "slower" groups, sorted by effect size.
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path


BENCHMARKS_DIR = Path(__file__).parent.parent / ".benchmarks"


def load(path: Path) -> dict[str, dict]:
    """Load a benchmark JSON and return {name: stats_dict}."""
    with open(path) as f:
        data = json.load(f)
    return {b["name"]: b["stats"] for b in data["benchmarks"]}


def sigfig(x: float, n: int = 3) -> str:
    """Format x to n significant figures."""
    if x == 0:
        return "0"
    magnitude = math.floor(math.log10(abs(x)))
    decimals = max(0, n - 1 - magnitude)
    return f"{x:.{decimals}f}"


def human_time(seconds: float) -> str:
    """Format a duration in human-readable units."""
    if seconds < 1e-6:
        return f"{seconds * 1e9:.1f} ns"
    if seconds < 1e-3:
        return f"{seconds * 1e6:.1f} µs"
    if seconds < 1.0:
        return f"{seconds * 1e3:.1f} ms"
    return f"{seconds:.3f} s"


def compare(
    new_path: Path, baseline_path: Path, threshold: float, include_cpu: bool
) -> None:
    new_data = load(new_path)
    base_data = load(baseline_path)

    print(f"New:      {new_path.name}")
    print(f"Baseline: {baseline_path.name}")
    print(f"Threshold: {threshold:.1f}σ of baseline")
    if not include_cpu:
        print("Filter:   excluding CPU benchmark entries")
    print()

    common = sorted(set(new_data) & set(base_data))
    only_new = sorted(set(new_data) - set(base_data))
    only_base = sorted(set(base_data) - set(new_data))

    if not include_cpu:
        def is_cpu_name(name: str) -> bool:
            return "[cpu-benchmark." in name

        common = [name for name in common if not is_cpu_name(name)]
        only_new = [name for name in only_new if not is_cpu_name(name)]
        only_base = [name for name in only_base if not is_cpu_name(name)]

    faster = []
    slower = []
    unchanged = 0

    for name in common:
        ns = new_data[name]
        bs = base_data[name]
        base_mean = bs["mean"]
        base_std = bs["stddev"]

        if base_std == 0:
            continue

        delta = ns["mean"] - base_mean          # positive → slower
        sigma = delta / base_std                 # in units of baseline σ

        if abs(sigma) < threshold:
            unchanged += 1
            continue

        ratio = ns["mean"] / base_mean           # >1 → slower, <1 → faster
        pct = (ratio - 1) * 100

        entry = {
            "name": name,
            "sigma": sigma,
            "ratio": ratio,
            "pct": pct,
            "new_mean": ns["mean"],
            "base_mean": base_mean,
            "base_std": base_std,
        }
        if delta < 0:
            faster.append(entry)
        else:
            slower.append(entry)

    faster.sort(key=lambda e: e["sigma"])        # most negative first
    slower.sort(key=lambda e: e["sigma"], reverse=True)

    def print_group(entries: list, label: str, arrow: str) -> None:
        if not entries:
            return
        print(f"{'─' * 72}")
        print(f"  {label} ({len(entries)} benchmark{'s' if len(entries) != 1 else ''})")
        print(f"{'─' * 72}")
        for e in entries:
            tag = f"{arrow} {abs(e['pct']):.1f}%  ({e['sigma']:+.1f}σ)"
            timing = (
                f"{human_time(e['new_mean'])} vs {human_time(e['base_mean'])} baseline"
            )
            print(f"  {e['name']}")
            print(f"    {tag:30s}  {timing}")
        print()

    print_group(faster, "FASTER", "↓")
    print_group(slower, "SLOWER", "↑")

    total = len(common)
    print(f"{'─' * 72}")
    print(f"  {total} benchmarks compared  |  "
          f"{len(faster)} faster  |  {len(slower)} slower  |  {unchanged} unchanged")
    if only_new:
        print(f"  {len(only_new)} new (not in baseline): {', '.join(only_new[:3])}"
              + (" ..." if len(only_new) > 3 else ""))
    if only_base:
        print(f"  {len(only_base)} removed (not in new): {', '.join(only_base[:3])}"
              + (" ..." if len(only_base) > 3 else ""))
    print()

    # Report MLX speedup vs CPU for matching benchmarks in the same new run.
    pair_rows = []
    for name, mlx_stats in sorted(new_data.items()):
        if "[mlx-benchmark." not in name:
            continue
        cpu_name = name.replace("[mlx-benchmark.", "[cpu-benchmark.")
        cpu_stats = new_data.get(cpu_name)
        if cpu_stats is None:
            continue
        mlx_mean = mlx_stats["mean"]
        cpu_mean = cpu_stats["mean"]
        if mlx_mean <= 0 or cpu_mean <= 0:
            continue
        speedup = cpu_mean / mlx_mean
        pair_rows.append(
            {
                "name": name,
                "mlx_mean": mlx_mean,
                "cpu_mean": cpu_mean,
                "speedup": speedup,
            }
        )

    if pair_rows:
        pair_rows.sort(key=lambda e: e["speedup"], reverse=True)
        print(f"{'─' * 72}")
        print(
            "  MLX VS CPU (same run; speedup = CPU mean / MLX mean)"
            f" ({len(pair_rows)} benchmarks)"
        )
        print(f"{'─' * 72}")
        for e in pair_rows:
            if e["speedup"] >= 1:
                tag = f"{e['speedup']:.2f}x faster"
            else:
                tag = f"{(1.0 / e['speedup']):.2f}x slower"
            print(f"  {e['name']}")
            print(
                f"    {tag:18s}  MLX {human_time(e['mlx_mean'])}  |  CPU {human_time(e['cpu_mean'])}"
            )
        print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "new",
        nargs="?",
        type=Path,
        help="New benchmark JSON to evaluate (default: most recent file in .benchmarks/)",
    )
    parser.add_argument(
        "baseline",
        nargs="?",
        type=Path,
        help="Baseline JSON to compare against (default: oldest clean file)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=2.0,
        help="Minimum change in baseline σ units to report (default: 2.0)",
    )
    parser.add_argument(
        "--include-cpu",
        action="store_true",
        help="Include CPU benchmark entries in comparison output",
    )
    args = parser.parse_args()

    # Resolve new file.
    if args.new is None:
        candidates = sorted(BENCHMARKS_DIR.glob("*.json"), reverse=True)
        if not candidates:
            sys.exit(f"No benchmark files found in {BENCHMARKS_DIR}")
        new_path = candidates[0]
    else:
        new_path = args.new

    # Resolve baseline.
    if args.baseline is None:
        # Oldest clean file that is not the new file itself.
        candidates = sorted(
            (
                p
                for p in BENCHMARKS_DIR.glob("*.json")
                if "_dirty" not in p.name and p.resolve() != new_path.resolve()
            ),
        )
        if not candidates:
            sys.exit(
                "No clean baseline found (other than the new file). "
                "Run scripts/benchmark.sh on a clean commit first."
            )
        baseline_path = candidates[0]
    else:
        baseline_path = args.baseline

    compare(new_path, baseline_path, args.threshold, include_cpu=args.include_cpu)


if __name__ == "__main__":
    main()
