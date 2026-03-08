#!/usr/bin/env python3
"""Compare two pytest-benchmark JSON result files.

Usage:
    # Compare a specific file against the most recent clean baseline:
    uv run scripts/benchmark_compare.py path/to/new.json

    # Compare two specific files:
    uv run scripts/benchmark_compare.py path/to/new.json path/to/baseline.json

    # Set the significance threshold (default: 1.0 std-dev of baseline):
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


def find_baseline() -> Path:
    """Return the most recent non-dirty benchmark file."""
    candidates = sorted(
        (p for p in BENCHMARKS_DIR.glob("*.json") if "_dirty" not in p.name),
        reverse=True,
    )
    if not candidates:
        sys.exit(
            f"No clean baseline found in {BENCHMARKS_DIR}. "
            "Run scripts/benchmark.sh on a clean commit first."
        )
    return candidates[0]


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


def compare(new_path: Path, baseline_path: Path, threshold: float) -> None:
    new_data = load(new_path)
    base_data = load(baseline_path)

    print(f"New:      {new_path.name}")
    print(f"Baseline: {baseline_path.name}")
    print(f"Threshold: {threshold:.1f}σ of baseline")
    print()

    common = sorted(set(new_data) & set(base_data))
    only_new = sorted(set(new_data) - set(base_data))
    only_base = sorted(set(base_data) - set(new_data))

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
        help="Baseline JSON to compare against (default: most recent clean file)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=1.0,
        help="Minimum change in baseline σ units to report (default: 1.0)",
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
        # Most recent clean file that is not the new file itself.
        candidates = sorted(
            (
                p
                for p in BENCHMARKS_DIR.glob("*.json")
                if "_dirty" not in p.name and p.resolve() != new_path.resolve()
            ),
            reverse=True,
        )
        if not candidates:
            sys.exit(
                "No clean baseline found (other than the new file). "
                "Run scripts/benchmark.sh on a clean commit first."
            )
        baseline_path = candidates[0]
    else:
        baseline_path = args.baseline

    compare(new_path, baseline_path, args.threshold)


if __name__ == "__main__":
    main()
