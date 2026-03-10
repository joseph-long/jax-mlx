#!/usr/bin/env python3
"""Summarize JAX upstream pytest JUnit XML output with failed-test selectors.

Usage:
  uv run python scripts/summarize_jax_tests.py [RUN_DIR]

If RUN_DIR is omitted, the newest .benchmarks/jax_tests_* directory is used.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
import re
import sys
import xml.etree.ElementTree as ET


REPO_ROOT = Path(__file__).resolve().parent.parent
BENCHMARKS_DIR = REPO_ROOT / ".benchmarks"


def latest_run_dir() -> Path:
    runs = sorted(
        (p for p in BENCHMARKS_DIR.glob("jax_tests_*") if p.is_dir()),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not runs:
        raise FileNotFoundError(
            f"No jax_tests_* run directories found under {BENCHMARKS_DIR}"
        )
    return runs[0]


def split_classname(classname: str) -> tuple[str, list[str]]:
    parts = classname.split(".")
    class_parts: list[str] = []
    while parts and re.match(r"[A-Z]", parts[-1]):
        class_parts.insert(0, parts.pop())
    module_path = "/".join(parts) + ".py" if parts else "<unknown>"
    return module_path, class_parts


def build_selector(file_attr: str | None, classname: str | None, name: str) -> str:
    if file_attr:
        module_path = file_attr
        class_parts: list[str] = []
        if classname:
            _, class_parts = split_classname(classname)
    elif classname:
        module_path, class_parts = split_classname(classname)
    else:
        module_path = "<unknown>"
        class_parts = []

    selector = module_path
    for cls in class_parts:
        selector += f"::{cls}"
    selector += f"::{name}"
    return selector


def parse_junit_file(path: Path) -> dict:
    try:
        root = ET.parse(path).getroot()
    except ET.ParseError as exc:
        return {
            "parse_error": str(exc),
            "file": str(path),
            "totals": {"passed": 0, "failed": 0, "errors": 0, "skipped": 0, "total": 0},
            "failures": [],
        }

    testcases = list(root.iter("testcase"))
    totals = Counter({"passed": 0, "failed": 0, "errors": 0, "skipped": 0, "total": 0})
    failures: list[dict] = []

    for tc in testcases:
        totals["total"] += 1
        name = tc.attrib.get("name", "<unknown>")
        classname = tc.attrib.get("classname")
        file_attr = tc.attrib.get("file")
        selector = build_selector(file_attr, classname, name)

        failure = tc.find("failure")
        error = tc.find("error")
        skipped = tc.find("skipped")
        if failure is not None:
            totals["failed"] += 1
            msg = (failure.attrib.get("message") or (failure.text or "")).strip().splitlines()
            failures.append(
                {"kind": "failed", "selector": selector, "message": msg[0] if msg else ""}
            )
        elif error is not None:
            totals["errors"] += 1
            msg = (error.attrib.get("message") or (error.text or "")).strip().splitlines()
            failures.append(
                {"kind": "error", "selector": selector, "message": msg[0] if msg else ""}
            )
        elif skipped is not None:
            totals["skipped"] += 1
        else:
            totals["passed"] += 1

    return {"file": str(path), "totals": dict(totals), "failures": failures}


def load_exit_codes(run_dir: Path) -> dict[str, int]:
    path = run_dir / "exit_codes.tsv"
    if not path.exists():
        return {}
    out: dict[str, int] = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        test_file, rc = line.split("\t")
        out[test_file] = int(rc)
    return out


def infer_crash_selector(log_path: Path, test_file: str) -> str:
    if not log_path.exists():
        return f"{test_file}::<process_crash>"
    text = log_path.read_text(errors="replace")
    m = re.search(r"in (test[^\s]+)\n", text)
    if m:
        return f"{test_file}::{m.group(1)}"
    return f"{test_file}::<process_crash>"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "run_dir",
        nargs="?",
        type=Path,
        help="Directory containing *.junit.xml (default: newest .benchmarks/jax_tests_*)",
    )
    args = parser.parse_args()

    run_dir = args.run_dir.resolve() if args.run_dir else latest_run_dir()
    if not run_dir.exists():
        print(f"Run directory does not exist: {run_dir}", file=sys.stderr)
        return 2

    xml_files = sorted(run_dir.glob("*.junit.xml"))
    if not xml_files:
        print(f"No *.junit.xml files found in {run_dir}", file=sys.stderr)
        return 2

    aggregate = Counter({"passed": 0, "failed": 0, "errors": 0, "skipped": 0, "total": 0})
    parsed: list[dict] = []
    all_failures: list[dict] = []
    parse_errors: list[str] = []
    crash_entries: list[dict] = []

    for xml_file in xml_files:
        result = parse_junit_file(xml_file)
        parsed.append(result)
        if "parse_error" in result:
            parse_errors.append(f"{xml_file}: {result['parse_error']}")
        totals = result["totals"]
        aggregate.update(totals)
        all_failures.extend(result["failures"])

    exit_codes = load_exit_codes(run_dir)
    for test_file, rc in exit_codes.items():
        if rc == 0:
            continue
        safe_name = test_file.replace("/", "_").replace(".", "_")
        xml_path = run_dir / f"{safe_name}.junit.xml"
        if xml_path.exists():
            continue
        log_path = run_dir / f"{safe_name}.log"
        selector = infer_crash_selector(log_path, test_file)
        crash_entries.append(
            {
                "kind": "crash",
                "selector": selector,
                "message": f"pytest process exited with code {rc} before JUnit XML was written",
                "test_file": test_file,
                "exit_code": rc,
            }
        )

    all_failures.extend(crash_entries)

    selectors = sorted({f["selector"] for f in all_failures})

    print(f"Run dir: {run_dir}")
    print(
        "Totals: "
        f"{aggregate['passed']} passed, "
        f"{aggregate['failed']} failed, "
        f"{aggregate['errors']} errors, "
        f"{aggregate['skipped']} skipped, "
        f"{aggregate['total']} total"
    )

    if parse_errors:
        print("Parse errors:")
        for err in parse_errors:
            print(f"  - {err}")

    if crash_entries:
        print("Crash-only files (no JUnit XML):")
        for c in crash_entries:
            print(f"  - {c['selector']} [{c['message']}]")

    if all_failures:
        print("\nFailed/error selectors:")
        for selector in selectors:
            print(f"  - {selector}")
    else:
        print("\nNo failures or errors found.")

    summary = {
        "run_dir": str(run_dir),
        "totals": dict(aggregate),
        "parse_errors": parse_errors,
        "crashes": crash_entries,
        "failures": all_failures,
        "failed_selectors": selectors,
        "files": parsed,
    }
    out_path = run_dir / "summary.json"
    out_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"\nWrote summary JSON: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
