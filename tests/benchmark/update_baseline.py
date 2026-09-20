#!/usr/bin/env python3
"""
Manually regenerates ragas_baseline.json from a fresh benchmark run.

This is a DELIBERATE, human-triggered action — never run automatically
by CI — so a real regression can't quietly become the new "normal."
Requires --confirm to actually write the file; without it, prints the
new scores and the diff against the current baseline and exits.

Usage:
    python tests/benchmark/update_baseline.py            # dry run, just shows the diff
    python tests/benchmark/update_baseline.py --confirm   # writes the new baseline
"""
import argparse
import json

from tests.benchmark.runner import BASELINE_PATH, load_baseline, run_benchmark


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--confirm", action="store_true", help="Actually write the new baseline file.")
    args = parser.parse_args()

    old_baseline = load_baseline()
    new_scores = run_benchmark()

    print("Current baseline vs new run:")
    for metric in sorted(set(old_baseline) | set(new_scores)):
        old = old_baseline.get(metric, "—")
        new = new_scores.get(metric, "—")
        print(f"  {metric}: {old} -> {new}")

    if not args.confirm:
        print("\nDry run only — pass --confirm to write tests/benchmark/ragas_baseline.json")
        return

    BASELINE_PATH.write_text(json.dumps(new_scores, indent=2) + "\n")
    print(f"\nWrote new baseline to {BASELINE_PATH}")


if __name__ == "__main__":
    main()
