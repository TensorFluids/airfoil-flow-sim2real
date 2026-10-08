"""run_all.py – Master script: make_targets → run_benchmark → compare_models.

Run from C:\\Desktop\\RealPDE\\Track1:
    python experiments/benchmark/run_all.py

Pass --skip-benchmark to skip inference (useful when predictions already exist):
    python experiments/benchmark/run_all.py --skip-benchmark
"""

from __future__ import annotations

import argparse
import os
import runpy
import sys

BENCHMARK_DIR = os.path.dirname(os.path.abspath(__file__))


def run_step(script_name: str) -> None:
    script_path = os.path.join(BENCHMARK_DIR, script_name)
    print(f"\n{'#'*60}")
    print(f"# Step: {script_name}")
    print(f"{'#'*60}")
    # runpy.run_path executes the script in a fresh namespace with __name__ == '__main__'
    runpy.run_path(script_path, run_name="__main__")


def main() -> None:
    parser = argparse.ArgumentParser(description="RealPDE Track 1 benchmark pipeline")
    parser.add_argument(
        "--skip-benchmark", action="store_true",
        help="Skip inference step (use existing predictions.npz files)"
    )
    args = parser.parse_args()

    run_step("make_targets.py")

    if not args.skip_benchmark:
        run_step("run_benchmark.py")
    else:
        print("\n[skip] run_benchmark.py (--skip-benchmark flag set)")

    run_step("compare_models.py")

    print("\n✓ Pipeline complete.")


if __name__ == "__main__":
    main()
