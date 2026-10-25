"""Import the saved official-baseline benchmark scores into MLflow.

Run from the Track1 directory after scoring the baseline predictions:
    python experiments/benchmark/log_baselines_to_mlflow.py
"""
from __future__ import annotations

from pathlib import Path
import sys

TRACK1_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(TRACK1_ROOT / "experiments"))

from benchmark.log_benchmark_scores_to_mlflow import main as attach_scores  # noqa: E402


BASELINES = {
    "fno_fp16": ("FNO", "sim_real_fno_fp16.pth"),
    "cno": ("CNO", "sim_real_cno.pth"),
    "transolver": ("Transolver", "sim_real_transolver.pth"),
}
SCORE_DIR = TRACK1_ROOT / "experiments" / "results" / "scores"


def main() -> None:
    # Reuse the same model runs instead of creating duplicate baseline runs.
    # Shared importer accepts historical time_score files and emits the new key.
    attach_scores(["--models", *BASELINES, *sys.argv[1:]])


if __name__ == "__main__":
    main()
