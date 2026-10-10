"""Attach saved benchmark scores to each model's existing MLflow run.

Run from the Track1 directory after scoring:
    python experiments/benchmark/log_benchmark_scores_to_mlflow.py

The scorer exposes five subscores but does not publish the leaderboard's
final_score formula. This script also logs their arithmetic mean as
``mean_subscore``; it is a derived, unofficial summary, not the official final
score. The optional --archive-standalone-runs flag soft-deletes the standalone
benchmark-score-* evaluation runs created by earlier versions of this script.
They remain restorable from MLflow's Deleted Runs view.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

TRACK1_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(TRACK1_ROOT / "experiments"))

import mlflow  # noqa: E402
from mlflow.tracking import MlflowClient  # noqa: E402
from mlflow_tracking import DEFAULT_TRACKING_DIR  # noqa: E402


SCORE_DIR = TRACK1_ROOT / "experiments" / "results" / "scores"
MODEL_RUNS = {
    "fno_fp16": "baseline-fno_fp16",
    "cno": "baseline-cno",
    "transolver": "baseline-transolver",
    "ensemble": "stacked-ensemble",
    "fst_hmoe": "fst-hmoe-mixer",
}
METRIC_KEYS = ("rel_l2_score", "tke_score", "mvpe_score", "time_score", "sps_score")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--archive-standalone-runs",
        action="store_true",
        help="Soft-delete benchmark-score-* runs created by the earlier importer.",
    )
    args = parser.parse_args()

    tracking_uri = os.environ.get(
        "MLFLOW_TRACKING_URI", DEFAULT_TRACKING_DIR.resolve().as_uri()
    )
    experiment_name = os.environ.get(
        "MLFLOW_EXPERIMENT_NAME", "airfoil-flow-forecasting"
    )
    mlflow.set_tracking_uri(tracking_uri)
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise RuntimeError(f"MLflow experiment not found: {experiment_name}")

    client = MlflowClient(tracking_uri=tracking_uri)
    runs = client.search_runs([experiment.experiment_id], max_results=1000)
    by_name: dict[str, list] = {}
    for run in runs:
        by_name.setdefault(run.info.run_name, []).append(run)

    # Resolve all destinations before writing anything, so ambiguous/missing
    # model runs cannot leave a partial update.
    destinations = {}
    for model, run_name in MODEL_RUNS.items():
        matches = by_name.get(run_name, [])
        if len(matches) != 1:
            raise RuntimeError(
                f"Expected exactly one MLflow run named {run_name!r}; found {len(matches)}."
            )
        destinations[model] = matches[0]

    loaded_scores = {}
    for model in MODEL_RUNS:
        score_path = SCORE_DIR / model / "scores.json"
        if not score_path.is_file():
            raise FileNotFoundError(f"Missing score file: {score_path}")
        with score_path.open(encoding="utf-8") as f:
            raw_scores = json.load(f)
        scores = {key: float(raw_scores[key]) for key in METRIC_KEYS if key in raw_scores}
        missing = sorted(set(METRIC_KEYS) - set(scores))
        if missing:
            raise ValueError(f"{score_path} is missing metrics: {', '.join(missing)}")
        scores["mean_subscore"] = sum(scores.values()) / len(METRIC_KEYS)
        loaded_scores[model] = (score_path, scores)

    for model, run in destinations.items():
        _, scores = loaded_scores[model]
        for key, value in scores.items():
            client.log_metric(run.info.run_id, key, value, timestamp=int(time.time() * 1000), step=0)
        client.set_tag(run.info.run_id, "benchmark_evaluated", "true")
        client.set_tag(run.info.run_id, "benchmark_evaluation_scope", "single_public_example_trajectory")
        client.set_tag(run.info.run_id, "mean_subscore_definition", "arithmetic_mean_of_five_subscores_unofficial")
        print(f"Attached {len(scores)} benchmark metrics to {run.info.run_name}")

    standalone_runs = [
        run for run in runs
        if run.data.tags.get("kind") == "benchmark_evaluation"
        and run.info.run_name.startswith("benchmark-score-")
    ]
    if standalone_runs and not args.archive_standalone_runs:
        print(
            f"Found {len(standalone_runs)} standalone benchmark-score runs. "
            "They were left untouched; pass --archive-standalone-runs to soft-delete them."
        )
    elif standalone_runs:
        for run in standalone_runs:
            client.delete_run(run.info.run_id)
        print(f"Archived {len(standalone_runs)} standalone benchmark-score runs (restorable).")


if __name__ == "__main__":
    main()
