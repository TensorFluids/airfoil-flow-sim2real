"""Import the saved official-baseline benchmark scores into MLflow.

Run from the Track1 directory after scoring the baseline predictions:
    python experiments/benchmark/log_baselines_to_mlflow.py
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

TRACK1_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(TRACK1_ROOT / "experiments"))

from mlflow_tracking import training_run  # noqa: E402


BASELINES = {
    "fno_fp16": ("FNO", "sim_real_fno_fp16.pth"),
    "cno": ("CNO", "sim_real_cno.pth"),
    "transolver": ("Transolver", "sim_real_transolver.pth"),
}
SCORE_DIR = TRACK1_ROOT / "experiments" / "results" / "scores"
METRIC_KEYS = ("rel_l2_score", "tke_score", "mvpe_score", "time_score", "sps_score")


def main() -> None:
    for model_key, (model_name, checkpoint_name) in BASELINES.items():
        score_path = SCORE_DIR / model_key / "scores.json"
        if not score_path.is_file():
            raise FileNotFoundError(
                f"Missing {score_path}. Run and score the baseline benchmark first."
            )

        with score_path.open(encoding="utf-8") as f:
            scores = json.load(f)
        metrics = {key: float(scores[key]) for key in METRIC_KEYS if key in scores}
        if len(metrics) != len(METRIC_KEYS):
            missing = sorted(set(METRIC_KEYS) - set(metrics))
            raise ValueError(f"{score_path} is missing metrics: {', '.join(missing)}")

        run_name = f"baseline-{model_key}"
        with training_run(
            run_name,
            {
                "model": model_name,
                "baseline_weight_file": checkpoint_name,
                "weights": "provided sim_real_ft baseline",
                "evaluation_trajectory": "3750_0",
                "evaluation_windows": 42,
            },
            tags={
                "kind": "baseline",
                "baseline_id": model_key,
                "evaluation_scope": "single_public_example_trajectory",
                "timing_device": "cpu",
            },
        ):
            import mlflow

            mlflow.log_metrics(metrics)
            mlflow.log_artifact(str(score_path), artifact_path="source_scores")
        print(f"Logged {model_name} baseline: {metrics}")


if __name__ == "__main__":
    main()
