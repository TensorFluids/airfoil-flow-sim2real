"""Shared local MLflow helpers for Track 1 experiments."""
from __future__ import annotations

from contextlib import contextmanager
import os
from pathlib import Path
from typing import Any, Iterator, Mapping

import mlflow


TRACK1_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRACKING_DIR = TRACK1_ROOT / "mlflow_runs"


def _mlflow_params(params: Mapping[str, Any]) -> dict[str, str | int | float | bool]:
    """Keep portable scalar settings; do not record machine-specific paths."""
    result: dict[str, str | int | float | bool] = {}
    for key, value in params.items():
        lowered = key.lower()
        if any(part in lowered for part in ("path", "dir", "ckpt", "checkpoint")):
            continue
        if value is None or isinstance(value, (dict, list, tuple, set)):
            continue
        if isinstance(value, (str, int, float, bool)):
            result[key] = value
    return result


@contextmanager
def training_run(
    run_name: str,
    params: Mapping[str, Any],
    tags: Mapping[str, str] | None = None,
) -> Iterator[Any]:
    """Start a run in the configured experiment and log its scalar settings."""
    default_uri = DEFAULT_TRACKING_DIR.resolve().as_uri()
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", default_uri)
    experiment_name = os.environ.get(
        "MLFLOW_EXPERIMENT_NAME", "airfoil-flow-forecasting"
    )

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name=run_name, tags=dict(tags or {})) as run:
        mlflow.log_params(_mlflow_params(params))
        yield run


def log_training_params(params: Mapping[str, Any]) -> None:
    values = _mlflow_params(params)
    if values:
        mlflow.log_params(values)


def log_epoch_metrics(metrics: Mapping[str, float], epoch: int) -> None:
    mlflow.log_metrics({key: float(value) for key, value in metrics.items()}, step=epoch)


def log_training_artifact(path: str | os.PathLike[str]) -> None:
    artifact = Path(path)
    if artifact.is_file():
        mlflow.log_artifact(str(artifact), artifact_path="checkpoints")
