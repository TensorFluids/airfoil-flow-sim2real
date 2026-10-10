"""Build an untrained equal-weight control from cached baseline predictions.

Run from Track1:
    python experiments/benchmark/run_average_benchmark.py

All three caches are physical-space predictions generated from the same
ref/samples.npz by run_benchmark.py. No model weights are loaded. Inference
time is estimated as the sum of the cached baseline times plus the measured
averaging overhead; it is not a fresh end-to-end timing measurement.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time

import numpy as np


RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
BASE_MODELS = ("fno_fp16", "cno", "transolver")
BATCH_SIZE = 8


def uniform_average(predictions: list[np.ndarray]) -> np.ndarray:
    """Average three aligned (N,T,H,W,3) fields with no fitted parameters."""
    if len(predictions) != 3:
        raise ValueError("The control requires exactly three baseline predictions.")
    shape = predictions[0].shape
    if len(shape) != 5 or shape[-1] != 3 or any(size <= 0 for size in shape):
        raise ValueError(f"Expected nonempty (N,T,H,W,3) predictions, got {shape}.")
    for prediction in predictions:
        if prediction.shape != shape:
            raise ValueError("Baseline prediction shapes do not match.")
        if not np.isfinite(prediction).all():
            raise ValueError("Baseline predictions contain non-finite values.")
        if np.any(prediction[..., 2] != 0.0):
            raise ValueError("Expected zero pressure channels for the real-data caches.")
    return np.mean(np.stack(predictions, axis=0), axis=0, dtype=np.float32)


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    predictions = []
    base_times = {}
    hashes = {}
    for model in BASE_MODELS:
        path = RESULTS_DIR / model / "predictions.npz"
        with np.load(path, allow_pickle=False) as data:
            predictions.append(data["prediction"].astype(np.float32))
            mean_time = float(np.asarray(data["mean_t_neural_s"]).item())
        if not np.isfinite(mean_time) or mean_time <= 0.0:
            raise ValueError(f"Invalid cached inference time for {model}: {mean_time}")
        base_times[model] = mean_time
        hashes[model] = file_hash(path)

    with np.load(RESULTS_DIR / "ref" / "targets.npz", allow_pickle=False) as data:
        target_shape = data["target"].shape
    with np.load(RESULTS_DIR / "ref" / "samples.npz", allow_pickle=False) as data:
        sample_count = data["input"].shape[0]
    averaged = uniform_average(predictions)
    if averaged.shape != target_shape or averaged.shape[0] != sample_count:
        raise ValueError("Cached predictions do not match the reference window shapes.")

    # Measure only numerical averaging, excluding validation, file I/O, and
    # compression. Weighted by window count, including the last smaller batch.
    elapsed = []
    for _ in range(5):
        start = time.perf_counter()
        for i in range(0, sample_count, BATCH_SIZE):
            np.mean(
                np.stack([p[i:i + BATCH_SIZE] for p in predictions], axis=0),
                axis=0,
                dtype=np.float32,
            )
        elapsed.append((time.perf_counter() - start) / sample_count)
    averaging_time = float(np.median(elapsed))
    estimated_total_time = sum(base_times.values()) + averaging_time

    out_dir = RESULTS_DIR / "uniform_average"
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_dir / "predictions.npz",
        prediction=averaged,
        mean_t_neural_s=np.float32(estimated_total_time),
    )
    metadata = {
        "model": "uniform_average",
        "components": list(BASE_MODELS),
        "weights": [1.0 / 3.0] * 3,
        "trainable_parameters": 0,
        "prediction_space": "physical",
        "evaluation_windows": sample_count,
        "uncertainty_intervals": "scorer_default_same_as_baseline_and_mixer_caches",
        "timing_method": "estimated_sum_cached_base_times_plus_measured_average",
        "cached_base_seconds_per_sample": base_times,
        "averaging_seconds_per_sample": averaging_time,
        "estimated_total_seconds_per_sample": estimated_total_time,
        "source_cache_sha256": hashes,
        "reference_samples_sha256": file_hash(RESULTS_DIR / "ref" / "samples.npz"),
        "reference_targets_sha256": file_hash(RESULTS_DIR / "ref" / "targets.npz"),
        "alignment_basis": "shared_ref_samples_and_run_benchmark_window_order",
    }
    # Generated results are intentionally excluded from version control.
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Saved {out_dir / 'predictions.npz'}; shape={averaged.shape}")
    print(f"Averaging overhead: {averaging_time:.6f} s/window")
    print(f"Estimated total inference time: {estimated_total_time:.6f} s/window")


if __name__ == "__main__":
    main()
