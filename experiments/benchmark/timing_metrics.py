"""Metric naming and validation for competition timing and fresh self timing."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import statistics


COMPETITION_METRIC_KEYS = (
    "rel_l2_score", "tke_score", "mvpe_score", "timescore_competition", "sps_score",
)
SELF_METRIC_KEYS = ("timescore_self", "latency_ms_self")
DEFAULT_TIMING_PATH = Path(__file__).resolve().parents[1] / "results" / "timing_self.json"
TIMING_SCHEMA_VERSION = 1
TIMING_BOUNDARY = "cpu_physical_input_to_cpu_physical_output_including_all_models_excluding_loading_and_file_io"


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def competition_metrics(raw: dict) -> dict[str, float]:
    """Read old score files, but only emit the explicitly named competition key."""
    values = dict(raw)
    if "time_score" in values:
        if "timescore_competition" in values and float(values["time_score"]) != float(values["timescore_competition"]):
            raise ValueError("Conflicting legacy and renamed competition timing metrics.")
        values["timescore_competition"] = values["time_score"]
    missing = set(COMPETITION_METRIC_KEYS) - values.keys()
    if missing:
        raise ValueError(f"Missing competition metrics: {sorted(missing)}")
    result = {key: float(values[key]) for key in COMPETITION_METRIC_KEYS}
    if any(not math.isfinite(value) or not 0 <= value <= 100 for value in result.values()):
        raise ValueError("Competition scores must be finite and in [0, 100].")
    return result


def self_metrics(report: dict, model: str) -> dict[str, float]:
    """The ratio uses medians measured in the SAME report, never cached timings."""
    baseline = float(report["models"]["fno_fp16"]["median_latency_s"])
    elapsed = float(report["models"][model]["median_latency_s"])
    if any(not math.isfinite(t) or t <= 0 for t in (baseline, elapsed)):
        raise ValueError("Self timing requires positive finite measured latency.")
    return {"timescore_self": baseline / elapsed, "latency_ms_self": 1000.0 * elapsed}


def load_self_timing(path: Path, root: Path, samples_path: Path) -> dict:
    """Reject mixing input sets, changed weights, code, or incomplete runs."""
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("schema_version") != TIMING_SCHEMA_VERSION:
        raise ValueError("Unsupported self timing report version.")
    protocol = report["protocol"]
    if protocol.get("boundary") != TIMING_BOUNDARY or protocol.get("batch_size") != 1:
        raise ValueError("Self timing must measure a complete batch-one forecast.")
    if protocol.get("samples_sha256") != file_hash(samples_path):
        raise ValueError("Timing input data has changed. Rerun benchmark_timing.py.")
    if protocol.get("warmup", 0) < 1 or protocol.get("repeats", 0) < 2:
        raise ValueError("Self timing requires warmup and repeated measurements.")
    if not protocol.get("code_sha256"):
        raise ValueError("Timing code fingerprints are missing.")
    for relative, expected in protocol["code_sha256"].items():
        if file_hash(root / relative) != expected:
            raise ValueError(f"Timing code has changed: {relative}. Rerun benchmark_timing.py.")
    if "fno_fp16" not in report["models"]:
        raise ValueError("A fresh FNO reference measurement is required.")
    for name, row in report["models"].items():
        if not row.get("checkpoint_sha256"):
            raise ValueError(f"Checkpoint fingerprints are missing for {name}.")
        for relative, expected in row["checkpoint_sha256"].items():
            if file_hash(root / relative) != expected:
                raise ValueError(f"Timed checkpoint has changed: {relative}. Rerun benchmark_timing.py.")
        samples = row["latencies_s"]
        if len(samples) != protocol["repeats"] or any(not math.isfinite(t) or t <= 0 for t in samples):
            raise ValueError(f"Invalid repeated timings for {name}.")
        if not math.isclose(statistics.median(samples), row["median_latency_s"], rel_tol=1e-12):
            raise ValueError(f"Median does not match measured timings for {name}.")
        self_metrics(report, name)
    return report


def optional_self_timing(path: Path | None, root: Path, skip: bool = False):
    if skip:
        return None, None
    actual = path if path is not None else root / "experiments/results/timing_self.json"
    if not actual.is_file():
        if path is not None:
            raise FileNotFoundError(actual)
        return None, None
    return load_self_timing(actual, root, root / "experiments/results/ref/samples.npz"), actual
