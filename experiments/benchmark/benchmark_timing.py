"""Measure fresh, complete forecast latency and FNO-relative timescore_self.

    python experiments/benchmark/benchmark_timing.py
    python experiments/benchmark/benchmark_timing.py --models fno_fp16 ensemble

Inference only. No training, prediction-cache overwrites, or MLflow writes.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import json
import platform
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
for directory in (ROOT, ROOT / "_vendor", ROOT / "experiments"):
    sys.path.insert(0, str(directory))

import numpy as np
import torch
from torch import nn

from benchmark.timing_metrics import (
    DEFAULT_TIMING_PATH, TIMING_BOUNDARY, TIMING_SCHEMA_VERSION, file_hash, self_metrics,
)
from models.stacked_ensemble import SpatialAttentionMixer, _MEAN_IN, _STD_IN, _MEAN_TGT, _STD_TGT

MODELS = ("fno_fp16", "cno", "transolver", "uniform_average", "ensemble", "fst_hmoe")
BASE_FILES = {
    "fno_fp16": "checkpoints/sim_real_ft/sim_real_fno_fp16.pth",
    "cno": "checkpoints/sim_real_ft/sim_real_cno.pth",
    "transolver": "checkpoints/sim_real_ft/sim_real_transolver.pth",
}
MIXER_FILES = {
    "ensemble": "checkpoints/ensemble/best_ensemble_mixer.pth",
    "fst_hmoe": "checkpoints/fst_hmoe/best_fst_hmoe.pth",
}


def checkpoint_paths(name: str, root: Path) -> list[Path]:
    paths = [root / BASE_FILES[name]] if name in BASE_FILES else [root / value for value in BASE_FILES.values()]
    if name in MIXER_FILES:
        paths.append(root / MIXER_FILES[name])
    return paths


class ForecastPipeline(nn.Module):
    """Same raw-input/raw-output boundary for every model; no cached bases."""

    def __init__(self, bases, mixer=None):
        super().__init__()
        self.bases = nn.ModuleList(bases)
        self.mixer = mixer
        for name, values in (("mean_in", _MEAN_IN), ("std_in", _STD_IN),
                             ("mean_target", _MEAN_TGT), ("std_target", _STD_TGT)):
            self.register_buffer(name, torch.tensor(values, dtype=torch.float32))

    def forward(self, raw):
        normalized = (raw - self.mean_in) / self.std_in
        predictions = [base(normalized) for base in self.bases]
        if self.mixer is not None:
            prediction = self.mixer(torch.cat(predictions, dim=-1))
        elif len(predictions) == 1:
            prediction = predictions[0]
        else:
            prediction = torch.stack(predictions).mean(dim=0)
        physical = prediction * self.std_target + self.mean_target
        physical[..., 2] = 0.0
        return physical


def build_pipeline(name: str, device: str, root: Path) -> ForecastPipeline:
    from load_baseline import load_baseline

    base_names = [name] if name in BASE_FILES else list(BASE_FILES)
    bases = [load_baseline(str(root / BASE_FILES[key]), device=device)[0] for key in base_names]
    mixer = None
    if name in MIXER_FILES:
        checkpoint = torch.load(root / MIXER_FILES[name], map_location="cpu", weights_only=False)
        if name == "ensemble":
            mixer = SpatialAttentionMixer(hidden=checkpoint.get("mixer_hidden", 32))
            mixer.load_state_dict(checkpoint["mixer_state_dict"])
        else:
            from models.fst_hmoe_mixer import FSTHMoEConfig, FSTHMoEMixer
            mixer = FSTHMoEMixer(FSTHMoEConfig(**checkpoint["model_config"]))
            mixer.load_state_dict(checkpoint["model_state_dict"])
    pipeline = ForecastPipeline(bases, mixer).to(device).eval()
    pipeline.requires_grad_(False)
    return pipeline


def measure_pipeline(pipeline, inputs, device: str, warmup: int, repeats: int) -> dict:
    """Round-robin same windows for each model; include both CPU/GPU transfers."""
    def synchronize():
        if device.startswith("cuda"):
            torch.cuda.synchronize(torch.device(device))

    def predict(index):
        raw = torch.from_numpy(np.ascontiguousarray(inputs[index:index + 1])).to(device)
        return pipeline(raw).float().cpu().numpy()

    latencies = []
    with torch.inference_mode():
        for i in range(warmup):
            predict(i % len(inputs))
        synchronize()
        for i in range(repeats):
            synchronize()
            started = time.perf_counter()
            output = predict(i % len(inputs))
            synchronize()
            latencies.append(time.perf_counter() - started)
            if output.shape != inputs[:1].shape or not np.isfinite(output).all():
                raise ValueError("Forecast output is malformed or non-finite.")
    return {
        "median_latency_s": float(np.median(latencies)),
        "p25_latency_s": float(np.percentile(latencies, 25)),
        "p75_latency_s": float(np.percentile(latencies, 75)),
        "latencies_s": latencies,
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--num-threads", type=int, default=1)
    parser.add_argument("--samples", type=Path, default=ROOT / "experiments/results/ref/samples.npz")
    parser.add_argument("--output", type=Path, default=DEFAULT_TIMING_PATH)
    args = parser.parse_args(argv)
    if args.warmup < 1 or args.repeats < 2 or args.num_threads < 1:
        parser.error("Use warmup >= 1, repeats >= 2, and num-threads >= 1.")
    args.models = list(dict.fromkeys(["fno_fp16", *args.models]))
    return args


def main(argv=None):
    args = parse_args(argv)
    if torch.device(args.device).type not in ("cpu", "cuda"):
        raise ValueError("Use cpu or cuda[:index] for this timing protocol.")
    # Resolve all required inputs before beginning an expensive benchmark.
    checkpoint_hashes = {}
    for name in args.models:
        checkpoint_hashes[name] = {p.relative_to(ROOT).as_posix(): file_hash(p) for p in checkpoint_paths(name, ROOT)}
    with np.load(args.samples, allow_pickle=False) as data:
        inputs = np.asarray(data["input"], dtype=np.float32)
    if inputs.ndim != 5 or tuple(inputs.shape[1:]) != (20, 32, 64, 3) or len(inputs) == 0 or not np.isfinite(inputs).all():
        raise ValueError("Expected finite physical inputs with shape (N,20,32,64,3).")
    torch.set_num_threads(args.num_threads)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    code = [Path(__file__), Path(__file__).with_name("timing_metrics.py"), ROOT / "load_baseline.py"]
    code += sorted((ROOT / "experiments/models").glob("*.py"))
    code += sorted((ROOT / "rpde_baselines").rglob("*.py"))
    protocol = {
        "boundary": TIMING_BOUNDARY, "batch_size": 1,
        "warmup": args.warmup, "repeats": args.repeats, "num_threads": args.num_threads,
        "device": args.device, "gpu_name": torch.cuda.get_device_name(torch.device(args.device)) if args.device.startswith("cuda") else None,
        "platform": platform.platform(), "processor": platform.processor(),
        "torch_version": str(torch.__version__), "numpy_version": np.__version__,
        "cuda_version": torch.version.cuda, "cudnn_version": torch.backends.cudnn.version(),
        "precision": "float32_no_autocast_tf32_disabled", "cudnn_benchmark": False,
        "model_residency": "all_required_models_resident_for_each_timed_forecast",
        "samples_sha256": file_hash(args.samples), "samples_shape": list(inputs.shape),
        "timed_window_indices": [i % len(inputs) for i in range(args.repeats)],
        "code_sha256": {p.relative_to(ROOT).as_posix(): file_hash(p) for p in code},
    }
    report = {"schema_version": TIMING_SCHEMA_VERSION, "created_utc": datetime.now(timezone.utc).isoformat(), "protocol": protocol, "models": {}}
    for name in args.models:
        print(f"Timing {name}: batch=1, warmup={args.warmup}, repeats={args.repeats}, device={args.device}", flush=True)
        pipeline = build_pipeline(name, args.device, ROOT)
        try:
            row = measure_pipeline(pipeline, inputs, args.device, args.warmup, args.repeats)
        finally:
            del pipeline
            gc.collect()
            if args.device.startswith("cuda"):
                torch.cuda.empty_cache()
        row["checkpoint_sha256"] = checkpoint_hashes[name]
        report["models"][name] = row
        row["metrics"] = self_metrics(report, name)
        print(json.dumps(row["metrics"], indent=2), flush=True)
    # Publish only complete runs; leave a previous valid report intact on failure.
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(args.output)
    print(f"Saved {args.output}. Run compare_models.py and the MLflow score importer to attach these measurements.", flush=True)


if __name__ == "__main__":
    main()
