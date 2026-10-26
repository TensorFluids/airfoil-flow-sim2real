"""Controlled stacked-ensemble TKE-loss sweep; see experiments/TKE_ABLATION.md.

Inspect without training or writing MLflow runs:
    python experiments/finetune/train_tke_ablation.py --plan
Prepare shared caches and measure loss scales without optimization:
    python experiments/finetune/train_tke_ablation.py --prepare-only
Train/resume four arms and evaluate validation trajectories (NOT test):
    python experiments/finetune/train_tke_ablation.py
Evaluate one explicitly selected arm on test, after reviewing validation:
    python experiments/finetune/train_tke_ablation.py --evaluate-selected lambda_0p05 --selection-reason "Validation trade-off"
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import random
import re
import sys
import time

# Required for deterministic CUDA matrix multiplications, before importing torch.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
ROOT = Path(__file__).resolve().parents[2]
for directory in (ROOT, ROOT / "_vendor", ROOT / "experiments"):
    sys.path.insert(0, str(directory))

import mlflow
from mlflow.tracking import MlflowClient
import numpy as np
import torch

from finetune.dataset import _MEAN_IN, _STD_IN, _MEAN_TGT, _STD_TGT
from finetune.tke_ablation import (
    CACHE_VERSION, build_datasets, calibrate, epoch_loader, evaluate, file_hash,
    json_hash, make_splits, prepare_cache, run_epoch, save_checkpoint,
    seed_everything, state_hash, tradeoff_metrics, write_json,
)
from mlflow_tracking import DEFAULT_TRACKING_DIR
from models.stacked_ensemble import SpatialAttentionMixer


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT.parent / "data" / "train_real")
    parser.add_argument("--sweep-id", default="stacked-tke-v2")
    parser.add_argument("--lambdas", type=float, nargs=3, default=[0.0, 0.05, 0.10])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--cache-batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--test-fraction", type=float, default=0.15)
    parser.add_argument("--calibration-windows", type=int, default=64)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--skip-legacy", action="store_true", help="Omit the original variance-loss reference arm.")
    parser.add_argument("--checkpoint-selection", choices=["best_validation_field", "final_epoch"], default="best_validation_field")
    parser.add_argument("--selection-reason", help="Required with --evaluate-selected; records the validation-only rationale.")
    parser.add_argument("--max-l2-increase-pct", type=float)
    parser.add_argument("--max-mvpe-increase-pct", type=float)
    parser.add_argument("--max-sps-drop-points", type=float)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--plan", action="store_true")
    mode.add_argument("--prepare-only", action="store_true")
    mode.add_argument("--evaluate-selected", metavar="ARM", help="Lock and test exactly one arm, e.g. lambda_0p05 or legacy_variance.")
    args = parser.parse_args(argv)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", args.sweep_id):
        parser.error("sweep-id must contain only letters, digits, underscores, and hyphens.")
    if any(not math.isfinite(x) or x < 0 for x in args.lambdas) or len(set(args.lambdas)) != 3 or 0.0 not in args.lambdas:
        parser.error("Use three distinct nonnegative lambdas, including the zero control.")
    args.lambdas.sort()
    if len({arm_key(w) for w in args.lambdas}) != len(args.lambdas):
        parser.error("Lambda values must produce distinct arm identifiers (8 significant digits).")
    if min(args.epochs, args.batch_size, args.cache_batch_size, args.calibration_windows) < 1:
        parser.error("Epoch and batch/window counts must be positive.")
    if not math.isfinite(args.lr) or args.lr <= 0 or not 0 <= args.seed < 2**32:
        parser.error("Use a positive finite learning rate and a seed in [0, 2**32).")
    limits = [args.max_l2_increase_pct, args.max_mvpe_increase_pct, args.max_sps_drop_points]
    if any(x is not None for x in limits) and not all(x is not None and math.isfinite(x) and x >= 0 for x in limits):
        parser.error("Supply all three nonnegative trade-off limits, or omit all three.")
    if args.evaluate_selected:
        if args.evaluate_selected not in [arm_key(w) for w in study_arms(args)]:
            parser.error("Selected arm is not part of this study.")
        if not args.selection_reason or not args.selection_reason.strip():
            parser.error("--evaluate-selected requires a nonempty --selection-reason based on validation results.")
    elif args.selection_reason:
        parser.error("--selection-reason is only used with --evaluate-selected.")
    return args


def arm_key(weight: float | str) -> str:
    if weight == "legacy_variance":
        return weight
    return "lambda_" + format(weight, ".8g").replace(".", "p")


def study_arms(args):
    return list(args.lambdas) + ([] if args.skip_legacy else ["legacy_variance"])


def selected_checkpoint(args, study_dir, key):
    return study_dir / key / ("best_validation_field.pth" if args.checkpoint_selection == "best_validation_field" else "last.pth")


def discover(args):
    data_dir = args.data_dir.resolve()
    names = sorted(
        path.name for path in data_dir.glob("*.h5")
        if path.name != "7575_0.h5" and not path.name.startswith("exclude_")
    )
    splits = make_splits(names, args.seed, args.val_fraction, args.test_fraction)
    datasets = build_datasets(data_dir, splits)
    checkpoints = {
        "fno": ROOT / "checkpoints" / "sim_real_ft" / "sim_real_fno_fp16.pth",
        "cno": ROOT / "checkpoints" / "sim_real_ft" / "sim_real_cno.pth",
        "transolver": ROOT / "checkpoints" / "sim_real_ft" / "sim_real_transolver.pth",
    }
    for path in checkpoints.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    return data_dir, splits, datasets, checkpoints


def prepare_study(args, data_dir, splits, datasets, checkpoints):
    print("Fingerprinting source data and checkpoints ...", flush=True)
    data_hashes = {name: file_hash(data_dir / name) for names in splits.values() for name in names}
    # Differently named copies must not cross split boundaries.
    sets = {split: {data_hashes[name] for name in names} for split, names in splits.items()}
    if sets["train"] & sets["val"] or sets["train"] & sets["test"] or sets["val"] & sets["test"]:
        raise ValueError("Duplicate trajectory contents cross split boundaries.")
    identity = {
        "cache_version": CACHE_VERSION,
        "splits": splits,
        "data_sha256": data_hashes,
        "checkpoint_sha256": {name: file_hash(path) for name, path in checkpoints.items()},
        "sample_ids": {split: [[Path(path).name, start] for path, start in ds.samples] for split, ds in datasets.items()},
        "normalization": {"mean_in": _MEAN_IN.tolist(), "std_in": _STD_IN.tolist(), "mean_target": _MEAN_TGT.tolist(), "std_target": _STD_TGT.tolist()},
        "baseline_code_sha256": {str(path.relative_to(ROOT)): file_hash(path) for path in sorted((ROOT / "rpde_baselines").rglob("*.py"))},
        "loader_sha256": file_hash(ROOT / "load_baseline.py"),
        "dataset_sha256": file_hash(ROOT / "experiments" / "finetune" / "dataset.py"),
        "cache_builder_sha256": file_hash(Path(__file__).with_name("tke_ablation.py")),
        "device": args.device,
        "gpu_name": torch.cuda.get_device_name(torch.device(args.device)) if args.device.startswith("cuda") else None,
        "cache_batch_size": args.cache_batch_size,
        "torch_version": str(torch.__version__),
        "numpy_version": np.__version__,
        "cache_prediction_dtype": "float16",
        "cache_target_dtype": "float32_physical_with_original_zero_mask",
    }
    cache_id = json_hash(identity)
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items() if key not in ("plan", "prepare_only", "evaluate_selected", "selection_reason")}
    spec = {
        "config": config, "cache_id": cache_id, "mixer_hidden": 32,
        "optimizer": "Adam", "learning_rate_schedule": "constant", "grad_clip": 1.0,
        "checkpoint_selection": args.checkpoint_selection, "early_stopping": False,
        "loss_mask_policy": "full_grid_no_inferred_validity_mask",
        "test_policy": "explicit_single_arm_selection_after_validation",
        "held_out_scope": "held_out_from_mixer_training_not_guaranteed_unseen_by_supplied_baselines",
        "code_sha256": {path.name: file_hash(path) for path in (
            Path(__file__), Path(__file__).with_name("tke_ablation.py"), ROOT / "experiments" / "models" / "stacked_ensemble.py", ROOT / "scoring.py",
        )},
    }
    study_dir = ROOT / "checkpoints" / "tke_ablation" / args.sweep_id
    manifest_path = study_dir / "manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text(encoding="utf-8"))["spec"] != spec:
        raise ValueError("This sweep-id already has different data/code/settings. Use a new --sweep-id.")
    write_json(manifest_path, {"spec": spec, "cache_identity": identity})
    seed_everything(args.seed)
    initial_model = SpatialAttentionMixer(hidden=32)
    initial_path = study_dir / "initial_state.pth"
    if initial_path.exists():
        initial_model.load_state_dict(torch.load(initial_path, map_location="cpu", weights_only=True))
    else:
        save_checkpoint(initial_path, initial_model.state_dict())
    initial_state = {key: value.clone() for key, value in initial_model.state_dict().items()}
    fingerprint = json_hash({"spec": spec, "initial_state_sha256": state_hash(initial_state)})
    cache = prepare_cache(ROOT / "experiments" / "cache" / "tke_ablation" / cache_id, datasets, checkpoints, identity, args.device, args.cache_batch_size)
    calibration_path = study_dir / "calibration.json"
    if calibration_path.exists():
        calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
        if calibration["initial_state_sha256"] != state_hash(initial_state):
            raise ValueError("Calibration and initial model state do not match.")
    else:
        calibration = calibrate(initial_model.to(args.device), cache["train"], args.device, args.seed, args.calibration_windows, args.batch_size)
        calibration["initial_state_sha256"] = state_hash(initial_state)
        write_json(calibration_path, calibration)
    print(json.dumps(calibration, indent=2), flush=True)
    print("Initial weighted TKE/field contributions:", {str(weight): weight for weight in args.lambdas}, flush=True)
    diagnostic = calibration["gradient_diagnostics"]
    print("Initial weighted TKE/field gradient ratios:", {str(w): w * diagnostic["scaled_tke_to_field_gradient_ratio"] for w in args.lambdas}, flush=True)
    print("Legacy beta=1 variance/field gradient ratio:", diagnostic["legacy_to_field_gradient_ratio"], flush=True)
    del initial_model
    return study_dir, cache, initial_state, calibration, fingerprint


def get_run(client, experiment_id, args, weight, params, checkpoint):
    key = arm_key(weight)
    matches = client.search_runs([experiment_id], filter_string=f"tags.sweep_id = '{args.sweep_id}' AND tags.ablation_arm = '{key}'")
    if len(matches) > 1:
        raise RuntimeError(f"Ambiguous MLflow runs for {args.sweep_id}/{key}.")
    if checkpoint and (not matches or matches[0].info.run_id != checkpoint["run_id"]):
        raise RuntimeError("The saved training run is missing/deleted in MLflow; restore it before resuming.")
    if matches:
        run = matches[0]
        if run.data.params.get("sweep_fingerprint") != params["sweep_fingerprint"]:
            raise RuntimeError("Existing MLflow run belongs to different settings or initialization.")
    else:
        run = client.create_run(experiment_id, tags={
            "mlflow.runName": f"{args.sweep_id}-{key}",
            "sweep_id": args.sweep_id, "ablation_arm": key,
            "kind": "trained_model", "model_family": "stacked_ensemble",
            "checklist_status": "pending",
        })
    return run.info.run_id


def set_checklist_phase(args, weight, run_id, phase, client):
    path = ROOT / "experiments" / "results" / "tke_ablation" / args.sweep_id / "checklist.json"
    if path.is_file():
        checklist = json.loads(path.read_text(encoding="utf-8"))
        for entry in checklist["experiments"]:
            if entry["lambda"] == weight:
                entry.update({"status": phase, "run_id": run_id})
        write_json(path, checklist)
        client.log_artifact(run_id, str(path), artifact_path="study")


def train_arm(args, weight, study_dir, cache, datasets, initial_state, calibration, fingerprint, client, experiment_id):
    key = arm_key(weight)
    legacy = weight == "legacy_variance"
    tke_weight = 0.0 if legacy else weight
    checkpoint_path = study_dir / key / "last.pth"
    result_dir = ROOT / "experiments" / "results" / "tke_ablation" / args.sweep_id / key
    scores_path = result_dir / "validation" / "scores.json"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False) if checkpoint_path.exists() else None
    params = {
        "model": "stacked_ensemble", "lambda_tke": tke_weight, "beta_legacy_variance": 1.0 if legacy else 0.0, "epochs": args.epochs,
        "batch_size": args.batch_size, "learning_rate": args.lr, "seed": args.seed,
        "mixer_hidden": 32, "optimizer": "Adam", "lr_schedule": "constant",
        "gradient_clip": 1.0, "weight_decay": 0.0,
        "early_stopping": False, "checkpoint_selection": args.checkpoint_selection,
        "loss_mask_policy": "full_grid_no_inferred_validity_mask",
        "train_windows": len(cache["train"]), "validation_windows": len(cache["val"]), "test_windows": len(cache["test"]),
        "tke_scale": calibration["tke_scale"], "effective_tke_coefficient": tke_weight * calibration["tke_scale"],
        "initial_field_mse": calibration["initial_field_mse"], "initial_physical_tke_mse": calibration["initial_physical_tke_mse"],
        "initial_state_sha256": state_hash(initial_state), "sweep_fingerprint": fingerprint,
        "loss_definition": "normalized_field_mse_plus_beta1_original_variance_mse" if legacy else "normalized_field_mse_plus_lambda_times_scaled_physical_tke_mse",
        "precision": "float32_training_float16_cached_base_predictions", "device": args.device,
    }
    if checkpoint and (checkpoint["sweep_fingerprint"] != fingerprint or checkpoint["ablation_arm"] != key):
        raise ValueError("Checkpoint does not match this experiment arm.")
    run_id = get_run(client, experiment_id, args, weight, params, checkpoint)
    if checkpoint and checkpoint["epoch"] == args.epochs and scores_path.is_file() and client.get_run(run_id).data.tags.get("checklist_status") == "complete":
        print(f"Already complete: {key} ({run_id})", flush=True)
        return run_id, json.loads(scores_path.read_text(encoding="utf-8"))

    seed_everything(args.seed)
    model = SpatialAttentionMixer(hidden=32).to(args.device)
    model.load_state_dict(initial_state)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    start_epoch, history = 1, []
    if checkpoint:
        model.load_state_dict(checkpoint["mixer_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch, history = checkpoint["epoch"] + 1, checkpoint["history"]
        torch.set_rng_state(checkpoint["torch_rng_state"])
        if args.device.startswith("cuda"):
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng_states"])
        random.setstate(checkpoint["python_rng_state"])
        np.random.set_state(checkpoint["numpy_rng_state"])
    best_field = min((row["metrics"]["validation_field_mse"] for row in history), default=float("inf"))

    try:
        with mlflow.start_run(run_id=run_id):
            mlflow.log_params(params)
            mlflow.set_tags({"checklist_status": "training", "evaluation_scope": "validation_only_until_explicit_test_selection"})
            set_checklist_phase(args, weight, run_id, "training", client)
            mlflow.log_artifact(str(study_dir / "manifest.json"), artifact_path="study")
            mlflow.log_artifact(str(study_dir / "calibration.json"), artifact_path="study")
            diagnostic = calibration["gradient_diagnostics"]
            mlflow.log_metrics({"initial_" + k: float(v) for k, v in diagnostic.items() if isinstance(v, (int, float))}, step=0)
            ratio = diagnostic["legacy_to_field_gradient_ratio"] if legacy else tke_weight * diagnostic["scaled_tke_to_field_gradient_ratio"]
            mlflow.log_metric("initial_weighted_penalty_to_field_gradient_ratio", ratio, step=0)
            if history:
                mlflow.log_metrics(history[-1]["metrics"], step=history[-1]["epoch"])
            for epoch in range(start_epoch, args.epochs + 1):
                started = time.perf_counter()
                train = run_epoch(model, epoch_loader(cache["train"], args.batch_size, args.seed, epoch, True), optimizer, args.device, tke_weight, calibration["tke_scale"], 1.0, legacy=legacy)
                val = run_epoch(model, epoch_loader(cache["val"], args.batch_size, args.seed, epoch, False), None, args.device, tke_weight, calibration["tke_scale"], 1.0, legacy=legacy)
                metrics = {**{f"train_{k}": v for k, v in train.items()}, **{f"validation_{k}": v for k, v in val.items()}, "learning_rate": args.lr}
                metrics["epoch_seconds"] = time.perf_counter() - started
                history.append({"epoch": epoch, "metrics": metrics})
                payload = {
                    "mixer_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
                    "mixer_hidden": 32, "epoch": epoch, "lambda_tke": tke_weight, "ablation_arm": key, "history": history,
                    "sweep_fingerprint": fingerprint, "run_id": run_id,
                    "torch_rng_state": torch.get_rng_state(),
                    "cuda_rng_states": torch.cuda.get_rng_state_all() if args.device.startswith("cuda") else [],
                    "python_rng_state": random.getstate(), "numpy_rng_state": np.random.get_state(),
                }
                if val["field_mse"] < best_field:
                    best_field = val["field_mse"]
                    save_checkpoint(study_dir / key / "best_validation_field.pth", payload)
                save_checkpoint(checkpoint_path, payload)
                mlflow.log_metrics(metrics, step=epoch)
                seconds = metrics["epoch_seconds"]
                remaining_minutes = (args.epochs - epoch) * seconds / 60
                print(f"{key} epoch={epoch}/{args.epochs} train={train['loss']:.6f} val={val['loss']:.6f} field={val['field_mse']:.6f} scaled_tke={val['tke_mse_scaled']:.6f} {seconds:.1f}s/epoch arm_ETA~{remaining_minutes:.0f}min", flush=True)
            mlflow.set_tag("checklist_status", "evaluating")
            set_checklist_phase(args, weight, run_id, "evaluating", client)
            # Preserve the fixed-final-epoch ablation, irrespective of the primary
            # checkpoint rule. No test scoring happens in the training workflow.
            final_result = evaluate(model, cache["val"], datasets["val"], args.device, args.batch_size, result_dir / "validation_final")
            log_evaluation(final_result, "validation_final", args.epochs, result_dir / "validation_final")
            chosen = torch.load(selected_checkpoint(args, study_dir, key), map_location="cpu", weights_only=False)
            model.load_state_dict(chosen["mixer_state_dict"])
            result = evaluate(model, cache["val"], datasets["val"], args.device, args.batch_size, result_dir / "validation")
            result["checkpoint_epoch"] = chosen["epoch"]
            result["checkpoint_selection"] = args.checkpoint_selection
            result["checkpoint_sha256"] = file_hash(selected_checkpoint(args, study_dir, key))
            write_json(scores_path, result)
            log_evaluation(result, "validation", chosen["epoch"], result_dir / "validation")
            mlflow.log_metric("selected_checkpoint_epoch", chosen["epoch"])
            mlflow.log_artifact(str(checkpoint_path), artifact_path="checkpoints")
            mlflow.log_artifact(str(study_dir / key / "best_validation_field.pth"), artifact_path="checkpoints")
            mlflow.set_tag("checklist_status", "complete")
    except BaseException:
        client.set_tag(run_id, "checklist_status", "interrupted")
        set_checklist_phase(args, weight, run_id, "interrupted", client)
        raise
    finally:
        del model, optimizer
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()
    return run_id, result


def log_evaluation(result, split, epoch, output):
    mlflow.log_metrics({f"{split}_{key}": value for key, value in result["aggregate"].items()}, step=epoch)
    for trajectory, row in result["trajectories"].items():
        mlflow.log_metrics({f"{split}/{Path(trajectory).stem}/{key}": value for key, value in row["metrics"].items()}, step=epoch)
    mlflow.log_artifact(str(output / "scores.json"), artifact_path=f"evaluation/{split}")


def update_checklist(args, completed, client):
    output = ROOT / "experiments" / "results" / "tke_ablation" / args.sweep_id
    limits = {key: getattr(args, key) for key in ("max_l2_increase_pct", "max_mvpe_increase_pct", "max_sps_drop_points")}
    checklist = {"sweep_id": args.sweep_id, "limits": limits, "evaluation_split": "validation", "checkpoint_selection": args.checkpoint_selection, "experiments": []}
    control = completed.get(0.0)
    for weight in study_arms(args):
        entry = {"arm": arm_key(weight), "lambda": weight, "status": "complete" if weight in completed else "pending"}
        if weight in completed:
            run_id, result = completed[weight]
            entry.update({"run_id": run_id, "metrics": result["aggregate"]})
            if control:
                changes = tradeoff_metrics(result["aggregate"], control[1]["aggregate"], limits)
                entry["changes_vs_control"] = changes
                for key, value in changes.items():
                    if value is not None:
                        client.log_metric(run_id, "validation_vs_control_" + key, float(value), step=args.epochs)
        checklist["experiments"].append(entry)
    write_json(output / "checklist.json", checklist)
    lines = [f"# {args.sweep_id}", "", f"Validation only; primary checkpoint rule: {args.checkpoint_selection}.", "Same trajectory split, initialization, batches and full epoch budget for all arms.", ""]
    for row in checklist["experiments"]:
        lines.append(f"- [{'x' if row['status'] == 'complete' else ' '}] {row['arm']}: {row['status']}")
        if "metrics" in row:
            lines.append("  " + ", ".join(f"{key}={row['metrics'][key]:.5f}" for key in ("rel_l2_score", "tke_score", "mvpe_score", "sps_score")))
            lines.append("  Changes vs control: " + json.dumps(row.get("changes_vs_control", {})))
    lines += ["", "Positive error changes mean deterioration; positive SPS changes mean improvement. "
              "Acceptability is assessed only when all three trade-off limits are supplied. "
              "Select one arm from validation results; test evaluation requires a separate explicit command. "
              "The supplied frozen baselines may have seen these trajectories during their original training."]
    (output / "comparison.md").write_text("\n".join(lines), encoding="utf-8")
    for run_id, _ in completed.values():
        client.log_artifact(run_id, str(output / "checklist.json"), artifact_path="study")
        client.log_artifact(run_id, str(output / "comparison.md"), artifact_path="study")


def lock_test_selection(path, selection):
    """Once test is exposed, do not permit another arm/checkpoint in this study."""
    if path.exists():
        previous = json.loads(path.read_text(encoding="utf-8"))
        immutable = ("arm", "checkpoint_sha256", "checkpoint_epoch", "sweep_fingerprint", "run_id")
        if any(previous[key] != selection[key] for key in immutable):
            raise ValueError("Test selection is already locked to a different model. Do not use test results for further tuning.")
        return previous
    write_json(path, selection)
    return selection


def evaluate_selected(args, study_dir, cache, datasets, fingerprint, client):
    output = ROOT / "experiments" / "results" / "tke_ablation" / args.sweep_id
    # Require every requested arm's validation comparison before exposing test.
    for weight in study_arms(args):
        if not (output / arm_key(weight) / "validation" / "scores.json").is_file():
            raise ValueError("Finish validation evaluation for every requested arm before selecting a test model.")
    key = args.evaluate_selected
    path = selected_checkpoint(args, study_dir, key)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    last = torch.load(study_dir / key / "last.pth", map_location="cpu", weights_only=False)
    if checkpoint["sweep_fingerprint"] != fingerprint or checkpoint["ablation_arm"] != key or last["epoch"] != args.epochs:
        raise ValueError("Selected checkpoint does not match the completed study.")
    validation = json.loads((output / key / "validation" / "scores.json").read_text(encoding="utf-8"))
    if validation["checkpoint_sha256"] != file_hash(path):
        raise ValueError("Checkpoint has changed since validation scoring.")
    run_id = checkpoint["run_id"]
    run = client.get_run(run_id)
    if run.data.params.get("sweep_fingerprint") != fingerprint or run.info.lifecycle_stage != "active":
        raise ValueError("Restore the original MLflow model run before testing.")
    selection = lock_test_selection(study_dir / "test_selection.json", {
        "arm": key, "checkpoint_sha256": file_hash(path), "checkpoint_epoch": checkpoint["epoch"],
        "sweep_fingerprint": fingerprint, "run_id": run_id,
        "selection_reason": args.selection_reason.strip(), "validation_metrics": validation["aggregate"],
        "checkpoint_selection": args.checkpoint_selection,
    })
    result_dir = output / key / "test"
    scores_path = result_dir / "scores.json"
    with mlflow.start_run(run_id=run_id):
        mlflow.set_tags({"selected_for_test": "true", "test_status": "evaluating"})
        mlflow.log_artifact(str(study_dir / "test_selection.json"), artifact_path="study")
        if scores_path.is_file():
            result = json.loads(scores_path.read_text(encoding="utf-8"))
        else:
            model = SpatialAttentionMixer(hidden=32).to(args.device)
            model.load_state_dict(checkpoint["mixer_state_dict"])
            result = evaluate(model, cache["test"], datasets["test"], args.device, args.batch_size, result_dir)
        log_evaluation(result, "test", selection["checkpoint_epoch"], result_dir)
        mlflow.set_tag("test_status", "complete")
    print(f"Selected-model test complete: {key}. Scores attached to existing run {run_id}.", flush=True)


def main(argv=None):
    args = parse_args(argv)
    data_dir, splits, datasets, checkpoints = discover(args)
    summary = {
        "sweep_id": args.sweep_id, "arms": [arm_key(w) for w in study_arms(args)], "epochs_each": args.epochs,
        "learning_rate": args.lr, "batch_size": args.batch_size,
        "splits": {split: {"files": names, "trajectories": len(names), "windows": len(datasets[split])} for split, names in splits.items()},
        "checkpoint_selection": args.checkpoint_selection, "early_stopping": False,
        "loss_mask_policy": "full_grid_no_inferred_validity_mask",
        "test_evaluation": "explicit_selected_arm_only",
        "calibration": "training_windows_only_before_any_optimizer_steps",
        "estimated_cache_gib": sum(len(ds) for ds in datasets.values()) * 20 * 32 * 64 * (9 * 2 + 3 * 4) / 1024**3,
    }
    print(json.dumps(summary, indent=2), flush=True)
    if args.plan:
        return
    if args.evaluate_selected and not (ROOT / "checkpoints" / "tke_ablation" / args.sweep_id / "manifest.json").is_file():
        raise ValueError("No prepared study found; train and validate the arms first.")
    seed_everything(args.seed)
    study_dir, cache, initial_state, calibration, fingerprint = prepare_study(args, data_dir, splits, datasets, checkpoints)
    if args.prepare_only:
        print("Shared cache and loss calibration ready. No optimization or MLflow runs started.", flush=True)
        return
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_DIR.resolve().as_uri()))
    experiment = mlflow.set_experiment(os.environ.get("MLFLOW_EXPERIMENT_NAME", "airfoil-flow-forecasting"))
    client = MlflowClient()
    if args.evaluate_selected:
        evaluate_selected(args, study_dir, cache, datasets, fingerprint, client)
        return
    completed = {}
    update_checklist(args, completed, client)
    for weight in study_arms(args):
        completed[weight] = train_arm(args, weight, study_dir, cache, datasets, initial_state, calibration, fingerprint, client, experiment.experiment_id)
        update_checklist(args, completed, client)
    print(f"Training/validation complete; test NOT evaluated. Review: {ROOT / 'experiments' / 'results' / 'tke_ablation' / args.sweep_id / 'comparison.md'}", flush=True)
    print("Select one arm using validation trade-offs, then run --evaluate-selected ARM --selection-reason REASON with the same study settings.", flush=True)


if __name__ == "__main__":
    main()
