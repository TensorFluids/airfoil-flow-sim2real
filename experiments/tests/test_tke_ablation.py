"""Numerical/protocol checks only: no training or real MLflow writes."""
from pathlib import Path
import sys
import tempfile
import json
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT, ROOT / "experiments"):
    sys.path.insert(0, str(path))

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset

from finetune.dataset import RealPDEDataset, _MEAN_TGT, _STD_TGT
from finetune.tke_ablation import (
    aggregate_trajectories, calibrate, epoch_loader, loss_terms, make_splits,
    physical_tke, save_checkpoint, score_fields, state_hash, tradeoff_metrics,
    legacy_variance_loss, run_epoch, file_hash,
)
from finetune.train_tke_ablation import (
    get_run, parse_args, set_checklist_phase, study_arms, arm_key,
    lock_test_selection, train_arm, evaluate_selected, selected_checkpoint,
)
from models.stacked_ensemble import SpatialAttentionMixer
from scoring import kinetic_energy


class TKEAblationTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        torch.set_num_threads(1)

    def test_physical_tke_matches_scorer_population_variance(self):
        values = np.random.default_rng(3).normal(size=(2, 5, 4, 4, 3))
        values[..., 2] *= 1000  # Pressure must have no influence.
        expected = kinetic_energy(values)
        actual = physical_tke(torch.from_numpy(values)).numpy()
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
        shifted = values.copy()
        shifted[..., :2] += 3.0
        np.testing.assert_allclose(physical_tke(torch.from_numpy(shifted)).numpy(), expected, atol=1e-12)

    def test_control_is_original_field_mse_and_tke_has_gradients(self):
        target = torch.randn(2, 5, 4, 4, 3, dtype=torch.float64) * 0.02
        target[..., 2] = 0
        pred = torch.randn_like(target, requires_grad=True)
        field, tke = loss_terms(pred, target)
        normalized_target = (target - target.new_tensor(_MEAN_TGT)) / target.new_tensor(_STD_TGT)
        torch.testing.assert_close(field + 0.0 * tke, F.mse_loss(pred, normalized_target))
        gradient = torch.autograd.grad(tke, pred)[0]
        self.assertTrue(torch.isfinite(gradient).all())
        self.assertGreater(gradient[..., :2].abs().sum().item(), 0)
        self.assertEqual(gradient[..., 2].abs().sum().item(), 0)

    def test_split_is_reproducible_disjoint_and_complete(self):
        names = [f"trajectory_{i}.h5" for i in range(81)]
        first = make_splits(names, 42, 0.15, 0.15)
        self.assertEqual(first, make_splits(list(reversed(names)), 42, 0.15, 0.15))
        self.assertEqual([len(first[key]) for key in ("train", "val", "test")], [57, 12, 12])
        self.assertEqual(set(sum(first.values(), [])), set(names))
        self.assertFalse(set(first["train"]) & set(first["test"]))
        self.assertFalse(set(first["val"]) & set(first["test"]))
        self.assertFalse(set(first["train"]) & set(first["val"]))

    def test_explicit_dataset_preserves_raw_zero_masks_and_default_behavior(self):
        with tempfile.TemporaryDirectory() as directory:
            values = np.full((40, 8, 8), 0.125, dtype=np.float32)
            values[:, :4, :4] = 0.0
            with h5py.File(Path(directory) / "case.h5", "w") as handle:
                handle.create_dataset("u", data=values)
                handle.create_dataset("v", data=values * 0.1)
            raw = RealPDEDataset(directory, file_names=["case.h5"], normalize_targets=False)[0][1]
            norm = RealPDEDataset(directory, file_names=["case.h5"])[0][1]
            self.assertEqual(torch.count_nonzero(raw[:, :2, :2]).item(), 0)
            torch.testing.assert_close(norm, (raw - raw.new_tensor(_MEAN_TGT)) / raw.new_tensor(_STD_TGT))
            with self.assertRaises(ValueError):
                RealPDEDataset(directory, file_names=["missing.h5"])

    def test_batch_order_is_independent_of_global_rng_and_previous_epochs(self):
        dataset = TensorDataset(torch.arange(37))
        def order(epoch):
            return torch.cat([row[0] for row in epoch_loader(dataset, 8, 42, epoch, True)])
        first = order(9)
        torch.rand(100)
        order(3)
        self.assertTrue(torch.equal(first, order(9)))
        self.assertFalse(torch.equal(first, order(10)))
        self.assertEqual(sorted(first.tolist()), list(range(37)))

    def test_shared_initialization_and_calibration_do_not_fit_weights(self):
        model = SpatialAttentionMixer(hidden=2)
        initial = {key: value.clone() for key, value in model.state_dict().items()}
        targets = torch.rand(8, 5, 4, 4, 3) * 0.1
        targets[..., 2] = 0
        stacked = torch.randn(8, 5, 4, 4, 9)
        cache = TensorDataset(targets, stacked)
        calibration = calibrate(model, cache, "cpu", 42, 5, 2)
        again = calibrate(model, cache, "cpu", 42, 5, 2)
        self.assertEqual(calibration, again)
        self.assertEqual(state_hash(initial), state_hash(model.state_dict()))
        self.assertTrue(all(p.grad is None for p in model.parameters()))
        diagnostic = calibration["gradient_diagnostics"]
        self.assertEqual(diagnostic["window_indices"], calibration["window_indices"][:2])
        self.assertGreater(diagnostic["field_gradient_norm"], 0)
        self.assertGreater(diagnostic["physical_tke_gradient_norm"], 0)
        self.assertAlmostEqual(diagnostic["scaled_tke_to_field_gradient_ratio"],
                               calibration["tke_scale"] * diagnostic["physical_tke_gradient_norm"] / diagnostic["field_gradient_norm"])
        self.assertAlmostEqual(calibration["tke_scale"] * calibration["initial_physical_tke_mse"], calibration["initial_field_mse"])
        copy = SpatialAttentionMixer(hidden=2)
        copy.load_state_dict(initial)
        with torch.no_grad():
            torch.testing.assert_close(model(stacked), copy(stacked), rtol=0, atol=0)

    def test_trajectory_aggregation_matches_joint_scoring_with_unequal_masks(self):
        rng = np.random.default_rng(4)
        target = rng.uniform(0.01, 0.2, size=(5, 5, 32, 64, 3)).astype(np.float32)
        target[..., 2] = 0
        target[:2, :, :20, :, :2] = 0
        prediction = target * 0.97 + rng.normal(0, 0.002, target.shape).astype(np.float32)
        prediction[..., 2] = 0
        rows = {}
        for name, selection in (("short", slice(0, 2)), ("long", slice(2, 5))):
            p, t = prediction[selection], target[selection]
            rows[name] = {"windows": len(t), "scored_elements": int(np.count_nonzero(t[..., :2])), "metrics": score_fields(p, t)}
        combined = aggregate_trajectories(rows)
        direct = score_fields(prediction, target)
        for key in combined:
            self.assertAlmostEqual(combined[key], direct[key], places=5)

    def test_tradeoff_rejects_tke_gain_with_excess_field_deterioration(self):
        control = {"rel_l2_error": 0.1, "tke_rel_l2_error": 0.5, "mvpe_rel_l2_error": 0.1, "sps_score": 25.0}
        candidate = dict(control, rel_l2_error=0.12, tke_rel_l2_error=0.4)
        limits = {"max_l2_increase_pct": 2.0, "max_mvpe_increase_pct": 2.0, "max_sps_drop_points": 1.0}
        self.assertFalse(tradeoff_metrics(candidate, control, limits)["acceptable_tke_improvement"])
        candidate["rel_l2_error"] = 0.101
        self.assertTrue(tradeoff_metrics(candidate, control, limits)["acceptable_tke_improvement"])
        self.assertNotIn("acceptable_tke_improvement", tradeoff_metrics(candidate, control, dict.fromkeys(limits)))

    def test_existing_mlflow_arm_is_reused_and_fingerprint_checked(self):
        run = SimpleNamespace(info=SimpleNamespace(run_id="existing"), data=SimpleNamespace(params={"sweep_fingerprint": "abc"}))
        client = Mock()
        client.search_runs.return_value = [run]
        args = SimpleNamespace(sweep_id="study")
        self.assertEqual(get_run(client, "experiment", args, 0.05, {"sweep_fingerprint": "abc"}, {"run_id": "existing"}), "existing")
        client.create_run.assert_not_called()
        with self.assertRaises(RuntimeError):
            get_run(client, "experiment", args, 0.05, {"sweep_fingerprint": "different"}, None)

    def test_atomic_checkpoint_preserves_initial_state_and_rng(self):
        model = SpatialAttentionMixer(hidden=2)
        payload = {"mixer_state_dict": model.state_dict(), "epoch": 4, "torch_rng_state": torch.get_rng_state()}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "last.pth"
            save_checkpoint(path, payload)
            restored = torch.load(path, map_location="cpu", weights_only=True)
            self.assertEqual(state_hash(restored["mixer_state_dict"]), state_hash(model.state_dict()))
            self.assertEqual(restored["epoch"], 4)
            self.assertTrue(torch.equal(restored["torch_rng_state"], payload["torch_rng_state"]))
            self.assertFalse(path.with_suffix(".pth.tmp").exists())

    def test_checklist_updates_only_the_active_arm(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "experiments/results/tke_ablation/study/checklist.json"
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps({"experiments": [
                {"lambda": 0.0, "status": "complete"},
                {"lambda": 0.05, "status": "pending"},
                {"lambda": 0.10, "status": "pending"},
            ]}), encoding="utf-8")
            client = Mock()
            with patch("finetune.train_tke_ablation.ROOT", root):
                set_checklist_phase(SimpleNamespace(sweep_id="study"), 0.05, "existing", "training", client)
            rows = json.loads(path.read_text(encoding="utf-8"))["experiments"]
            self.assertEqual([row["status"] for row in rows], ["complete", "training", "pending"])
            self.assertEqual(rows[1]["run_id"], "existing")
            client.log_artifact.assert_called_once()

    def test_cli_defaults_use_identical_full_budget(self):
        args = parse_args(["--plan"])
        self.assertEqual(args.lambdas, [0.0, 0.05, 0.10])
        self.assertEqual(args.epochs, 100)
        self.assertEqual(args.batch_size, 16)
        self.assertEqual(study_arms(args), [0.0, 0.05, 0.10, "legacy_variance"])
        self.assertEqual(args.checkpoint_selection, "best_validation_field")
        self.assertEqual(arm_key("legacy_variance"), "legacy_variance")
        self.assertEqual(len(study_arms(parse_args(["--skip-legacy"]))), 3)

    def test_legacy_matches_original_normalized_sample_variance_including_pressure(self):
        target = torch.rand(2, 5, 4, 4, 3)
        pred = torch.randn_like(target, requires_grad=True)
        normalized = (target - target.new_tensor(_MEAN_TGT)) / target.new_tensor(_STD_TGT)
        expected = F.mse_loss(pred.var(dim=1), normalized.var(dim=1))
        torch.testing.assert_close(legacy_variance_loss(pred, target), expected)
        gradient = torch.autograd.grad(legacy_variance_loss(pred, target), pred)[0]
        self.assertGreater(gradient[..., 2].abs().sum().item(), 0)

    def test_legacy_and_control_objectives_without_optimizer_steps(self):
        model = SpatialAttentionMixer(hidden=2)
        cache = TensorDataset(torch.rand(2, 5, 4, 4, 3), torch.randn(2, 5, 4, 4, 9))
        initial = state_hash(model.state_dict())
        loader = epoch_loader(cache, 2, 42, 1, False)
        control = run_epoch(model, loader, None, "cpu", 0, 100, 1)
        legacy = run_epoch(model, loader, None, "cpu", 0, 100, 1, legacy=True)
        self.assertAlmostEqual(control["loss"], control["field_mse"])
        np.testing.assert_allclose(legacy["loss"], legacy["field_mse"] + legacy["legacy_variance_mse"], rtol=1e-6)
        self.assertEqual(legacy["tke_penalty"], 0)
        self.assertEqual(initial, state_hash(model.state_dict()))

    def test_test_selection_lock_refuses_different_arm_or_checkpoint(self):
        selection = dict(arm="lambda_0p05", checkpoint_sha256="abc", checkpoint_epoch=12,
                         sweep_fingerprint="study", run_id="run", selection_reason="validation")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "selection.json"
            self.assertEqual(lock_test_selection(path, selection), selection)
            self.assertEqual(lock_test_selection(path, dict(selection, selection_reason="retry")), selection)
            for changed in (dict(arm="legacy_variance"), dict(checkpoint_sha256="def"), dict(checkpoint_epoch=13)):
                with self.assertRaises(ValueError):
                    lock_test_selection(path, dict(selection, **changed))

    def test_selected_test_requires_validation_for_all_arms(self):
        args = parse_args(["--evaluate-selected", "lambda_0p05", "--selection-reason", "validation only"])
        with tempfile.TemporaryDirectory() as directory, patch("finetune.train_tke_ablation.ROOT", Path(directory)):
            with patch("finetune.train_tke_ablation.evaluate") as evaluate_mock:
                with self.assertRaises(ValueError):
                    evaluate_selected(args, Path(directory), {}, {}, "fingerprint", Mock())
                evaluate_mock.assert_not_called()

    def test_selected_test_locks_before_scoring_reuses_run_and_saved_scores(self):
        import finetune.train_tke_ablation as runner
        args = parse_args(["--device", "cpu", "--evaluate-selected", "lambda_0p05", "--selection-reason", "validation trade-offs"])
        cache = {key: [key] for key in ("train", "val", "test")}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            study = root / "study"
            output = root / "experiments/results/tke_ablation" / args.sweep_id
            initial = SpatialAttentionMixer(hidden=32).state_dict()
            for arm in study_arms(args):
                key = arm_key(arm)
                payload = {"mixer_state_dict": initial, "epoch": 10, "sweep_fingerprint": "fp", "ablation_arm": key, "run_id": key + "-run"}
                path = selected_checkpoint(args, study, key)
                save_checkpoint(path, payload)
                save_checkpoint(study / key / "last.pth", dict(payload, epoch=100))
                scores = output / key / "validation/scores.json"
                scores.parent.mkdir(parents=True)
                scores.write_text(json.dumps({"checkpoint_sha256": file_hash(path), "aggregate": {"tke_score": 50}}), encoding="utf-8")
            result = {"aggregate": {"tke_score": 51}, "trajectories": {}}
            def fake_evaluate(model, cached, original, device, batch_size, destination):
                self.assertIs(cached, cache["test"])
                self.assertTrue((study / "test_selection.json").is_file())
                destination.mkdir(parents=True)
                (destination / "scores.json").write_text(json.dumps(result), encoding="utf-8")
                return result
            client = Mock()
            client.get_run.return_value = SimpleNamespace(data=SimpleNamespace(params={"sweep_fingerprint": "fp"}), info=SimpleNamespace(lifecycle_stage="active"))
            with patch.object(runner, "ROOT", root), patch.object(runner, "mlflow") as tracking, \
                 patch.object(runner, "evaluate", side_effect=fake_evaluate) as evaluation:
                evaluate_selected(args, study, cache, cache, "fp", client)
                evaluate_selected(args, study, cache, cache, "fp", client)
                self.assertEqual(evaluation.call_count, 1)
                self.assertEqual(tracking.start_run.call_count, 2)
                tracking.start_run.assert_called_with(run_id="lambda_0p05-run")
                client.create_run.assert_not_called()
                args.evaluate_selected = "legacy_variance"
                with self.assertRaises(ValueError):
                    evaluate_selected(args, study, cache, cache, "fp", client)
                self.assertEqual(evaluation.call_count, 1)

    def test_training_orchestration_never_scores_test_and_retains_best_checkpoint(self):
        # Mock every epoch/optimizer and all external tracking; this is NOT a
        # miniature training job. Check dispatch, checkpoint rules, and resume.
        import finetune.train_tke_ablation as runner
        args = parse_args(["--epochs", "2", "--device", "cpu"])
        initial = SpatialAttentionMixer(hidden=32).state_dict()
        calibration = {"tke_scale": 2.0, "initial_field_mse": 1.0, "initial_physical_tke_mse": 0.5,
                       "gradient_diagnostics": {"scaled_tke_to_field_gradient_ratio": 2.0, "legacy_to_field_gradient_ratio": 3.0}}
        cache = {key: [key] for key in ("train", "val", "test")}
        metric = {"loss": 1.0, "field_mse": 1.0, "tke_mse_scaled": 1.0}
        def fake_evaluate(model, cached, original, device, batch_size, output):
            self.assertIs(cached, cache["val"])
            output.mkdir(parents=True, exist_ok=True)
            result = {"aggregate": {"tke_score": 50.0}, "trajectories": {}}
            (output / "scores.json").write_text(json.dumps(result), encoding="utf-8")
            return result
        with tempfile.TemporaryDirectory() as directory:
            study = Path(directory) / "study"
            client = Mock()
            with patch.object(runner, "ROOT", Path(directory)), patch.object(runner, "mlflow"), \
                 patch.object(runner, "get_run", return_value="run"), \
                 patch.object(runner, "epoch_loader"), \
                 patch.object(runner.torch.optim, "Adam", return_value=Mock(state_dict=lambda: {})) as adam, \
                 patch.object(runner, "run_epoch", side_effect=[metric, metric, KeyboardInterrupt(), metric, dict(metric, field_mse=2.0)]) as epochs, \
                 patch.object(runner, "evaluate", side_effect=fake_evaluate) as evaluation:
                with self.assertRaises(KeyboardInterrupt):
                    train_arm(args, "legacy_variance", study, cache, cache, initial, calibration, "fp", client, "experiment")
                interrupted = torch.load(study / "legacy_variance/last.pth", weights_only=False)
                self.assertEqual(interrupted["epoch"], 1)
                evaluation.assert_not_called()
                _, result = train_arm(args, "legacy_variance", study, cache, cache, initial, calibration, "fp", client, "experiment")
                adam.return_value.load_state_dict.assert_called_once_with({})
                self.assertEqual(result["checkpoint_epoch"], 1)
                self.assertEqual(evaluation.call_count, 2)
                self.assertTrue(all(call.kwargs["legacy"] for call in epochs.call_args_list))
                self.assertTrue((study / "legacy_variance" / "last.pth").exists())
                self.assertTrue(selected_checkpoint(args, study, "legacy_variance").exists())
                adam.return_value.step.assert_not_called()
                client.get_run.return_value.data.tags = {"checklist_status": "complete"}
                train_arm(args, "legacy_variance", study, cache, cache, initial, calibration, "fp", client, "experiment")
                self.assertEqual(epochs.call_count, 5)  # Interrupted epoch repeated; completed arm skipped.


if __name__ == "__main__":
    unittest.main()
