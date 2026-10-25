"""Timing protocol tests with tiny CPU tensors and mocked MLflow; no model training."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[2]
for directory in (ROOT, ROOT / "experiments"):
    sys.path.insert(0, str(directory))

import numpy as np
import torch
from torch import nn

from benchmark.benchmark_timing import ForecastPipeline, measure_pipeline, parse_args
from benchmark.timing_metrics import (
    COMPETITION_METRIC_KEYS, TIMING_BOUNDARY, competition_metrics, file_hash,
    load_self_timing, optional_self_timing, self_metrics,
)
from scoring import ZERO_SCORES, _score_submission, score_time_competition


class TimingTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.legacy = {"rel_l2_score": 96.0, "tke_score": 81.0, "mvpe_score": 97.0,
                       "time_score": 63.88099828966087, "sps_score": 27.0}

    def test_competition_formula_is_preserved_and_old_files_remain_readable(self):
        self.assertEqual(score_time_competition(0.72896), 50)
        self.assertAlmostEqual(score_time_competition(0.72896 / 4), 100 / 1.5)
        for value in (0, -1, float("nan"), float("inf")):
            self.assertEqual(score_time_competition(value), 0)
        scores = competition_metrics(self.legacy)
        self.assertNotIn("time_score", scores)
        self.assertEqual(scores["timescore_competition"], self.legacy["time_score"])
        self.assertEqual(set(scores), set(COMPETITION_METRIC_KEYS))
        self.assertEqual(competition_metrics(scores), scores)
        with self.assertRaises(ValueError):
            competition_metrics(dict(self.legacy, timescore_competition=12))

    def test_real_scorer_emits_renamed_competition_key_and_preserves_quality(self):
        rng = np.random.default_rng(4)
        fields = rng.uniform(0.01, 0.2, size=(2, 4, 32, 64, 3)).astype(np.float32)
        fields[..., 2] = 0
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            np.savez(root / "predictions.npz", prediction=fields, mean_t_neural_s=0.72896)
            np.savez(root / "targets.npz", target=fields)
            scores = _score_submission(root, root)
        self.assertEqual(set(scores), set(COMPETITION_METRIC_KEYS))
        self.assertEqual(scores["timescore_competition"], 50)
        self.assertEqual(scores["rel_l2_score"], 100)
        self.assertEqual(scores["tke_score"], 100)
        self.assertEqual(scores["mvpe_score"], 100)
        self.assertNotIn("time_score", ZERO_SCORES)

    def test_self_score_is_unbounded_fno_relative_ratio_not_competition_ratio(self):
        report = {"models": {"fno_fp16": {"median_latency_s": 2.0}, "ensemble": {"median_latency_s": 4.0}}}
        self.assertEqual(self_metrics(report, "fno_fp16"), {"timescore_self": 1, "latency_ms_self": 2000})
        self.assertEqual(self_metrics(report, "ensemble"), {"timescore_self": 0.5, "latency_ms_self": 4000})
        report["models"]["ensemble"]["median_latency_s"] = 0.01
        self.assertEqual(self_metrics(report, "ensemble")["timescore_self"], 200)
        for value in (0, -1, float("nan"), float("inf")):
            report["models"]["ensemble"]["median_latency_s"] = value
            with self.assertRaises(ValueError):
                self_metrics(report, "ensemble")

    def make_report(self, root):
        samples = root / "samples.npz"
        np.savez(samples, input=np.ones((2, 4, 2, 2, 3), dtype=np.float32))
        (root / "weights.pth").write_bytes(b"fixture checkpoint")
        (root / "code.py").write_text("# fixture code\n", encoding="utf-8")
        report = {
            "schema_version": 1,
            "protocol": {"boundary": TIMING_BOUNDARY, "batch_size": 1, "warmup": 1, "repeats": 3,
                         "samples_sha256": file_hash(samples), "code_sha256": {"code.py": file_hash(root / "code.py")}},
            "models": {name: {"median_latency_s": multiplier * 2.0, "latencies_s": [multiplier * i for i in (1.0, 2.0, 3.0)],
                              "checkpoint_sha256": {"weights.pth": file_hash(root / "weights.pth")}}
                       for name, multiplier in (("fno_fp16", 1.0), ("ensemble", 2.0))},
        }
        path = root / "timing.json"
        path.write_text(json.dumps(report), encoding="utf-8")
        return path, samples, report

    def test_report_rejects_stale_data_weights_code_and_incompatible_protocols(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path, samples, report = self.make_report(root)
            self.assertEqual(load_self_timing(path, root, samples), report)
            for name, change in (
                ("no_fno", lambda r: r["models"].pop("fno_fp16")),
                ("wrong_boundary", lambda r: r["protocol"].update(boundary="mixer_only")),
                ("batch_size", lambda r: r["protocol"].update(batch_size=8)),
                ("no_warmup", lambda r: r["protocol"].update(warmup=0)),
                ("different_inputs", lambda r: r["protocol"].update(samples_sha256="changed")),
                ("different_weights", lambda r: r["models"]["ensemble"]["checkpoint_sha256"].update({"weights.pth": "changed"})),
                ("different_code", lambda r: r["protocol"]["code_sha256"].update({"code.py": "changed"})),
                ("incomplete", lambda r: r["models"]["ensemble"].update(latencies_s=[1.0])),
                ("wrong_median", lambda r: r["models"]["ensemble"].update(median_latency_s=99)),
            ):
                with self.subTest(name=name):
                    changed = json.loads(json.dumps(report))
                    change(changed)
                    path.write_text(json.dumps(changed), encoding="utf-8")
                    with self.assertRaises(ValueError):
                        load_self_timing(path, root, samples)

    def test_absent_self_report_omits_self_metrics_instead_of_reusing_old_times(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.assertEqual(optional_self_timing(None, root), (None, None))
            with self.assertRaises(FileNotFoundError):
                optional_self_timing(root / "missing.json", root)
            self.assertEqual(optional_self_timing(root / "missing.json", root, skip=True), (None, None))

    def test_complete_pipeline_invokes_all_bases_and_the_mixer(self):
        class Base(nn.Module):
            def __init__(self, offset):
                super().__init__()
                self.offset, self.calls = offset, 0

            def forward(self, x):
                self.calls += 1
                return x + self.offset

        class Mixer(nn.Module):
            def forward(self, x):
                self.shape = x.shape
                return x.reshape(*x.shape[:-1], 3, 3).mean(dim=-2)

        bases = [Base(i) for i in range(3)]
        mixer = Mixer()
        pipeline = ForecastPipeline(bases, mixer).eval()
        raw = torch.rand(1, 4, 2, 2, 3)
        with torch.inference_mode():
            result = pipeline(raw)
            expected = ((raw - pipeline.mean_in) / pipeline.std_in + 1) * pipeline.std_target + pipeline.mean_target
            expected[..., 2] = 0
        torch.testing.assert_close(result, expected)
        self.assertEqual([base.calls for base in bases], [1, 1, 1])
        self.assertEqual(mixer.shape[-1], 9)
        self.assertEqual(torch.count_nonzero(result[..., 2]), 0)

    def test_warmup_is_not_timed_and_each_repeat_receives_a_fresh_input(self):
        pipeline = Mock(side_effect=lambda x: x)
        inputs = np.ones((2, 4, 2, 2, 3), dtype=np.float32)
        inputs[1] *= 2
        with patch("benchmark.benchmark_timing.time.perf_counter", side_effect=[1, 1.2, 2, 2.4, 3, 3.6]):
            report = measure_pipeline(pipeline, inputs, "cpu", warmup=2, repeats=3)
        self.assertEqual(pipeline.call_count, 5)
        np.testing.assert_allclose(report["latencies_s"], [0.2, 0.4, 0.6])
        self.assertAlmostEqual(report["median_latency_s"], 0.4)
        self.assertEqual([call.args[0][0, 0, 0, 0, 0].item() for call in pipeline.call_args_list], [1, 2, 1, 2, 1])

    def test_cli_always_measures_fno_reference_and_requires_repeats(self):
        args = parse_args(["--models", "ensemble", "ensemble", "--device", "cpu"])
        self.assertEqual(args.models, ["fno_fp16", "ensemble"])
        self.assertEqual(args.warmup, 5)
        self.assertEqual(args.repeats, 30)

    def test_importer_reuses_runs_and_excludes_self_metrics_from_competition_mean(self):
        import benchmark.log_benchmark_scores_to_mlflow as importer

        with tempfile.TemporaryDirectory() as directory:
            score_dir = Path(directory)
            runs = []
            timing = {"protocol": {"device": "cpu"}, "models": {}}
            for index, (name, run_name) in enumerate(importer.MODEL_RUNS.items()):
                folder = score_dir / name
                folder.mkdir()
                (folder / "scores.json").write_text(json.dumps(self.legacy), encoding="utf-8")
                runs.append(SimpleNamespace(info=SimpleNamespace(run_id=name, run_name=run_name), data=SimpleNamespace(tags={})))
                timing["models"][name] = {"median_latency_s": 0.2 * (index + 1)}
            client = Mock()
            client.search_runs.return_value = runs
            with patch.object(importer, "SCORE_DIR", score_dir), patch.object(importer, "mlflow") as tracking, \
                 patch.object(importer, "MlflowClient", return_value=client), \
                 patch.object(importer, "optional_self_timing", return_value=(timing, score_dir / "timing.json")):
                tracking.get_experiment_by_name.return_value = SimpleNamespace(experiment_id="experiment")
                importer.main([])
            client.create_run.assert_not_called()
            client.delete_run.assert_not_called()
            logged = {(call.args[0], call.args[1]): call.args[2] for call in client.log_metric.call_args_list}
            self.assertNotIn(("fno_fp16", "time_score"), logged)
            self.assertEqual(logged["fno_fp16", "timescore_competition"], self.legacy["time_score"])
            self.assertEqual(logged["fno_fp16", "timescore_self"], 1)
            self.assertEqual(logged["fno_fp16", "latency_ms_self"], 200)
            self.assertAlmostEqual(logged["fno_fp16", "mean_subscore"], sum(self.legacy.values()) / 5)


if __name__ == "__main__":
    unittest.main()
