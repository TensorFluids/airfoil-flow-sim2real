# Research evaluation decisions

## 2026-10-25: Timing measurements and baseline selection

Status: implemented on branch `personal-timescore-metric`; fresh timing
measurements and a live MLflow import have not been run as part of implementation.

### Report inference time instead of a competition time score

For the personal research project and a potential paper, report measured
inference latency in seconds or milliseconds per complete 20-frame forecast.
Do not invent a replacement 0-100 time score.

The competition's original `time_score`, now named `timescore_competition`, uses a fixed numerical-solver reference of
`0.72896` seconds. That constant is present in the original v9 starting kit's
`scoring.py`, but the kit does not document the solver, hardware, or measurement
procedure that produced it. Keep existing scores as historical benchmark
results; they do not establish a controlled speedup over CFD.

### Use the supplied FNO as the primary reference

Use the supplied simulation-pretrained, real-data-fine-tuned FNO checkpoint,
currently `checkpoints/sim_real_ft/sim_real_fno_fp16.pth`, as the primary model
reference for prediction quality and inference speed.

The basis for this choice is the existing local benchmark: FNO leads CNO and
Transolver on relative L2, TKE, MVPE, and SPS scores across 42 windows from
`example_data/3750_0.h5` (Re = 3750, AOA = 0 degrees). This is a preliminary
single-recording result; confirm the comparison on broader evaluation data.

Retain CNO and Transolver as additional comparisons. For learned ensembles,
also retain the untrained uniform average of all three models as a control:
it tests whether fitting a mixer adds value beyond averaging.

### Define speedup using actual measured latency

```text
speedup_vs_fno = latency_fno / latency_complete_proposed_model
```

- A value above 1 means the proposed model is faster than FNO.
- A value below 1 means the proposed model takes longer than FNO.
- For a three-model ensemble, include FNO, CNO, Transolver, and the mixer in
  the complete prediction cost. Timing only the mixer on cached predictions
  is not the latency of forecasting a fresh input.
- Remeasure models on the same hardware and with consistent settings and timed
  boundaries. Document batch size, numerical precision, software environment,
  warm-up procedure, and repeated measurements. The implemented defaults and
  timed boundary are recorded below.

Report quality and computational cost separately so an accuracy improvement
can be assessed alongside its additional cost. Do not take a ratio of the
transformed competition time scores to calculate speedup.

### Preserve the historical FNO result with its scope

The saved FNO `time_score` is `63.88099828966087`, corresponding to approximately
`0.233` seconds per example under the old batched measurement, rather than a
direct batch-one latency measurement. Its source is
`experiments/results/scores/fno_fp16/scores.json` (generated and Git-ignored).
It is historical context, not a newly measured reference for future runs.

Existing `mean_subscore` values average the five competition subscores and
therefore include that legacy time score. They are not the newly agreed
latency or speedup measurements. The scorer, report generator, and importers now
use `timescore_competition`. Training objectives, existing MLflow records and
generated results were not changed during implementation.

## Implementation and commands

The requested names are:

| Metric | Definition | Interpretation |
|---|---|---|
| `timescore_competition` | `100 / (1 + sqrt(old_recorded_seconds / 0.72896))` | Legacy 0-100 competition score |
| `timescore_self` | `median_fno_seconds / median_complete_model_seconds` | Measured speedup ratio; FNO = 1, faster > 1; not bounded at 100 |
| `latency_ms_self` | `1000 * median_complete_model_seconds` | Actual median latency per 20-frame forecast; lower is better |

`timescore_self` names the previously agreed speedup measurement; it is not a
new nonlinear 0-100 transformation. It and raw latency remain separate from
`mean_subscore`, which still averages the five competition subscores only.

From `Track1`, after activating the project environment and preparing
`experiments/results/ref/samples.npz`:

```powershell
# Fresh inference measurements; no training or MLflow writes.
python experiments/benchmark/benchmark_timing.py

# Optional subset; FNO is always included automatically as the reference.
python experiments/benchmark/benchmark_timing.py --models ensemble --device cuda

# Recompute accuracy/competition scores and join validated fresh self timings.
python experiments/benchmark/compare_models.py

# Import the renamed and self metrics into each existing model run.
python experiments/benchmark/log_benchmark_scores_to_mlflow.py
```

The benchmark defaults to all three individual baselines, the uniform average,
the stacked mixer and HMoE. It uses batch size 1, 5 warm-up calls and 30 timed
calls per model, cycling over the same input windows in the same order. Latency
is the median of those calls; raw samples and the 25th/75th percentiles are saved.
`--warmup`, `--repeats`, `--num-threads`, `--device`, `--samples` and `--output`
are configurable. The default CPU thread count is 1; computation is float32
with autocast and TF32 disabled.

The counts 5 and 30 are configurable starting defaults chosen during
implementation, not organizer requirements or empirically validated counts for
this hardware. With 42 available windows, the default times windows 0-29 once
each after warm-up; it does not measure 30 complete dataset passes. Before paper
reporting, check warm-up and timing stability and consider repeated passes over
all evaluation windows.

Timing starts with an already-loaded physical input array in CPU memory and
ends with the physical prediction back in CPU memory. It includes input tensor
creation/transfer, normalization, every required base-model forward, the mixer
or averaging operation, denormalization, pressure zeroing, and output transfer.
Checkpoint/model construction, input-file loading, scoring, and output-file
compression are excluded. CUDA is synchronized at the timing boundaries.
All required models stay resident for each measured forecast; if they do not
fit in GPU memory, choose CPU for the whole comparison or define and document a
separate deployment protocol. The timer does not silently switch to cached bases.

Complete runs are written atomically to `experiments/results/timing_self.json`.
The report contains hardware/software settings, sampled window indices, input
and checkpoint hashes, model/timing-code hashes, and repeated measurements.
Code, weight or input changes invalidate the report; rerun timing to update it.
These measurements are separate from old cached accuracy predictions; before
paper reporting, regenerate accuracy predictions from the exact weights being
timed if their cache provenance is uncertain.

Reports and importers auto-detect this file, or accept `--timings-self PATH`.
Absent measurements produce no self metrics rather than estimates from legacy
times. `--competition-only` explicitly skips self measurements. A timing report
covering a subset adds self metrics only to that subset. The ratio always uses
the FNO measured in that same report, not the historical 0.233-second value.

Old score JSON files with `time_score` remain readable; importers emit the new
`timescore_competition` key and recompute self metrics from a validated timing
report. Existing MLflow history is retained: the old `time_score` key can remain
visible on historical runs because no stored metric files were deleted. New
logging uses the renamed key and tags the self measurement definition/device.
Importers reuse existing model runs rather than creating evaluation-only runs.

The prepared TKE study fingerprints `scoring.py` as part of its immutable code
manifest. This branch's scorer rename can therefore cause an already-prepared
study to refuse resuming under changed code. Resume a study with its recorded
code version; do not edit its manifest to bypass that check. Timing changes do
not change the training loss.
