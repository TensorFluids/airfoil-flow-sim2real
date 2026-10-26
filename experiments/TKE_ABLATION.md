# Stacked-ensemble TKE-loss ablation

Protocol v2 compares four fresh copies of `SpatialAttentionMixer`. No study
training was run while implementing this protocol.

## Branch purpose and forecasting task

This experiment is maintained on `tke-loss-ablation`, created from
`personal-timescore-metric` at `7c832e2`. The parent contains the separate
competition/self timing metrics; this branch adds training-loss comparisons.
The TKE runner reports prediction quality, not the timing benchmark's metrics.

Each HDF5 file is a real PIV velocity recording at a fixed Re/AOA. The dataset
takes every second spatial point to reduce 64x128 measurements to 32x64 and
extracts 20 measured input frames followed by 20 measured target frames, with
start positions separated by 20 frames. Adjacent examples share an observed/
target time block, so the file-level split keeps them together. Inputs are
measured histories; predictions are not recursively fed into later examples.

The supplied `sim_real_ft` FNO/CNO/Transolver weights already incorporate
simulation pretraining and real-data fine-tuning. They stay frozen here. A
cached nine-channel stack of their normalized predictions is passed to the
existing mixer; only its weights are optimized. No CFD training files or new
baseline training are required for this study.

The goal is to compare physical fluctuation supervision with no penalty and
with the older variance formulation, while keeping all other training choices
fixed. A TKE gain alone is not enough to declare success: field error, mean
profiles, and SPS must be considered together on validation data.

## The four arms

| Arm key | Objective | Purpose |
|---|---|---|
| `lambda_0` | Field MSE only | No auxiliary penalty control |
| `lambda_0p05` | Field MSE + 0.05 * scale * physical TKE MSE | Weak physical TKE supervision |
| `lambda_0p1` | Field MSE + 0.10 * scale * physical TKE MSE | Stronger physical TKE supervision |
| `legacy_variance` | Field MSE + original normalized variance MSE, beta = 1.0 | Original-loss reference |

All four start from the same newly initialized tensors, not an old trained
checkpoint. D reproduces the old **loss**, not the old scheduler/early-stopping
recipe: training settings are held fixed across this study. It compares old
versus new objectives but does not independently isolate formulation from
weighting. Use `--skip-legacy` before preparing the study to run only A/B/C.

## Loss definitions and mask policy

The field loss is the original MSE over normalized `[u,v,p]` outputs. Physical
targets retain float32 precision and original zeros used by the SPS scorer.

```text
L_field = MSE(pred_normalized, target_normalized)
k(x) = 0.5 * (population_variance_over_time(u) + population_variance_over_time(v))
L_TKE = MSE(k(pred_physical), k(target_physical))
scale = initial_L_field / initial_L_TKE
L_new = L_field + lambda * scale * L_TKE

L_legacy_variance = MSE(sample_variance_over_time(pred_normalized),
                        sample_variance_over_time(target_normalized))
L_D = L_field + 1.0 * L_legacy_variance
```

Physical TKE uses physical `u,v`, combines their variances, and uses
`unbiased=False` (population variance), matching `scoring.py`. The legacy term
uses normalized `u,v,p` separately with `unbiased=True` (sample variance,
correction = 1), exactly as in `train_ensemble.py`. Pressure is unavailable in
the measurements; its target is zero. Its prediction still participates in
the original training objective. Evaluation sets predicted pressure to zero.

**All loss terms remain full-grid/unmasked**, including solid/unmeasured regions.
Inspection of all 81 usable local HDF5 files found only `aoa,re,t,u,v,x,y`
datasets, with no explicit validity-mask dataset. A zero velocity is not by
itself proof of invalid measurement; no mask is fabricated from per-frame
nonzero values. Supplied L2/TKE/MVPE scores are also unmasked; only SPS follows
the scorer's nonzero-target mask. This policy is logged in the manifest and
MLflow. It preserves the original-loss comparison but does not claim that
full-grid supervision is physically ideal. A future masked-loss experiment
needs a verified geometry/PIV-validity mask and, for time-varying missing
measurements, masked temporal means/variances. Treat that as a separate change.

## Calibration and gradient diagnostics

Before optimization, the shared initial model measures field and physical TKE
losses on 64 deterministically sampled **training** windows. Their ratio is
frozen and shared across physical-TKE arms. Lambda = 0.05/0.10 means an initial
scalar contribution of approximately 5%/10% on that sample, not necessarily
the same influence on optimization. D uses beta = 1.0 without this calibration.

On the first deterministic batch of those calibration windows, before any
optimizer step, the runner also records:

- Parameter-gradient norms for field MSE, physical TKE MSE, and legacy variance MSE.
- `lambda * scale * ||grad(L_TKE)|| / ||grad(L_field)||` for each physical arm,
  and `||grad(L_legacy_variance)|| / ||grad(L_field)||` for D.
- Gradient cosine similarities between each penalty and field MSE, raw batch
  loss magnitudes, and exact window indices.

These use `autograd.grad`, do not populate parameter `.grad`, and do not
update weights. They describe gradients **before clipping and Adam**, not
actual optimizer-update magnitudes. They do not automatically retune weights.
Inspect `calibration.json` or use `--prepare-only` to review diagnostics before
training. Changed weights require a new study ID and must be chosen without test.

## Fixed setup and checkpoint selection

- Seed 42; trajectory split approximately 70%/15%/15%. Current local data:
  57/12/12 trajectories and 2381/485/475 train/validation/test windows.
- Exclude `7575_0.h5` and names beginning with `exclude_`. All windows from a
  trajectory stay together; content hashes detect renamed copies across splits.
- Same initial tensors, hidden width 32, batch size 16, Adam, constant learning
  rate `1e-4`, gradient clipping at 1.0, and epoch-wise batch permutations.
- Exactly **100 epochs per arm** by default. No early stopping or adaptive LR.
- Save `last.pth` each epoch (model, optimizer, history and RNG states) and
  `best_validation_field.pth` when validation field MSE strictly improves.
  Ties retain the earlier checkpoint. The shared criterion is field MSE,
  **not** each arm's differently weighted total loss.
- Default primary checkpoint: **lowest validation field MSE**. This prioritizes
  field fidelity; it does not claim to maximize TKE, SPS, or a multi-objective
  utility. Every arm completes the full budget. All four quality scores are
  computed for this checkpoint after training, not after every epoch.
- Also score final checkpoints on validation for the fixed-final-epoch ablation.
  To make final checkpoints the primary comparison, supply
  `--checkpoint-selection final_epoch` before preparation. Do not switch rules
  after inspecting test results.
- Frozen FNO/CNO/Transolver predictions are cached once and shared across arms.
  Baselines load one at a time; arrays are memory-mapped (~3.83 GiB). Predictions
  use float16 storage; targets and mixer computations use float32. Test predictions
  may be cached during preparation, but are not scored or used for calibration,
  gradients, training, or checkpoint selection.
- PyTorch deterministic algorithms are required; unsupported operations fail.
  Changing code, data or settings under an existing ID is rejected.

These trajectories are held out from **new mixer training**. Supplied baselines
may have seen them originally. This is an ablation of learned corrections, not
evidence that the complete pretrained system generalizes to entirely unseen
conditions. One paired seed does not measure seed variability.

## Validation first, locked test second

Training ends after validation evaluation of every arm. The comparison report
uses validation L2, TKE, MVPE and SPS, including changes relative to A. It does
**not** automatically declare a winner or evaluate any test scores.

Choose acceptable deterioration limits before inspecting results. Optional
guardrails, supplied together before preparing the study, are:

```text
--max-l2-increase-pct <percent>
--max-mvpe-increase-pct <percent>
--max-sps-drop-points <score-points>
```

A candidate passes only if validation TKE error decreases and all three limits
are met relative to A. With preset limits, a reasonable selection rule is lowest
validation TKE error among acceptable candidates, retaining A if none qualifies.
The script reports pass/fail but leaves the final choice explicit. Without
limits, acceptability is not computed: highest TKE is not an automatic winner.
Document the validation-only selection rule/rationale; preset limits are
preferable to post-hoc judgments.

Select **one** arm with `--evaluate-selected` and provide `--selection-reason`.
The command requires validation scores for every requested arm and verifies
that the selected checkpoint matches its validation scores. It writes
`test_selection.json` **before** exposing test results, locking the arm and
checkpoint hash. Retries reuse this choice and any saved test scores. Another
arm/checkpoint is refused within that study. Test metrics attach to the selected
model's existing run; no evaluation-only MLflow run is created. Keep the test
split locked across later studies too: a new ID is not a scientific reset of
a test set already used to tune decisions.

## Commands

From `Track1`, with the project environment activated:

Required files under the default layout:

```text
RealPDE/
  data/train_real/*.h5
  Track1/
    checkpoints/sim_real_ft/sim_real_fno_fp16.pth
    checkpoints/sim_real_ft/sim_real_cno.pth
    checkpoints/sim_real_ft/sim_real_transolver.pth
```

`--data-dir PATH` changes the input data location. Baseline checkpoint paths
are fixed to the files above. Preparation hashes and reads the selected HDF5
files/checkpoints, then runs the frozen models to create the shared cache.
The old 42-window example cache does not supply this larger split. Allow disk
space for shared arrays, per-arm predictions and checkpoints; 3.83 GiB describes
only the current shared arrays, not the whole study's eventual disk usage.

```powershell
# Read-only: inspect actual splits and settings.
python experiments/finetune/train_tke_ablation.py --plan

# Optional: cache frozen predictions and inspect scalar/gradient calibration.
# No optimizer steps or MLflow runs; baseline forward passes can take time.
python experiments/finetune/train_tke_ablation.py --prepare-only

# Run/resume all FOUR arms sequentially; stop after validation scoring.
python experiments/finetune/train_tke_ablation.py

# LATER, after validation review: example syntax, NOT a recommendation of this arm.
python experiments/finetune/train_tke_ablation.py --evaluate-selected lambda_0p05 --selection-reason "Chosen using the documented validation trade-off rule"
```

Default ID: `stacked-tke-v2`. Preserve **all** settings across preparation,
training/resume and selected-test commands (including custom lambdas, skip-legacy,
epoch count, checkpoint rule and guardrails). Evaluation switches and the reason
are not part of the immutable training configuration. Use a new ID if changing
the protocol; v1 studies are not silently migrated.

Rerunning training resumes interrupted arms from the last completed epoch,
restores optimizer/RNG state, reuses each run and skips completed arms. An
incomplete epoch is repeated. Old single-example caches do not cover this
larger split; first preparation creates fresh frozen predictions. Interrupted
cache creation rebuilds the incomplete shared cache.

The manifest fingerprints the runner, numerical helper, dataset, mixer and
scorer, among other inputs. In particular, the timing branch renamed the
competition scorer's time key. A study prepared before that source change may
refuse to resume even though this study's numerical quality functions did not
change. Use its recorded code version when resuming; do not rewrite manifests
or mix old checkpoints into a newly specified sweep. Merely creating this
branch does not migrate existing caches, manifests or checkpoints.

To inspect an existing study without launching training, read its
`manifest.json`, `calibration.json`, and results `checklist.json`. The `--plan`
command describes the requested current configuration; it does not certify
that it matches a previously prepared manifest.

## MLflow, files and timing

Default experiment: `airfoil-flow-forecasting`, local store `mlflow_runs/`.
One run per arm: `stacked-tke-v2-lambda_0`, `...-lambda_0p05`, `...-lambda_0p1`,
and `...-legacy_variance`. Each contains:

- Epoch curves for training/validation loss, field MSE, raw/scaled physical TKE,
  legacy variance MSE, applied auxiliary penalty and learning rate. The physical
  TKE penalty is zero for D; D's auxiliary penalty is its legacy variance term.
- Initial scalar/gradient diagnostics, effective coefficients, hashes and manifest.
- Last and best-validation-field checkpoints, selected checkpoint epoch.
- Primary quality scores prefixed `validation_`; fixed-final-epoch scores prefixed
  `validation_final_`; only the selected arm receives `test_` scores.
  Per-trajectory metrics use `validation/`, `validation_final/`, and `test/`.
- `checklist_status`, comparison/checklist artifacts, validation changes versus A,
  and (for the selected arm) `selected_for_test` / `test_status` tags.
- `epoch_seconds` and a printed ETA for the **current arm**, excluding later arms,
  initial caching, scoring and checkpoint I/O.

Raw relative errors are aggregated by window count before score conversion.
SPS and coverage use scored target-element counts, matching joint scoring.
There is no timing score or five-component mean score in this study.

Protocol/checkpoints: `checkpoints/tke_ablation/<sweep-id>/`.
Predictions/scores/report: `experiments/results/tke_ablation/<sweep-id>/`, with
`<arm>/validation`, `<arm>/validation_final` and selected `<arm>/test` subfolders.
Shared cache: `experiments/cache/tke_ablation/`. Generated data is Git-ignored.

Runtime has not been benchmarked for this exact runner. Older local HMoE logs
show roughly 45 seconds/epoch; 400 epochs at that rate would be about 5 hours
**before** initial cache preparation and scoring. The stacked mixer is a different,
simpler architecture; deterministic settings and disk/cache overhead also differ.
Treat this as a planning reference, not a measured prediction. Allow an overnight
window and revise expectations from the first completed epochs. Keep the laptop
plugged in and prevent system sleep; closing the terminal that owns the process
can interrupt it, though the next run resumes saved epochs.

## Verification

```powershell
python -m unittest discover -s experiments/tests -p test_tke_ablation.py -v
```

Tests cover loss definitions, no-update calibration/gradients, shared splits and
batches, aggregation, guardrails, validation-only orchestration, checkpoint
selection, run reuse and the selected-test lock. Orchestration tests mock epoch
execution and external MLflow writes; they do not run a training job.
