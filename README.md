# Airfoil Flow Forecasting

Research code for predicting experimental airfoil flow fields from short histories of measured velocity. The project explores whether compact learned corrections and model combinations can improve on neural-operator baselines, especially on temporal fluctuation and turbulence metrics.

## Problem

The task is to forecast flow around a NACA4418 airfoil. Each example contains 20 observed frames on a `32 × 64` grid; the model predicts the following 20 frames. The real measurements provide horizontal and vertical velocity (`u`, `v`). Pressure (`p`) is unavailable in the measurements and is represented as zero.

The data comes from paired experimental particle-image-velocimetry (PIV) measurements and computational-fluid-dynamics (CFD) simulations. This repository focuses on the real-flow forecasting experiments. The dataset and supplied baseline checkpoints are described in the [official data release](https://huggingface.co/datasets/AI4Science-WestlakeU/RealPDE-Competition-Data).

## Experiments

The repository contains three exploratory approaches:

- **FNO + ConvLSTM:** a frozen Fourier Neural Operator prediction with a learned ConvLSTM residual correction.
- **Learned baseline ensemble:** combines predictions from frozen FNO, CNO, and Transolver models with a learned spatiotemporal mixer.
- **FST-HMoE mixer:** combines the same baseline predictions using temporal and local spatial attention with a mixture-of-experts block.

An **untrained uniform-average control** combines FNO, CNO, and Transolver with fixed weights of `1/3` each. Comparing both trainable mixers with this control measures what their learned corrections add beyond averaging.

Training code is under `experiments/finetune/`, model definitions are under `experiments/models/`, and local evaluation tools are under `experiments/benchmark/`.

## TKE loss experiments: what this branch adds

The `tke-loss-ablation` branch was created from `personal-timescore-metric` at
commit `7c832e2`. It therefore includes the timing metrics and adds a separate
training study. The timing branch keeps the committed timing implementation;
this branch stores the TKE experiment code, dataset extensions, tests and protocol.

The research question is: **does directly supervising physical velocity
fluctuation energy improve the stacked ensemble without unacceptable loss of
field accuracy, mean velocity profile accuracy, or SPS?** This study changes
the training objective of the existing `SpatialAttentionMixer`, rather than
introducing a new forecasting architecture. FNO, CNO and Transolver remain
frozen; only the mixer is trained. The supplied checkpoints have already been
pretrained on simulation and fine-tuned on PIV data. This study reads real PIV
recordings; it does not retrain the baseline models or use CFD training files.

Four freshly initialized mixers are compared:

| Arm | Objective | Why it is included |
|---|---|---|
| `lambda_0` | Normalized field MSE | Control with no auxiliary penalty |
| `lambda_0p05` | Field MSE + 0.05 * calibrated physical TKE MSE | Weak physical TKE supervision |
| `lambda_0p1` | Field MSE + 0.10 * calibrated physical TKE MSE | Stronger physical TKE supervision |
| `legacy_variance` | Field MSE + original normalized per-channel variance MSE, beta = 1 | Reference to the older ensemble loss |

The original variance penalty and the new physical TKE loss are different
formulations. The former compares normalized per-channel sample variances;
the latter compares half the sum of physical u/v population variances. The
legacy arm reproduces the old loss under this study's common training setup,
not the old training run or its scheduler/early stopping. The zero-lambda
control is field MSE alone. Exact formulas and the full-grid mask policy are in
[TKE_ABLATION.md](experiments/TKE_ABLATION.md).

### What is held fixed

- Seed 42 and identical initial mixer weights, epoch-wise batch order, optimizer
  and budget across arms; the old trained mixer is not used as initialization.
- Default budget: 100 epochs per arm, batch size 16, hidden width 32, Adam with
  constant learning rate `1e-4`, gradient clipping at 1.0, no early stopping.
- A file-level train/validation/test split, approximately 70%/15%/15%. With the
  current local data this is 57/12/12 trajectories and 2381/485/475 windows.
  Windows from the same recording stay in one split. Known bad/excluded files
  are skipped, and content hashes detect renamed copies across splits.
- One shared disk cache of frozen-model predictions (~3.83 GiB for the current
  split). Baseline predictions use float16 storage; targets retain physical
  float32 values and original zeros. Mixer training uses float32.
- A physical-TKE scale measured once on 64 deterministic training windows
  before optimization. Initial gradient norms and weighted gradient ratios are
  also recorded; scalar loss balance does not guarantee equal gradient influence.

### How to run and resume

From `Track1`, with the project environment activated:

```powershell
# Read-only inspection: prints configuration and actual trajectory splits.
python experiments/finetune/train_tke_ablation.py --plan

# Optional preparation: frozen-model forward passes and loss/gradient diagnostics.
# This creates caches and protocol files, but performs no optimizer steps.
python experiments/finetune/train_tke_ablation.py --prepare-only

# Train/resume all four arms sequentially, then evaluate validation trajectories.
python experiments/finetune/train_tke_ablation.py
```

Default study ID: `stacked-tke-v2`. Rerun the exact same command to resume from
the last completed epoch, including optimizer and RNG state. Completed arms
are skipped and their existing MLflow runs reused. A partially completed epoch
is repeated. Data/code/configuration fingerprints reject silently changing a
prepared study; a branch switch does not make an old manifest compatible with
different source files. Do not edit a manifest to bypass this protection.

### Validation, test selection and saved outputs

Every arm completes its full epoch budget and retains both `last.pth` and
`best_validation_field.pth`. The default primary checkpoint is the one with
lowest validation field MSE, a common criterion across all objectives. Each
final checkpoint is also scored separately for the fixed-budget ablation.

Training stops after validation evaluation. Compare validation relative L2,
TKE, MVPE and SPS; define acceptable trade-offs before selecting a winner.
Optional L2/MVPE/SPS deterioration limits are documented in the protocol.
The training command does not score the test split automatically.

After validation review, the separate `--evaluate-selected ARM` command with a
`--selection-reason` locks one arm and checkpoint hash before test scoring.
It attaches test metrics to that model's existing MLflow run. Once test results
are exposed, do not use them to choose another configuration. See the
[complete commands](experiments/TKE_ABLATION.md#commands) before this step.

- Protocol, calibration, initial weights and checkpoints:
  `checkpoints/tke_ablation/<sweep-id>/`.
- Shared frozen predictions: `experiments/cache/tke_ablation/`.
- Scores, per-trajectory predictions and comparison/checklist files:
  `experiments/results/tke_ablation/<sweep-id>/`.
- MLflow experiment: `airfoil-flow-forecasting`, one model run per arm with
  epoch curves and validation/test metric prefixes. All generated artifacts
  remain outside Git.

No training job is started by committing or checking out this branch. The code
is an experiment implementation, not evidence that a particular loss wins.
Holdouts are excluded from the new mixer training; they are not guaranteed
unseen by the supplied pretrained baselines. The losses remain full-grid
because no explicit validity mask is present in the inspected release, and a
single seed does not establish robustness across independent training runs.

For setup, limitations, all CLI options, exact loss definitions and tests, use
the [full experiment protocol](experiments/TKE_ABLATION.md).

## Evaluation

The [research evaluation decisions](experiments/EVALUATION_DECISIONS.md) define the timing metrics. `timescore_competition` preserves the original 0-100 formula. `timescore_self` is the measured FNO latency divided by the complete proposed model's latency: 1 means equal speed, above 1 means faster, and below 1 means slower. `latency_ms_self` reports the measured median latency. The supplied fine-tuned FNO is the primary reference; new self measurements require a fresh benchmark.

To measure timing and attach the results to the existing model runs, use:

```powershell
python experiments/benchmark/benchmark_timing.py
python experiments/benchmark/compare_models.py
python experiments/benchmark/log_benchmark_scores_to_mlflow.py
```

The timing benchmark uses batch size 1, 5 warm-up forecasts and 30 measured forecasts per model. It includes normalization, all required model forward passes, output conversion, and CPU/GPU input/output transfers; it excludes checkpoint loading and file I/O. All component models stay resident during a timed forecast. Use the same device for all models; `--device cpu` is available if GPU memory is insufficient. Predictions used for accuracy are not overwritten. Only models present in a validated timing report receive self metrics; the report records environment, input, code and checkpoint fingerprints. See the [decision record](experiments/EVALUATION_DECISIONS.md#implementation-and-commands) for details.

Prediction quality is described by relative velocity error, turbulent kinetic energy (TKE) error, mean velocity profile error (MVPE), and Safe Prediction Score (SPS), which considers prediction accuracy and uncertainty intervals. Research timing is reported separately. Legacy competition reports retain five subscores, including `timescore_competition`; `mean_subscore` averages those five only and excludes self metrics.

The original benchmark scripts evaluate windows from a single example trajectory. Those results are preliminary and do not establish generalization across airfoil operating conditions. In particular, some recorded runtime measurements exclude the baseline-model forward passes, so they should not be read as end-to-end timings. The TKE-ablation runner instead uses disjoint trajectory-level training, validation and test splits. These splits are held out from the new mixer training, not guaranteed unseen by the supplied pretrained baselines.

Once the three baseline predictions are cached, run the control and comparison from the repository root:

```bash
python experiments/benchmark/run_average_benchmark.py
python experiments/benchmark/compare_models.py
python experiments/benchmark/log_benchmark_scores_to_mlflow.py --models uniform_average
```

This reuses the physical-space predictions without training or loading the baseline models again. The comparison report includes raw prediction-error reductions against the average. Its runtime is an estimate from cached baseline times plus measured averaging overhead; use prediction errors to assess the mixers until all models have comparable end-to-end timings. The control's scores are stored in one `uniform-average` MLflow run, which is reused on subsequent imports.

## Repository contents

- `experiments/models/` — custom model architectures.
- `experiments/finetune/` — data loading and training scripts.
- `experiments/benchmark/` — prediction, scoring, and visualization scripts.
- `rpde_baselines/` — supplied FNO, CNO, and Transolver implementations used by the experiments.
- `scoring.py` — local implementation of the task metrics.
- `requirements.txt` — Python dependencies.

Model checkpoints, downloaded data, generated results, and packaged submissions are kept outside version control. The dataset is released under CC BY-NC 4.0; see its [license and usage notes](https://huggingface.co/datasets/AI4Science-WestlakeU/RealPDE-Competition-Data).

## Setup

Use Python 3.10 or newer and install the listed dependencies:

```bash
python -m pip install -r requirements.txt
```

The training scripts need the released real PIV data and baseline checkpoints. Download those from the official data release and place them in the paths expected by the scripts. Some scripts currently contain machine-specific data paths; making those configurable is planned before the training instructions are considered portable.

The baseline loader can load FNO, CNO, and Transolver checkpoints. Checkpoints are not included in this repository; see the official data release for available files.

## Experiment tracking

The training scripts log hyperparameters, per-epoch training and validation metrics, and the best custom checkpoint to a local MLflow store in `mlflow_runs/`. That folder is ignored by Git. The default experiment is `airfoil-flow-forecasting`; you can override the tracking URI or experiment name with `MLFLOW_TRACKING_URI` and `MLFLOW_EXPERIMENT_NAME`.

To import the saved FNO, CNO, and Transolver baseline scores into the same experiment, first generate and score their local benchmark predictions, then run:

```bash
python experiments/benchmark/log_baselines_to_mlflow.py
```

Start the local comparison UI from the Track1 directory with:

```bash
mlflow ui --backend-store-uri ./mlflow_runs --host 127.0.0.1 --port 5000
```

Then open `http://127.0.0.1:5000` in a browser. Baseline runs import the existing local score files under `experiments/results/scores/`; they do not copy the large pretrained checkpoints into MLflow.

## Status

This is an in-progress research project. The current code records initial model experiments and local benchmark comparisons, and provides a controlled four-arm TKE-loss protocol with trajectory-level holdouts. The protocol's implementation is not evidence of completed experiments. Next steps include running the study, checking robustness across seeds, improving path portability in older scripts, and revisiting end-to-end timing.

## Acknowledgments

This work builds on the RealPDEBench airfoil PIV/CFD dataset and the supplied FNO, CNO, and Transolver baselines. Their original code and dataset documentation should be credited in any public release or publication. The experiment code in this repository is exploratory work built around those resources.
