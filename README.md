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

## Evaluation

Experiments use five complementary measures: relative velocity error, turbulent kinetic energy (TKE) error, mean velocity profile error (MVPE), inference time, and Safe Prediction Score (SPS), which considers prediction accuracy and uncertainty intervals.

The checked-in experiment scripts currently evaluate against windows from a single example trajectory. Their results are preliminary and do not establish generalization across airfoil operating conditions. In particular, some recorded runtime measurements exclude the baseline-model forward passes, so they should not be read as end-to-end timings.

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

This is an in-progress research project. The current code records initial model experiments and local benchmark comparisons. Next steps include making data and checkpoint paths configurable, evaluating on trajectory-level held-out conditions, revisiting end-to-end timing, and documenting reproducible training runs.

## Acknowledgments

This work builds on the RealPDEBench airfoil PIV/CFD dataset and the supplied FNO, CNO, and Transolver baselines. Their original code and dataset documentation should be credited in any public release or publication. The experiment code in this repository is exploratory work built around those resources.
