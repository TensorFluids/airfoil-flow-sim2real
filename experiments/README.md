# Track 1 Experiments

This is where all our custom work lives. The starting kit files one level up are
read-only references — never modify them.

## Controlled TKE-loss study

See [TKE_ABLATION.md](TKE_ABLATION.md) for the current four-arm stacked-mixer
protocol: field MSE alone, physical TKE penalties at lambda 0.05 and 0.10, and
the original normalized per-channel variance penalty at beta 1.0. Each arm
uses the same initialization, trajectory split and 100-epoch budget.

Run these commands from the **Track1 repository root**, not this subfolder:

```powershell
# Inspect without training or creating runs.
python experiments/finetune/train_tke_ablation.py --plan

# Run/resume four arms; validation scoring only, never automatic test scoring.
python experiments/finetune/train_tke_ablation.py
```

Optional `--prepare-only` builds the shared baseline cache and records initial
loss/gradient diagnostics without optimizer steps. The default checkpoint
rule is minimum validation field MSE, shared across arms; final checkpoints
are retained and scored too. After reviewing validation trade-offs, use the
documented `--evaluate-selected ARM --selection-reason "..."` command to lock
and test one checkpoint in its existing MLflow run. Test scores must not be used
to choose another arm. The loss remains full-grid because the local data has
no explicit validity-mask dataset; see the protocol for limitations.

Generated caches, checkpoints, scores and predictions remain Git-ignored.
The packaging instructions and competition rules below are historical starting-kit
notes, not requirements for running this standalone research study.

## Research timing

For the separate research timing benchmark, run the following from `Track1`:

```powershell
python experiments/benchmark/benchmark_timing.py
python experiments/benchmark/compare_models.py
python experiments/benchmark/log_benchmark_scores_to_mlflow.py
```

`timescore_competition` is the original time formula. `timescore_self` is fresh
FNO-relative speedup, and `latency_ms_self` is median time per complete forecast.
The benchmark always measures FNO in the same run. Use `--models fno_fp16 ensemble`
to time a subset, or `--device cpu` to benchmark on CPU. Cached base predictions
are never substituted for ensemble forward passes. Details and compatibility
notes are in [EVALUATION_DECISIONS.md](EVALUATION_DECISIONS.md).

## Structure

```
experiments/
├── models/         custom model architectures (CNN, U-Net, ConvLSTM, hybrids, etc.)
├── ensemble/       ensemble strategies combining FNO + CNO + Transolver + custom models
├── finetune/       fine-tuning scripts for the provided baselines on train_real
├── results/        saved prediction .npz files and local scores.json outputs
└── submissions/    ready-to-zip submission folders (each is a self-contained archive root)
```

## How to run an experiment

1. **Define your model** in `models/` or `ensemble/`
2. **Train / fine-tune** using data from `../../data/train_real/` and `../../data/train_sim/`
3. **Save predictions** to `results/<experiment_name>/predictions.npz`
4. **Score locally** using the kit's scoring.py:
   ```
   python ../scoring.py results/<experiment_name>/ results/<experiment_name>/
   ```
5. **Package for submission** into `submissions/<experiment_name>/`:
   - `submission.py` at root
   - checkpoint `.pth` at root
   - `load_baseline.py`, `rpde_baselines/`, `_vendor/` copied from the kit
   - Zip it: total extracted size must be < 256 MB

## Key paths

| Resource | Path |
|----------|------|
| Real training data | `../../data/train_real/` (exclude `7575_0.h5`!) |
| Sim training data | `../../data/train_sim/` |
| Baseline checkpoints | `../checkpoints/sim_real_ft/` |
| Example data | `../example_data/3750_0.h5` |
| Kit scoring script | `../scoring.py` |
| Kit load_baseline | `../load_baseline.py` |

## Rules reminder

- Training data: only `train_real/` and `train_sim/` from this repo
- No pretrained weights from outside the competition release
- Exclude `train_real/7575_0.h5` (known bad file)
- Submission size < 256 MB extracted
- No `scipy`, `h5py`, `matplotlib`, `pandas` in `submission.py`
- `DataLoader` must use `num_workers=0`
- See `../../RULES.md` for the full ruleset
