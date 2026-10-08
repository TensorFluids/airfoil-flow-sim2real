"""compare_models.py – Score models and generate comparison plots / report.

Run from C:\\Desktop\\RealPDE\\Track1:
    python experiments/benchmark/compare_models.py

Reads:
    experiments/results/<model>/predictions.npz
    experiments/results/ref/targets.npz

Writes to experiments/results/:
    scores/<model>/scores.json
    comparison_table.txt
    bar_subscores.png
    line_rel_l2.png
    heatmap_u.png
    heatmap_v.png
    tke_comparison.png
    benchmark_report.md
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

# ── make kit importable ────────────────────────────────────────────────────────
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "_vendor"))

from scoring import (  # noqa: E402
    _score_submission,
    rel_l2_per_sample,
    tke_rel_l2_per_sample,
    kinetic_energy,
    measured_channels,
)

# ── paths ──────────────────────────────────────────────────────────────────────
RESULTS_DIR = os.path.join(REPO_ROOT, "experiments", "results")
REF_DIR = os.path.join(RESULTS_DIR, "ref")

MODELS = ["fno_fp16", "cno", "transolver", "ensemble", "fst_hmoe"]
SUBSCORES = ["rel_l2_score", "tke_score", "mvpe_score", "time_score", "sps_score"]


# ── helpers ────────────────────────────────────────────────────────────────────

def load_predictions(model: str) -> tuple[np.ndarray, float]:
    path = os.path.join(RESULTS_DIR, model, "predictions.npz")
    with np.load(path, allow_pickle=False) as d:
        pred = d["prediction"].astype(np.float32)
        mean_t = float(np.asarray(d.get("mean_t_neural_s", [0.0])).reshape(-1)[0])
    return pred, mean_t


def load_targets() -> np.ndarray:
    path = os.path.join(REF_DIR, "targets.npz")
    with np.load(path, allow_pickle=False) as d:
        return d["target"].astype(np.float32)


def score_model(model: str, targets: np.ndarray) -> dict:
    input_dir = os.path.join(RESULTS_DIR, model)
    reference_dir = REF_DIR
    scores = _score_submission(
        __import__("pathlib").Path(input_dir),
        __import__("pathlib").Path(reference_dir),
    )
    return scores


def save_scores(model: str, scores: dict) -> str:
    scores_dir = os.path.join(RESULTS_DIR, "scores", model)
    os.makedirs(scores_dir, exist_ok=True)
    path = os.path.join(scores_dir, "scores.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(scores, f, indent=2)
    return path


def print_table(all_scores: dict[str, dict]) -> str:
    col_w = 14
    header = f"{'Model':<14}" + "".join(f"{s:>{col_w}}" for s in SUBSCORES)
    sep = "-" * len(header)
    rows = [header, sep]
    for model, scores in all_scores.items():
        row = f"{model:<14}" + "".join(
            f"{scores.get(s, float('nan')):>{col_w}.2f}" for s in SUBSCORES
        )
        rows.append(row)
    table = "\n".join(rows)
    print(table)
    return table


# ── plotting ───────────────────────────────────────────────────────────────────

def plot_bar_subscores(all_scores: dict[str, dict], out_path: str) -> None:
    import matplotlib.pyplot as plt

    models = list(all_scores.keys())
    n_models = len(models)
    x = np.arange(len(SUBSCORES)) * (n_models + 1.5)  # extra gap between subscore groups
    width = 0.85  # width of each individual bar (gap of 0.15 between adjacent bars)

    fig, ax = plt.subplots(figsize=(max(10, n_models * 2.2), 5))
    for i, model in enumerate(models):
        vals = [all_scores[model].get(s, 0.0) for s in SUBSCORES]
        ax.bar(x + i * width, vals, width, label=model)

    ax.set_xticks(x + width * (n_models - 1) / 2)
    ax.set_xticklabels(SUBSCORES, rotation=15, ha="right")
    ax.set_ylabel("Score (0–100)")
    ax.set_title("RealPDE Track 1 – Baseline Subscores Comparison")
    ax.legend()
    ax.set_ylim(0, 105)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"Saved  {out_path}")


def plot_line_rel_l2(
    model_preds: dict[str, np.ndarray], targets: np.ndarray, out_path: str
) -> None:
    import matplotlib.pyplot as plt

    c = measured_channels(targets)
    fig, ax = plt.subplots(figsize=(9, 4))
    for model, pred in model_preds.items():
        errs = rel_l2_per_sample(pred, targets, c)
        ax.plot(errs, marker="o", markersize=3, label=model)
    ax.set_xlabel("Sample index")
    ax.set_ylabel("Relative L2 error")
    ax.set_title("Per-sample Rel-L2 Error")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"Saved  {out_path}")


def plot_heatmap(
    model_preds: dict[str, np.ndarray], targets: np.ndarray,
    channel: int, channel_name: str, out_path: str
) -> None:
    import matplotlib.pyplot as plt

    models = list(model_preds.keys())
    ncols = len(models) + 1  # last col = target
    fig, axes = plt.subplots(1, ncols, figsize=(4 * ncols, 4))

    # Mean absolute error averaged over (N, T); shape (H, W)
    tgt_c = targets[..., channel]  # (N, T, H, W)
    tgt_mean = np.mean(np.abs(tgt_c), axis=(0, 1))

    vmin, vmax = 0.0, None
    mae_maps = []
    for model, pred in model_preds.items():
        mae = np.mean(np.abs(pred[..., channel] - tgt_c), axis=(0, 1))
        mae_maps.append(mae)
    if mae_maps:
        vmax = max(m.max() for m in mae_maps)

    for i, (model, mae) in enumerate(zip(models, mae_maps)):
        im = axes[i].imshow(mae, origin="lower", aspect="auto", vmin=vmin, vmax=vmax, cmap="hot")
        axes[i].set_title(f"{model}\nMAE({channel_name})")
        axes[i].axis("off")
        fig.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)

    # Target mean magnitude for reference
    im = axes[-1].imshow(tgt_mean, origin="lower", aspect="auto", cmap="viridis")
    axes[-1].set_title(f"Target\n|{channel_name}| mean")
    axes[-1].axis("off")
    fig.colorbar(im, ax=axes[-1], fraction=0.046, pad=0.04)

    fig.suptitle(f"Spatial Error Heatmap – channel {channel_name}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"Saved  {out_path}")


def plot_tke(
    model_preds: dict[str, np.ndarray], targets: np.ndarray, out_path: str
) -> None:
    """TKE comparison: predicted vs target TKE field (mean over N), for each model."""
    import matplotlib.pyplot as plt

    models = list(model_preds.keys())
    ncols = len(models) + 1
    fig, axes = plt.subplots(1, ncols, figsize=(4 * ncols, 4))

    # kinetic_energy shape: (N, H, W)
    tgt_tke = kinetic_energy(targets)               # (N, H, W)
    tgt_tke_mean = np.mean(tgt_tke, axis=0)         # (H, W)

    tke_maps = [np.mean(kinetic_energy(pred), axis=0) for pred in model_preds.values()]
    vmax = max(tgt_tke_mean.max(), max(m.max() for m in tke_maps)) if tke_maps else None

    for i, (model, tke_map) in enumerate(zip(models, tke_maps)):
        im = axes[i].imshow(tke_map, origin="lower", aspect="auto", vmin=0, vmax=vmax, cmap="plasma")
        axes[i].set_title(f"{model}\nTKE")
        axes[i].axis("off")
        fig.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)

    im = axes[-1].imshow(tgt_tke_mean, origin="lower", aspect="auto", vmin=0, vmax=vmax, cmap="plasma")
    axes[-1].set_title("Target\nTKE")
    axes[-1].axis("off")
    fig.colorbar(im, ax=axes[-1], fraction=0.046, pad=0.04)

    fig.suptitle("Turbulent Kinetic Energy (TKE) – mean over samples")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"Saved  {out_path}")


# ── report ─────────────────────────────────────────────────────────────────────

def write_report(all_scores: dict[str, dict], table: str, out_path: str) -> None:
    lines = [
        "# RealPDE Track 1 – Benchmark Report",
        "",
        "## Score Comparison",
        "",
        "```",
        table,
        "```",
        "",
        "## Per-Model Details",
        "",
    ]
    for model, scores in all_scores.items():
        lines.append(f"### {model}")
        lines.append("")
        for k, v in scores.items():
            lines.append(f"- **{k}**: {v:.4f}" if isinstance(v, float) else f"- **{k}**: {v}")
        lines.append("")

    lines += [
        "## Plots",
        "",
        "| File | Description |",
        "|------|-------------|",
        "| bar_subscores.png | Bar chart comparing all 5 subscores |",
        "| line_rel_l2.png | Per-sample relative L2 error over time |",
        "| heatmap_u.png | Spatial MAE heatmap – u channel |",
        "| heatmap_v.png | Spatial MAE heatmap – v channel |",
        "| tke_comparison.png | TKE field comparison (predicted vs target) |",
        "",
    ]

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Saved  {out_path}")


# ── main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    try:
        import matplotlib  # noqa: F401
    except ImportError:
        sys.exit("matplotlib is required.  Install it with:  pip install matplotlib")

    targets = load_targets()
    print(f"Targets shape: {targets.shape}")

    all_scores: dict[str, dict] = {}
    model_preds: dict[str, np.ndarray] = {}

    for model in MODELS:
        pred_path = os.path.join(RESULTS_DIR, model, "predictions.npz")
        if not os.path.exists(pred_path):
            print(f"SKIP {model}: {pred_path} not found")
            continue

        print(f"\nScoring {model} …")
        try:
            pred, _ = load_predictions(model)
            scores = score_model(model, targets)
        except Exception as exc:
            print(f"  ERROR: {exc}")
            import traceback; traceback.print_exc()
            continue

        all_scores[model] = scores
        model_preds[model] = pred
        path = save_scores(model, scores)
        print(f"  scores.json -> {path}")
        for k, v in scores.items():
            print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    if not all_scores:
        sys.exit("No model results found.  Run run_benchmark.py first.")

    print("\n" + "=" * 60)
    print("COMPARISON TABLE")
    print("=" * 60)
    table = print_table(all_scores)

    table_path = os.path.join(RESULTS_DIR, "comparison_table.txt")
    with open(table_path, "w", encoding="utf-8") as f:
        f.write(table + "\n")
    print(f"\nSaved  {table_path}")

    # ── plots ──────────────────────────────────────────────────────────────────
    print("\nGenerating plots …")
    plot_bar_subscores(all_scores, os.path.join(RESULTS_DIR, "bar_subscores.png"))

    if model_preds:
        plot_line_rel_l2(model_preds, targets, os.path.join(RESULTS_DIR, "line_rel_l2.png"))
        plot_heatmap(model_preds, targets, 0, "u", os.path.join(RESULTS_DIR, "heatmap_u.png"))
        plot_heatmap(model_preds, targets, 1, "v", os.path.join(RESULTS_DIR, "heatmap_v.png"))
        plot_tke(model_preds, targets, os.path.join(RESULTS_DIR, "tke_comparison.png"))

    report_path = os.path.join(RESULTS_DIR, "benchmark_report.md")
    write_report(all_scores, table, report_path)

    print("\nAll done.")


if __name__ == "__main__":
    main()
