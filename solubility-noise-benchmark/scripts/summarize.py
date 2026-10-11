"""Turn results/results.csv into summary tables (Markdown) and three figures."""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
MODELS = ["train_mean", "gaussian_process", "random_forest", "fingerprint_net", "graph_net"]
LABELS = {"train_mean": "Training mean (no model)", "gaussian_process": "Gaussian process (descriptors)", "random_forest": "Random forest (fingerprints)",
          "fingerprint_net": "Fingerprint network (PyTorch)", "graph_net": "Graph network (PyTorch)"}
COLORS = {"train_mean": "#c44e52", "gaussian_process": "#1b6ca8", "random_forest": "#8a8a8a", "fingerprint_net": "#d98c1f", "graph_net": "#2a9d6f"}


def mean_std(frame, value):
    grouped = frame.groupby(["split", "sigma", "model"])[value].agg(["mean", "std", "count"]).reset_index()
    return grouped


def table(stats, split, value_name):
    sigmas = sorted(stats["sigma"].unique())
    lines = ["| Model | " + " | ".join(f"noise {s:g}" for s in sigmas) + " |", "|---|" + "---|" * len(sigmas)]
    for model in MODELS:
        cells = []
        for sigma in sigmas:
            row = stats[(stats["split"] == split) & (stats["sigma"] == sigma) & (stats["model"] == model)]
            cells.append(f"{row['mean'].iloc[0]:.2f} ± {row['std'].iloc[0]:.2f}" if len(row) else "n/a")
        lines.append(f"| {LABELS[model]} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def line_figure(stats, value_label, path, reference=None):
    splits = [s for s in ["random", "scaffold"] if s in set(stats["split"])]
    fig, axes = plt.subplots(1, len(splits), figsize=(5.2 * len(splits), 4.4), sharey=True, squeeze=False)
    for ax, split in zip(axes[0], splits):
        for model in MODELS:
            rows = stats[(stats["split"] == split) & (stats["model"] == model)].sort_values("sigma")
            ax.errorbar(rows["sigma"], rows["mean"], yerr=rows["std"], marker="o", capsize=3,
                        color=COLORS[model], label=LABELS[model], linewidth=1.8)
        if reference is not None:
            ax.axhline(reference, color="black", linestyle="--", linewidth=1, label=f"ideal ({reference:g})")
        ax.set_title(f"{split} split")
        ax.set_xlabel("noise added to training labels (log units)")
        ax.grid(alpha=0.3)
    axes[0][0].set_ylabel(value_label)
    handles, labels = axes[0][-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main():
    results = pd.read_csv(ROOT / "results" / "results.csv")
    (ROOT / "figures").mkdir(exist_ok=True)
    n_seeds = results["seed"].nunique()
    out = [f"Mean ± standard deviation over {n_seeds} seeds. Test labels are always clean.\n"]
    for value, title, label, name, reference in [
        ("rmse", "Test RMSE (log10 mol/L, lower is better)", "test RMSE (log units)", "rmse_vs_noise.png", None),
        ("coverage_68", "Coverage of the one-sigma interval (ideal 0.68)", "one-sigma coverage", "coverage_vs_noise.png", 0.68),
        ("error_rank_corr", "Rank correlation between predicted uncertainty and actual error (higher is better)",
         "uncertainty-error rank correlation", "rank_corr_vs_noise.png", None),
    ]:
        stats = mean_std(results, value)
        line_figure(stats, label, ROOT / "figures" / name, reference)
        for split in ["random", "scaffold"]:
            if split in set(stats["split"]):
                out.append(f"### {title}: {split} split\n\n{table(stats, split, value)}\n")
    (ROOT / "results" / "summary.md").write_text("\n".join(out), encoding="utf-8")
    print("\n".join(out))


if __name__ == "__main__":
    sys.exit(main())
