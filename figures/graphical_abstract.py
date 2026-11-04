"""Graphical abstract: 8 x 4 cm, 600 dpi.

Left: intervals look overcautious against reported labels, and are calibrated
against a new measurement of the same quality.
Right: under label shift, predicted uncertainty stops tracking error.
"""
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

matplotlib.use("Agg")

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures" / "v2"
CM = 1 / 2.54

BLUE, PINK, GREY, ORANGE = "#0072B2", "#CC79A7", "#999999", "#E69F00"
MODELS = ["gp_desc", "gp_tanimoto", "rf_desc", "rf_fp", "fp_net", "graph_net"]

plt.rcParams.update({
    "font.size": 5.6, "axes.labelsize": 5.8, "axes.titlesize": 6.2,
    "xtick.labelsize": 5.2, "ytick.labelsize": 5.2, "legend.fontsize": 5.0,
    "axes.linewidth": 0.5, "xtick.major.width": 0.5, "ytick.major.width": 0.5,
    "xtick.major.size": 2, "ytick.major.size": 2, "lines.linewidth": 1.0,
    "axes.spines.top": False, "axes.spines.right": False,
    "font.family": "DejaVu Sans",
})


def main():
    m = pd.read_csv(ROOT / "results" / "v2" / "metrics_all.csv")
    m = m[m.model.isin(MODELS)]

    fig, axes = plt.subplots(1, 2, figsize=(8 * CM, 4 * CM))

    # ---- left: what the intervals are calibrated against
    g = m[(m.dataset == "esol") & (m.noise_type == "gaussian")].groupby("level")
    clean = g.cov95_total.mean()
    noisy = g.noisytest_cov95_total.mean()
    ax = axes[0]
    ax.axhline(0.95, color=GREY, lw=0.6, ls=(0, (3, 2)), zorder=1)
    ax.plot(clean.index, clean.values, color=PINK, marker="o", ms=2.6, zorder=3,
            label="vs reported label")
    ax.plot(noisy.index, noisy.values, color=BLUE, marker="s", ms=2.6, zorder=3,
            label="vs new measurement")
    ax.set_xlabel("Added label noise σ")
    ax.set_ylabel("Coverage (nominal 0.95)")
    ax.set_ylim(0.88, 1.01)
    ax.set_yticks([0.90, 0.95, 1.00])
    ax.set_title("Noise is absorbed", pad=3)
    ax.legend(frameon=False, loc="lower right", handlelength=1.4,
              borderpad=0.1, labelspacing=0.25, handletextpad=0.4)

    # ---- right: ranking of errors by confidence, by split
    d = m[(m.noise_type == "gaussian") & (m.level == 0) & (m.dataset == "esol")]
    splits = ["random", "cluster", "scaffold"]
    fp = ["gp_tanimoto", "rf_fp", "fp_net"]
    desc = ["gp_desc", "rf_desc", "graph_net"]
    ax = axes[1]
    x = range(len(splits))
    for models, color, label in [(desc, BLUE, "descriptors / graph"),
                                 (fp, ORANGE, "fingerprints")]:
        y = [d[(d.split == s) & (d.model.isin(models))].rho_total.mean() for s in splits]
        ax.plot(x, y, color=color, marker="o", ms=2.8, label=label)
    ax.axhline(0, color="black", lw=0.6)
    ax.set_xticks(list(x))
    ax.set_xticklabels(["random", "cluster", "scaffold"])
    ax.set_xlim(-0.25, 2.25)
    ax.set_ylabel("Error ranked by σ (ρ)")
    ax.set_title("Shift inverts confidence", pad=3)
    ax.legend(frameon=False, loc="lower left", handlelength=1.4,
              borderpad=0.1, labelspacing=0.25, handletextpad=0.4)

    fig.tight_layout(pad=0.3, w_pad=1.2)
    for ext in ("png", "pdf", "tif"):
        fig.savefig(OUT / f"graphical_abstract.{ext}", dpi=600,
                    bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)
    print("graphical abstract written to", OUT)


if __name__ == "__main__":
    main()
