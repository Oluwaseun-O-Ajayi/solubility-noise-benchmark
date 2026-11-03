"""Figure 1: study design schematic."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

OUT = Path(__file__).resolve().parents[2] / "figures" / "v2"
INK, MUTED, LINE = "#1d2433", "#5a6475", "#b9c0cc"
FILLS = ["#e8eef7", "#e9f3ee", "#fbefe3", "#efeaf6", "#f4f4f4"]
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7.5})


def box(ax, x, y, w, h, title, lines, fill):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.008,rounding_size=0.015",
                                fc=fill, ec=LINE, lw=0.8))
    ax.text(x + w / 2, y + h - 0.035, title, ha="center", va="top", fontsize=8.2, fontweight="bold", color=INK)
    for i, line in enumerate(lines):
        ax.text(x + w / 2, y + h - 0.105 - i * 0.062, line, ha="center", va="top", fontsize=6.6, color=MUTED)


def arrow(ax, x0, x1, y):
    ax.add_patch(FancyArrowPatch((x0, y), (x1, y), arrowstyle="-|>", mutation_scale=9, lw=0.9, color=INK))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.7, 2.2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 0.62)
    ax.axis("off")
    w, h, y, gap = 0.158, 0.44, 0.08, 0.045
    xs = [0.015 + i * (w + gap) for i in range(5)]
    panels = [
        ("Data", ["ESOL", "1,128 compounds", "AqSolDB", "8,489 compounds", "replicate SD"]),
        ("Split", ["80 / 10 / 10", "Random", "Scaffold:", "structure + label", "Cluster: structure"]),
        ("Label noise", ["train + validation", "Gaussian σ 0 to 2", "Outliers 5 to 20%", "of ±1 to 3 log", "test labels clean"]),
        ("Models", ["2 baselines", "2 Gaussian proc.", "2 random forests", "2 deep ensembles", "μ, σ²epi, σ²ale"]),
        ("Scoring", ["vs reported", "vs new noisy", "accuracy", "calibration", "ranking, recal."]),
    ]
    for (title, lines), x, fill in zip(panels, xs, FILLS):
        box(ax, x, y, w, h, title, lines, fill)
    for i in range(4):
        arrow(ax, xs[i] + w + 0.004, xs[i + 1] - 0.004, y + h / 2)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"fig1_design.{ext}", dpi=600, bbox_inches="tight")
    print("saved", OUT / "fig1_design.pdf")


if __name__ == "__main__":
    main()
