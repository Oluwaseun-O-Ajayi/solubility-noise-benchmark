"""Summary tables (mean and 95% bootstrap CI over seeds) and all result figures.

    python scripts/analyze.py [results/v2]

Writes results/v2/summary.csv, results/v2/paired.csv and figures/v2/fig*.{pdf,png}.
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import norm  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "figures" / "v2"

MODELS = ["train_mean", "esol_linear", "gp_desc", "gp_tanimoto", "rf_desc", "rf_fp", "fp_net", "graph_net"]
LABEL = {"train_mean": "Training mean", "esol_linear": "ESOL equation", "gp_desc": "GP, descriptors",
         "gp_tanimoto": "GP, Tanimoto", "rf_desc": "RF, descriptors", "rf_fp": "RF, fingerprints",
         "fp_net": "Ensemble, fingerprints", "graph_net": "Ensemble, graph"}
# Okabe-Ito colours by model family; dashed lines mark fingerprint inputs.
COLOR = {"train_mean": "#9a9a9a", "esol_linear": "#4d4d4d", "gp_desc": "#0072B2", "gp_tanimoto": "#0072B2",
         "rf_desc": "#E69F00", "rf_fp": "#E69F00", "fp_net": "#009E73", "graph_net": "#CC79A7"}
STYLE = {"train_mean": ":", "esol_linear": "-", "gp_desc": "-", "gp_tanimoto": "--", "rf_desc": "-",
         "rf_fp": "--", "fp_net": "--", "graph_net": "-"}
MARKER = {"train_mean": "", "esol_linear": "s", "gp_desc": "o", "gp_tanimoto": "o", "rf_desc": "^",
          "rf_fp": "^", "fp_net": "D", "graph_net": "D"}
SPLITS = ["random", "cluster", "scaffold"]
SPLIT_LABEL = {"random": "Random split", "scaffold": "Scaffold split", "cluster": "Cluster split"}
DATASET_LABEL = {"esol": "ESOL", "aqsoldb": "AqSolDB"}

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 7, "axes.labelsize": 7, "axes.titlesize": 7.5,
    "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6.5, "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "lines.linewidth": 1.1, "lines.markersize": 3,
    "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 600, "pdf.fonttype": 42,
})


def bootstrap_ci(values, n_boot=2000, seed=0):
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    if len(values) == 0:
        return np.nan, np.nan, np.nan
    if len(values) == 1:
        return values[0], np.nan, np.nan
    rng = np.random.default_rng(seed)
    means = rng.choice(values, size=(n_boot, len(values))).mean(1)
    return values.mean(), np.percentile(means, 2.5), np.percentile(means, 97.5)


def summarize(m):
    keys = ["dataset", "split", "noise_type", "level", "model"]
    metrics = [c for c in m.columns if m[c].dtype.kind == "f" and c not in ("level",)]
    rows = []
    for key, g in m.groupby(keys):
        row = dict(zip(keys, key), n_seeds=g.seed.nunique())
        for c in metrics:
            mean, lo, hi = bootstrap_ci(g[c])
            row[c], row[c + "_lo"], row[c + "_hi"] = mean, lo, hi
        rows.append(row)
    return pd.DataFrame(rows)


def paired(m, a, b, metric):
    """Within-seed difference a - b of ``metric``, with bootstrap CI."""
    keys = ["dataset", "split", "noise_type", "level", "seed"]
    wide = m.pivot_table(index=keys, columns="model", values=metric).reset_index()
    if a not in wide or b not in wide:
        return pd.DataFrame()
    wide["diff"] = wide[a] - wide[b]
    rows = []
    for key, g in wide.groupby(keys[:-1]):
        mean, lo, hi = bootstrap_ci(g["diff"])
        rows.append(dict(zip(keys[:-1], key), a=a, b=b, metric=metric, diff=mean, lo=lo, hi=hi, n=len(g)))
    return pd.DataFrame(rows)


def save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"{name}.{ext}", bbox_inches="tight")
    plt.close(fig)


def line_panels(s, dataset, metric, ylabel, name, models=MODELS, reference=None, noise="gaussian"):
    sub = s[(s.dataset == dataset) & (s.noise_type == noise)]
    splits = [sp for sp in SPLITS if sp in set(sub.split)]
    if not splits:
        return
    fig, axes = plt.subplots(1, len(splits), figsize=(6.7, 2.25), sharey=True, squeeze=False)
    for ax, sp in zip(axes[0], splits):
        for model in models:
            d = sub[(sub.split == sp) & (sub.model == model)].sort_values("level")
            if d.empty or d[metric].isna().all():
                continue
            ax.plot(d.level, d[metric], color=COLOR[model], ls=STYLE[model], marker=MARKER[model], label=LABEL[model])
            if (metric + "_lo") in d:
                ax.fill_between(d.level, d[metric + "_lo"], d[metric + "_hi"], color=COLOR[model], alpha=0.12, lw=0)
        if reference is not None:
            ax.axhline(reference, color="black", lw=0.6, ls=(0, (4, 3)))
        ax.set_title(SPLIT_LABEL[sp])
        ax.set_xlabel("Added label noise σ (log units)" if noise == "gaussian" else "Fraction of outlier labels")
    axes[0][0].set_ylabel(ylabel)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.tight_layout(rect=(0, 0.16, 1, 1))
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.0))
    save(fig, name)


def fig_coverage_model(s, dataset="esol"):
    """Observed one-sigma coverage against the prediction 2*Phi(sqrt(RMSE^2+sigma^2)/RMSE)-1."""
    models = ["esol_linear", "gp_desc", "gp_tanimoto", "rf_desc", "rf_fp", "fp_net", "graph_net"]
    fig, axes = plt.subplots(1, 2, figsize=(5.0, 2.3))

    # panel a: every model, split and data set
    ax = axes[0]
    allm = s[(s.noise_type == "gaussian") & s.model.isin(models)]
    panel_marker = {"esol_linear": "P", "gp_desc": "o", "gp_tanimoto": "v", "rf_desc": "^",
                    "rf_fp": "s", "fp_net": "D", "graph_net": "d"}
    seen = set()
    for (ds, sp, model), d in allm.groupby(["dataset", "split", "model"]):
        d = d.sort_values("level")
        base = d[d.level == 0]
        if base.empty:
            continue
        r0 = float(base.rmse.iloc[0])
        d = d[d.level > 0]
        pred = 2 * norm.cdf(np.sqrt(r0 ** 2 + d.level ** 2) / r0) - 1
        ax.plot(pred, d.cov68_total, color=COLOR[model], ls="none", marker=panel_marker[model], ms=3,
                mfc="white" if ds == "aqsoldb" else COLOR[model], mew=0.6, alpha=0.9,
                label=LABEL[model] if model not in seen else None)
        seen.add(model)
    lo = 0.65
    ax.plot([lo, 1.0], [lo, 1.0], color="black", lw=0.6, ls=(0, (4, 3)))
    ax.set_xlabel("Predicted coverage, one-line model")
    ax.set_ylabel("Observed one-sigma coverage")
    ax.set_title("a  All models, splits, both data sets", loc="left")
    ax.legend(frameon=False, loc="upper left", fontsize=5, ncol=2, handletextpad=0.3,
              columnspacing=0.8, labelspacing=0.25)

    sub = s[(s.dataset == dataset) & (s.noise_type == "gaussian") & (s.split == "random")]
    models = ["gp_desc", "gp_tanimoto", "fp_net", "graph_net"]
    ax = axes[1]
    for model in models:
        d = sub[sub.model == model].sort_values("level")
        if d.empty:
            continue
        ax.plot(d.level, d.cov68_total, color=COLOR[model], ls="-", marker=MARKER[model])
        ax.plot(d.level, d.noisytest_cov68_total, color=COLOR[model], ls=":", marker=MARKER[model], mfc="white")
    ax.axhline(0.683, color="black", lw=0.6, ls=(0, (4, 3)))
    ax.set_xlabel("Added label noise σ (log units)")
    ax.set_ylabel("One-sigma coverage")
    ax.set_title("b  Reported (solid) vs new noisy (dotted)", loc="left")
    fig.tight_layout()
    save(fig, "fig3_coverage_model")


def fig_error_calibration(preds, dataset="esol", levels=(0.0, 1.0)):
    models = ["gp_desc", "gp_tanimoto", "rf_desc", "rf_fp", "fp_net", "graph_net"]
    splits = ["random", "cluster", "scaffold"]
    fig, axes = plt.subplots(len(levels), len(splits), figsize=(6.7, 2.2 * len(levels)), squeeze=False)
    for i, level in enumerate(levels):
        for j, sp in enumerate(splits):
            ax = axes[i][j]
            p = preds[(preds.dataset == dataset) & (preds.split == sp) & (preds.level == level)
                      & (preds.noise_type == "gaussian")]
            top = 0
            for model in models:
                d = p[p.model == model]
                if d.empty:
                    continue
                sd = np.sqrt(d.var_epi + d.var_ale).to_numpy()
                err = (d.y - d["mean"]).to_numpy()
                order = np.argsort(sd)
                bins = np.array_split(order, 8)
                rmv = [np.sqrt(np.mean(sd[b] ** 2)) for b in bins]
                rmse = [np.sqrt(np.mean(err[b] ** 2)) for b in bins]
                top = max(top, max(rmv), max(rmse))
                ax.plot(rmv, rmse, color=COLOR[model], ls=STYLE[model], marker=MARKER[model], label=LABEL[model])
            top = min(top * 1.05, 4.0) if top else 1
            ax.plot([0, top], [0, top], color="black", lw=0.6, ls=(0, (4, 3)))
            ax.set_xlim(0, top)
            ax.set_ylim(0, top)
            ax.set_title(f"{SPLIT_LABEL[sp]}, σ = {level:g}")
            if i == len(levels) - 1:
                ax.set_xlabel("Root mean predicted variance (log units)")
            if j == 0:
                ax.set_ylabel("Observed RMSE (log units)")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=6, frameon=False, bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout()
    save(fig, "fig4_error_calibration")


def fig_similarity(preds, dataset="esol", level=0.0):
    models = ["gp_desc", "gp_tanimoto", "rf_desc", "rf_fp", "fp_net", "graph_net"]
    p = preds[(preds.dataset == dataset) & (preds.level == level) & (preds.noise_type == "gaussian")]
    edges = np.array([0.0, 0.3, 0.4, 0.5, 0.6, 0.8, 1.01])
    centres = (edges[:-1] + edges[1:]) / 2
    fig, axes = plt.subplots(1, 2, figsize=(5.4, 2.3), sharey=False)
    for model in models:
        d = p[p.model == model]
        if d.empty:
            continue
        k = np.digitize(d.max_sim, edges) - 1
        err = np.abs(d.y - d["mean"]).to_numpy()
        sd = np.sqrt(d.var_epi + d.var_ale).to_numpy()
        ok = [b for b in range(len(centres)) if (k == b).sum() >= 15]
        axes[0].plot(centres[ok], [np.sqrt(np.mean(err[k == b] ** 2)) for b in ok], color=COLOR[model],
                     ls=STYLE[model], marker=MARKER[model], label=LABEL[model])
        axes[1].plot(centres[ok], [np.sqrt(np.mean(sd[k == b] ** 2)) for b in ok], color=COLOR[model],
                     ls=STYLE[model], marker=MARKER[model])
    axes[0].set_ylabel("RMSE (log units)")
    axes[1].set_ylabel("Root mean predicted σ (log units)")
    axes[0].set_title("a  Error", loc="left")
    axes[1].set_title("b  Predicted uncertainty", loc="left")
    for ax in axes:
        ax.set_xlabel("Tanimoto similarity to nearest training molecule")
    axes[0].legend(frameon=False, fontsize=6)
    fig.tight_layout()
    save(fig, "fig5_similarity")


def fig_gp_noise(m, datasets=("esol", "aqsoldb")):
    sub = m[(m.noise_type == "gaussian") & m.model.isin(["gp_desc", "gp_tanimoto"])]
    datasets = [d for d in datasets if not sub[sub.dataset == d].empty]
    if not datasets:
        return
    fig, axes = plt.subplots(1, len(datasets), figsize=(3.2 * len(datasets), 2.6), squeeze=False)
    for ax, ds in zip(axes[0], datasets):
        d = sub[sub.dataset == ds]
        base = d[d.level == 0].groupby(["model", "split"]).fitted_noise_var.mean()
        for (model, sp), g in d.groupby(["model", "split"]):
            g = g.groupby("level").fitted_noise_var.mean().reset_index()
            implied = np.sqrt(np.clip(g.fitted_noise_var - base.loc[(model, sp)], 0, None))
            ax.plot(g.level, implied, color=COLOR[model], ls=STYLE[model],
                    marker={"random": "o", "scaffold": "s", "cluster": "^"}[sp], ms=3.5,
                    label=f"{LABEL[model]}, {SPLIT_LABEL[sp].split()[0].lower()}")
        lim = d.level.max() * 1.05
        ax.plot([0, lim], [0, lim], color="black", lw=0.6, ls=(0, (4, 3)))
        ax.set_xlabel("Injected noise σ (log units)")
        ax.set_title(DATASET_LABEL.get(ds, ds))
    axes[0][0].set_ylabel("Noise recovered by GP (log units)")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.tight_layout(rect=(0, 0.2, 1, 1))
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.0))
    save(fig, "fig6_gp_noise")


def main(out="results/v2"):
    out = ROOT / out
    m = pd.concat([pd.read_csv(f) for f in sorted((out / "metrics").glob("*.csv"))], ignore_index=True)
    s = summarize(m)
    s.to_csv(out / "summary.csv", index=False)
    pairs = [("rf_desc", "rf_fp", "rmse"), ("gp_desc", "gp_tanimoto", "rmse"), ("graph_net", "fp_net", "rmse"),
             ("rf_desc", "rf_fp", "cov68_total"), ("gp_desc", "esol_linear", "rmse")]
    pd.concat([paired(m, *p) for p in pairs], ignore_index=True).to_csv(out / "paired.csv", index=False)
    preds = pd.concat([pd.read_csv(f) for f in sorted((out / "predictions").glob("*.csv.gz"))], ignore_index=True)
    for ds in sorted(m.dataset.unique()):
        line_panels(s, ds, "rmse", "Test RMSE (log units)", f"fig2_rmse_{ds}")
        line_panels(s, ds, "cov68_total", "One-sigma coverage", f"figS_coverage_{ds}", reference=0.683)
        line_panels(s, ds, "rho_total", "Spearman ρ (σ vs |error|)", f"figS_rho_{ds}",
                    models=[x for x in MODELS if x != "train_mean"], reference=0.0)
        line_panels(s, ds, "rmse", "Test RMSE (log units)", f"figS_rmse_outlier_{ds}", noise="outlier")
        line_panels(s, ds, "cov68_recal", "One-sigma coverage after variance scaling", f"figS_recal_{ds}",
                    models=[x for x in MODELS if x != "train_mean"], reference=0.683)
    fig_coverage_model(s)
    fig_error_calibration(preds)
    fig_similarity(preds)
    fig_gp_noise(m)
    print(f"{len(m)} rows; summary, paired differences and figures written")


if __name__ == "__main__":
    main(*sys.argv[1:])
