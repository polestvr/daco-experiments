"""Publication figures for the DACo paper (PDF, IEEEtran column sizes).

Reads results/e2_dev.csv, e3_config_grid.csv, e3_per_machine.csv and writes
vector PDFs into --fig-dir (default: ../asd-domain-generalization-paper/figures
relative to this repo, i.e. the paper repo).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

COL_W = 3.5      # IEEE single-column width (inches)
plt.rcParams.update({
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
    "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "figure.dpi": 300, "pdf.fonttype": 42, "ps.fonttype": 42,
})


def fig_frontier(e2: pd.DataFrame, out: Path) -> None:
    """Mean source vs target AUC along the prior-strength frontier (E2)."""
    variants = ["raw", "daco-m0", "daco-m10", "daco-m30", "daco-m100",
                "daco-m300", "ratio-soft"]
    labels = {"raw": "raw kNN", "ratio-soft": "median-ratio"}
    means = e2[e2.variant.isin(variants)].groupby("variant")[
        ["auc_source", "auc_target"]].mean()
    fig, ax = plt.subplots(figsize=(COL_W, 2.6))
    m_line = [f"daco-m{m}" for m in (0, 10, 30, 100, 300)]
    xs = means.loc[m_line, "auc_source"]
    ys = means.loc[m_line, "auc_target"]
    ax.plot(xs, ys, "-o", ms=4, color="tab:blue", label="DACo ($m$ sweep)")
    offsets = {0: (4, 4), 10: (4, 4), 30: (4, 4), 100: (-8, -11), 300: (4, 4)}
    for m, x, y in zip((0, 10, 30, 100, 300), xs, ys):
        ax.annotate(f"$m$={m}", (x, y), textcoords="offset points",
                    xytext=offsets[m], fontsize=6)
    ax.plot(*means.loc["raw"], "s", ms=6, color="tab:red", label="raw kNN")
    ax.plot(*means.loc["ratio-soft"], "^", ms=6, color="tab:green",
            label="median-ratio")
    lims = [min(ax.get_xlim()[0], ax.get_ylim()[0]),
            max(ax.get_xlim()[1], ax.get_ylim()[1])]
    ax.plot(lims, lims, ":", lw=0.8, color="gray", zorder=0)
    ax.annotate(r"$\mathrm{AUC_s}=\mathrm{AUC_t}$",
                (lims[0] + 0.72 * (lims[1] - lims[0]),
                 lims[0] + 0.76 * (lims[1] - lims[0])),
                fontsize=6, color="gray", rotation=38)
    ax.set_xlabel("mean source-domain AUC (%)")
    ax.set_ylabel("mean target-domain AUC (%)")
    ax.legend(loc="lower left", frameon=False)
    fig.tight_layout()
    fig.savefig(out / "frontier.pdf")
    plt.close(fig)


def fig_transfer(grid: pd.DataFrame, out: Path) -> None:
    """dev omega vs eval omega, and -criterion vs eval omega (E3)."""
    fig, axes = plt.subplots(1, 2, figsize=(2 * COL_W, 2.6), sharey=True)
    fam = grid.config.str.extract(r"^(raw|conf-soft|conf-hard|ratio|zscore)")[0]
    colors = {"raw": "tab:red", "conf-soft": "tab:blue",
              "conf-hard": "tab:cyan", "ratio": "tab:green",
              "zscore": "tab:orange"}
    names = {"raw": "raw kNN", "conf-soft": "conformal (soft assign.)",
             "conf-hard": "conformal (hard assign.)",
             "ratio": "median-ratio", "zscore": "$z$-score"}
    for f, g in grid.groupby(fam):
        axes[0].scatter(g.dev_omega, g.eval_omega, s=14, c=colors[f],
                        label=names[f])
        axes[1].scatter(g.crit_dev, g.eval_omega, s=14, c=colors[f])
    axes[0].set_xlabel(r"development $\Omega$ (%)")
    axes[0].set_ylabel(r"evaluation $\Omega$ (%)")
    axes[0].set_title(r"(a) dev $\Omega$: $\rho_s$=+0.06", fontsize=8)
    axes[1].set_xlabel(r"balance criterion $C$ (dev machines; lower is better)")
    axes[1].invert_xaxis()
    axes[1].set_title(r"(b) label-free criterion: $\rho_s(-C)$=+0.91",
                      fontsize=8)
    axes[0].legend(frameon=False, loc="lower left", ncol=2)
    fig.tight_layout()
    fig.savefig(out / "transfer.pdf")
    plt.close(fig)


def fig_gain_vs_sep(pm: pd.DataFrame, out: Path) -> None:
    """Per-machine target-AUC gain of conf-soft-m0-k1 over raw-k1 vs
    domain separability (E3 per-machine table)."""
    raw = pm[pm.config == "raw-k1"].set_index("machine")
    cal = pm[pm.config == "conf-soft-m0-k1"].set_index("machine")
    gain = cal.auc_target - raw.auc_target
    sep = cal.separability
    is_eval = cal.split == "eval"
    fig, ax = plt.subplots(figsize=(COL_W, 2.6))
    ax.scatter(sep[~is_eval], gain[~is_eval], s=18, c="tab:blue",
               label="development")
    ax.scatter(sep[is_eval], gain[is_eval], s=18, c="tab:red", marker="^",
               label="evaluation")
    dodge = {"BandSealer": (-48, 8), "fan": (5, -11), "CoffeeGrinder": (-52, -4),
             "HomeCamera": (-56, 6), "ToyCar": (4, 3), "gearbox": (-40, -10),
             "ToyRCCar": (4, 7), "slider": (4, -11), "valve": (-30, 8),
             "bearing": (4, 1), "ScrewFeeder": (4, -10), "Polisher": (4, 3),
             "ToyPet": (4, 3), "AutoTrash": (4, 3), "ToyTrain": (4, 3)}
    for m in gain.index:
        ax.annotate(m, (sep[m], gain[m]), textcoords="offset points",
                    xytext=dodge.get(m, (3, 3)), fontsize=5.5)
    ax.axhline(0, color="gray", lw=0.8, ls=":")
    ax.set_xlabel("domain separability (LOO 1-NN balanced acc.)")
    ax.set_ylabel(r"$\Delta$ target AUC vs raw (pp)")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out / "gain_vs_separability.pdf")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path,
                    default=Path(__file__).resolve().parent.parent / "results")
    ap.add_argument("--fig-dir", type=Path, required=True)
    args = ap.parse_args()
    args.fig_dir.mkdir(parents=True, exist_ok=True)

    fig_frontier(pd.read_csv(args.results / "e2_dev.csv"), args.fig_dir)
    fig_transfer(pd.read_csv(args.results / "e3_config_grid.csv"), args.fig_dir)
    fig_gain_vs_sep(pd.read_csv(args.results / "e3_per_machine.csv"),
                    args.fig_dir)
    print(f"figures written to {args.fig_dir}")


if __name__ == "__main__":
    main()
