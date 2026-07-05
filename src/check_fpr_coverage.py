"""Empirical validation of approximate per-domain FPR equalization.

For the conformal m=0 configuration (soft assignment, k=1, BEATs) and for raw
kNN, sweep global thresholds tau over the calibrated-score range and measure
the empirical false-positive rate of TEST normals per true domain. If the
per-domain quantile maps work, the source and target FPR curves should
coincide (and, for the calibrated scores, track the nominal 1 - tau) for FPR
levels within calibration support (>= 1/(n_t+1) = 1/11).

Writes results/fpr_coverage.csv and prints a compact summary.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from daco.calibrate import DACoCalibrator, loo_knn_scores
from daco.criteria import _knn_scores
from daco.data import DEV_MACHINES, EVAL_MACHINES, list_clips
from daco.extract import cache_path_for

CACHE = Path.home() / "data/dcase2025t2/embeddings/multi"
MODEL_TAG = "BEATs_iter3_plus_AS2M"
DEV_ROOT = Path.home() / "data/dcase2025t2/dev/raw"
EVAL_ROOT = Path.home() / "data/dcase2025t2/eval/raw"
GT_ROOT = Path.home() / "tools/dcase2025_task2_evaluator"
NOMINAL = [0.10, 0.15, 0.20, 0.30, 0.50]


def load(machine: str, split: str):
    root = DEV_ROOT if machine in DEV_MACHINES else EVAL_ROOT
    gt = None if machine in DEV_MACHINES else GT_ROOT
    clips = [c for c in list_clips(root, machine, gt_root=gt)
             if c.split == split]
    f = cache_path_for(clips, CACHE, f"{machine}_{split}", MODEL_TAG)
    d = np.load(f, allow_pickle=True)
    return d["X"], d["domain"], d["label"].astype(int)


def main() -> None:
    rows = []
    for machine in DEV_MACHINES + EVAL_MACHINES:
        X_tr, dom_tr, _ = load(machine, "train")
        X_te, dom_te, y_te = load(machine, "test")
        loo = loo_knn_scores(X_tr, k=1)
        base = _knn_scores(X_te, X_tr, 1)
        cal = DACoCalibrator("soft", "conformal", 0).fit(X_tr, dom_tr, loo)
        scores = cal.transform(base, cal.domain_weights(X_te))

        normal = y_te == 0
        for nom in NOMINAL:
            # calibrated system: global threshold at nominal quantile
            tau = 1.0 - nom
            # raw system: label-free global threshold from pooled LOO quantile
            tau_raw = float(np.quantile(loo, 1.0 - nom))
            for dom in ("source", "target"):
                mask = normal & (dom_te == dom)
                rows.append({"machine": machine, "nominal_fpr": nom,
                             "domain": dom, "system": "daco-m0",
                             "empirical_fpr": round(float(
                                 (scores[mask] > tau).mean()), 4)})
                rows.append({"machine": machine, "nominal_fpr": nom,
                             "domain": dom, "system": "raw",
                             "empirical_fpr": round(float(
                                 (base[mask] > tau_raw).mean()), 4)})

    df = pd.DataFrame(rows)
    out = Path(__file__).resolve().parent.parent / "results" / "fpr_coverage.csv"
    df.to_csv(out, index=False)

    print("=== per-domain empirical FPR at label-free global thresholds "
          "(BEATs; mean over 15 machines) ===")
    agg = df.groupby(["system", "nominal_fpr", "domain"]) \
            .empirical_fpr.agg(["mean", "std"])
    print(agg.round(3).to_string())
    piv = df.pivot_table(index=["system", "machine", "nominal_fpr"],
                         columns="domain", values="empirical_fpr")
    gap = (piv.source - piv.target).abs().groupby(level=["system",
                                                          "nominal_fpr"])
    print("\nmean |FPR_source - FPR_target| per system and nominal level:")
    print(gap.mean().round(3).unstack(level="system").to_string())
    print(f"\nwritten {out}")


if __name__ == "__main__":
    main()
