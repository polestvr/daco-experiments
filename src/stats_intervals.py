"""Confidence intervals: Omega differences, exact permutation p,
partial correlations, and selection-baseline rows.

Outputs results/stats_intervals.json. Everything recomputable from cached
BEATs embeddings and the released grid CSVs.

  (a) Paired clip-level stratified bootstrap CI and machine-level jackknife
      for the two main evaluation-set Omega differences (BEATs):
        conf-soft-m0-k1  vs raw-k1   (59.34 - 55.83)
        conf-soft-m0-ldnK1 vs raw-k1 (61.05 - 55.83)
  (b) Family-block bootstrap CI for spearman(dev_omega, eval_omega) -- the
      symmetric counterpart of the +0.91 criterion CI.
  (c) EXACT permutation p (all 7! = 5040 permutations) for the family-mean
      rank correlation, instead of the t-approximation.
  (d) Partial correlation of -crit_dev with eval_omega controlling for
      log(m+1), assignment, and k (within the 30 conformal configs), and
      controlling family dummies + log(m+1) (all 45): does the criterion
      carry signal beyond the mechanical monotonicity in m?
  (e) Per-backbone table rows: criterion-only (no veto) pick and
      random-configuration baseline (mean +/- sd over the grid).
"""
from __future__ import annotations

import json
import sys
from itertools import permutations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import hmean, spearmanr, pearsonr
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))

from daco.calibrate import DACoCalibrator, loo_knn_scores
from daco.criteria import _knn_scores
from daco.data import DEV_MACHINES, EVAL_MACHINES, list_clips
from daco.extract import cache_path_for
from daco.ldn import bank_densities, ldn_scores, loo_ldn_scores

RESULTS = Path(__file__).resolve().parent.parent / "results"
CACHE = Path.home() / "data/dcase2025t2/embeddings/multi"
MODEL_TAG = "BEATs_iter3_plus_AS2M"
DEV_ROOT = Path.home() / "data/dcase2025t2/dev/raw"
EVAL_ROOT = Path.home() / "data/dcase2025t2/eval/raw"
GT_ROOT = Path.home() / "tools/dcase2025_task2_evaluator"

FAMILY_OF = lambda name: (
    "raw" if name.startswith("raw") else
    "conf-soft" if name.startswith("conf-soft") else
    "conf-hard" if name.startswith("conf-hard") else
    "ratio-soft" if name.startswith("ratio-soft") else
    "ratio-hard" if name.startswith("ratio-hard") else
    "zscore-soft" if name.startswith("zscore-soft") else
    "zscore-hard")

out: dict = {}


def load(machine, split):
    root = DEV_ROOT if machine in DEV_MACHINES else EVAL_ROOT
    gt = None if machine in DEV_MACHINES else GT_ROOT
    clips = [c for c in list_clips(root, machine, gt_root=gt)
             if c.split == split]
    d = np.load(cache_path_for(clips, CACHE, f"{machine}_{split}", MODEL_TAG),
                allow_pickle=True)
    return d["X"], d["domain"], d["label"].astype(int)


def machine_metrics_np(scores, labels, domains):
    anomalies = labels == 1
    res = []
    for dom in ("source", "target"):
        keep = anomalies | ((labels == 0) & (domains == dom))
        res.append(roc_auc_score(labels[keep], scores[keep]))
    res.append(roc_auc_score(labels, scores, max_fpr=0.1))
    return res  # [auc_s, auc_t, pauc]


# ---------- (a) paired CIs on evaluation-set Omega differences --------------
print("computing per-clip scores for 3 systems on 8 eval machines...",
      flush=True)
SYSTEMS = ["raw-k1", "conf-soft-m0-k1", "conf-soft-m0-ldnK1"]
scores_by = {}      # (machine, system) -> np per-clip scores
meta_by = {}        # machine -> (labels, domains)
for machine in EVAL_MACHINES:
    X_tr, dom_tr, _ = load(machine, "train")
    X_te, dom_te, y_te = load(machine, "test")
    meta_by[machine] = (y_te, dom_te)
    loo = loo_knn_scores(X_tr, k=1)
    base = _knn_scores(X_te, X_tr, 1)
    scores_by[(machine, "raw-k1")] = base
    cal = DACoCalibrator("soft", "conformal", 0).fit(X_tr, dom_tr, loo)
    scores_by[(machine, "conf-soft-m0-k1")] = cal.transform(
        base, cal.domain_weights(X_te))
    dens = bank_densities(X_tr, 1)
    base_l = ldn_scores(X_te, X_tr, 1, "ratio", dens)
    loo_l = loo_ldn_scores(X_tr, 1, "ratio", dens)
    cal_l = DACoCalibrator("soft", "conformal", 0).fit(X_tr, dom_tr, loo_l)
    scores_by[(machine, "conf-soft-m0-ldnK1")] = cal_l.transform(
        base_l, cal_l.domain_weights(X_te))


def omega_from(metric_lists):
    return float(hmean(np.concatenate(metric_lists))) * 100


def full_omega(system, machines=EVAL_MACHINES, idx_by=None):
    parts = []
    for mch in machines:
        y, d = meta_by[mch]
        s = scores_by[(mch, system)]
        if idx_by is not None:
            ii = idx_by[mch]
            y, d, s = y[ii], d[ii], s[ii]
        parts.append(machine_metrics_np(s, y, d))
    return omega_from(parts)


point = {s: full_omega(s) for s in SYSTEMS}
out["point_omegas"] = {k: round(v, 3) for k, v in point.items()}

rng = np.random.default_rng(0)
B = 1000
deltas = {("conf-soft-m0-k1", "raw-k1"): [],
          ("conf-soft-m0-ldnK1", "raw-k1"): []}
# stratified (label x domain) paired resampling within each machine
cells_by = {}
for mch in EVAL_MACHINES:
    y, d = meta_by[mch]
    cells_by[mch] = [np.flatnonzero((y == yy) & (d == dd))
                     for yy in (0, 1) for dd in ("source", "target")]
for b in range(B):
    idx_by = {}
    for mch in EVAL_MACHINES:
        idx_by[mch] = np.concatenate(
            [rng.choice(c, size=len(c), replace=True)
             for c in cells_by[mch] if len(c)])
    oms = {s: full_omega(s, idx_by=idx_by) for s in SYSTEMS}
    for a, bb in deltas:
        deltas[(a, bb)].append(oms[a] - oms[bb])
out["clip_bootstrap_CI95"] = {
    f"{a} - {b}": [round(x, 2) for x in np.percentile(v, [2.5, 97.5])]
    for (a, b), v in deltas.items()}

jack = {}
for a, b in deltas:
    ds = []
    for leave in EVAL_MACHINES:
        keep = [m for m in EVAL_MACHINES if m != leave]
        ds.append(full_omega(a, machines=keep) - full_omega(b, machines=keep))
    ds = np.array(ds)
    n = len(ds)
    se = float(np.sqrt((n - 1) / n * ((ds - ds.mean()) ** 2).sum()))
    full_d = point[a] - point[b]
    jack[f"{a} - {b}"] = {
        "delta": round(full_d, 2), "jackknife_se": round(se, 2),
        "ci95": [round(full_d - 1.96 * se, 2), round(full_d + 1.96 * se, 2)],
        "loo_range": [round(float(ds.min()), 2), round(float(ds.max()), 2)]}
out["machine_jackknife"] = jack

# ---------- (b) symmetric family-block CI for spearman(dev, eval) -----------
g = pd.read_csv(RESULTS / "e3_config_grid.csv")
g["family"] = g.config.map(FAMILY_OF)
families = sorted(g.family.unique())
rng = np.random.default_rng(1)
boot = []
for _ in range(5000):
    pick = rng.choice(families, size=len(families), replace=True)
    sub = pd.concat([g[g.family == f] for f in pick])
    if sub.dev_omega.nunique() > 2:
        boot.append(spearmanr(sub.dev_omega, sub.eval_omega).statistic)
out["dev_eval_rho_CI95"] = {
    "rho": round(spearmanr(g.dev_omega, g.eval_omega).statistic, 3),
    "family_block_CI95": [round(x, 3) for x in np.percentile(boot, [2.5, 97.5])]}

# ---------- (c) exact permutation p for family-mean correlation -------------
fam_means = g.groupby("family")[["crit_dev", "eval_omega"]].mean()
x = (-fam_means.crit_dev).values
y = fam_means.eval_omega.values
rho_obs = spearmanr(x, y).statistic
count_ge = 0
count_abs = 0
nperm = 0
for perm in permutations(range(7)):
    r = spearmanr(x, y[list(perm)]).statistic
    nperm += 1
    if r >= rho_obs - 1e-12:
        count_ge += 1
    if abs(r) >= abs(rho_obs) - 1e-12:
        count_abs += 1
out["family_mean_permutation"] = {
    "rho_obs": round(rho_obs, 4), "n_permutations": nperm,
    "p_one_sided_exact": round(count_ge / nperm, 5),
    "p_two_sided_exact": round(count_abs / nperm, 5)}

# ---------- (d) partial correlations -----------------------------------------
def residuals(v, X):
    X1 = np.column_stack([np.ones(len(v)), X])
    beta, *_ = np.linalg.lstsq(X1, v, rcond=None)
    return v - X1 @ beta


conf = g[g.family.str.startswith("conf")].copy()
conf["logm"] = np.log1p(conf.config.str.extract(r"-m(\d+)-")[0].astype(float))
conf["hard"] = (conf.family == "conf-hard").astype(float)
conf["k2"] = conf.config.str.endswith("k2").astype(float)
conf["k4"] = conf.config.str.endswith("k4").astype(float)
Xc = conf[["logm", "hard", "k2", "k4"]].values
rx = residuals((-conf.crit_dev).values, Xc)
ry = residuals(conf.eval_omega.values, Xc)
out["partial_conformal30"] = {
    "pearson": [round(v, 4) for v in pearsonr(rx, ry)],
    "spearman": [round(spearmanr(rx, ry).statistic, 4),
                 round(spearmanr(rx, ry).pvalue, 5)]}

g2 = g.copy()
g2["logm"] = np.log1p(
    g2.config.str.extract(r"-m(\d+)-")[0].fillna(0).astype(float))
fam_dum = pd.get_dummies(g2.family, drop_first=True).astype(float)
Xa = np.column_stack([g2[["logm"]].values, fam_dum.values])
rx = residuals((-g2.crit_dev).values, Xa)
ry = residuals(g2.eval_omega.values, Xa)
out["partial_full45_familydummies"] = {
    "pearson": [round(v, 4) for v in pearsonr(rx, ry)],
    "spearman": [round(spearmanr(rx, ry).statistic, 4),
                 round(spearmanr(rx, ry).pvalue, 5)]}

# ---------- (e) table rows: criterion-only and random baseline ---------------
rows = {}
for bb in ("beats", "eat", "panns"):
    gg = pd.read_csv(RESULTS / f"e45_{bb}_grid.csv")
    pick = gg.loc[gg.crit_dev.idxmin()]
    rows[bb] = {
        "criterion_only_no_veto": {"config": pick.config,
                                   "eval_omega": round(pick.eval_omega, 2)},
        "random_config": {"mean": round(gg.eval_omega.mean(), 2),
                          "sd": round(gg.eval_omega.std(ddof=1), 2)},
        "always_m0_soft_k1": {"eval_omega": round(
            gg[gg.config == "conf-soft-m0-k1"].eval_omega.iloc[0], 2)}}
out["table_rows"] = rows

RESULTS.joinpath("stats_intervals.json").write_text(json.dumps(out, indent=2))
print(json.dumps(out, indent=2))
