"""Per-year clustered statistics for the cross-year replication study.

Family-block bootstrap CIs + family-mean exact permutation p for the
    six per-year correlations (2023/2024/2025, 51-config grids), and
    machine-level jackknife CIs for the decisive Omega deltas:
      guarded - fixed_default   per year
      guarded - dev_pick        per year
Per-machine blind selection (9-config AUTO pool, seed-0 criterion) for
    2023 and 2024 from rep_*_per_machine.csv, for the Table IV row.
Partial Spearman (criterion vs eval, controlling log m + family) on the
    2023 grid.

Writes results/stats_yearly.json.
"""
from __future__ import annotations

import json
from itertools import permutations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import hmean, pearsonr, spearmanr

RESULTS = Path(__file__).resolve().parent.parent / "results"

AUTO_POOL = ["raw-k1", "conf-soft-m0-k1", "conf-soft-m10-k1",
             "conf-soft-m30-k1", "conf-soft-m100-k1", "conf-soft-m300-k1",
             "ratio-soft-k1", "ldn-ratio-K1", "conf-soft-m0-ldnK1"]


def family_of(name: str) -> str:
    for p in ("conf-soft-m0-ldnK", "conf-soft-m30-ldnK"):
        if name.startswith(p):
            return "conf-on-ldn"
    for p in ("raw", "conf-soft", "conf-hard", "ratio-soft", "ratio-hard",
              "zscore-soft", "zscore-hard", "ldn-ratio", "ldn-diff"):
        if name.startswith(p):
            return p
    raise ValueError(name)


def block_bootstrap_ci(grid, xcol, sign, n=5000, seed=0):
    fams = sorted(grid.family.unique())
    rng = np.random.default_rng(seed)
    boot = []
    for _ in range(n):
        pick = rng.choice(fams, size=len(fams), replace=True)
        sub = pd.concat([grid[grid.family == f] for f in pick])
        if sub[xcol].nunique() > 2 and sub.eval_omega.nunique() > 2:
            boot.append(spearmanr(sign * sub[xcol], sub.eval_omega).statistic)
    return [round(float(v), 3) for v in np.percentile(boot, [2.5, 97.5])]


def family_mean_perm_p(grid, xcol, sign, mc_draws=200_000, seed=7):
    """Exact permutation p over family means for <=8 families; Monte-Carlo
    (200k draws, +1 correction) otherwise."""
    fm = grid.groupby("family")[[xcol, "eval_omega"]].mean()
    x = (sign * fm[xcol]).values
    y = fm.eval_omega.values
    k = len(fm)
    rho_obs = spearmanr(x, y).statistic
    ge = absge = tot = 0
    if k <= 8:
        for perm in permutations(range(k)):
            r = spearmanr(x, y[list(perm)]).statistic
            tot += 1
            ge += r >= rho_obs - 1e-12
            absge += abs(r) >= abs(rho_obs) - 1e-12
        exact = True
    else:
        rng = np.random.default_rng(seed)
        for _ in range(mc_draws):
            r = spearmanr(x, rng.permutation(y)).statistic
            tot += 1
            ge += r >= rho_obs - 1e-12
            absge += abs(r) >= abs(rho_obs) - 1e-12
        ge, absge, tot = ge + 1, absge + 1, tot + 1   # add-one MC correction
        exact = False
    return {"rho_family_means": round(float(rho_obs), 3), "n_families": k,
            "exact": exact,
            "p_one_sided": round(ge / tot, 5), "p_two_sided": round(absge / tot, 5)}


def omega_of(pm, config, machines):
    vals = []
    for m in machines:
        r = pm[(pm.machine == m) & (pm.config == config)].iloc[0]
        vals += [r.auc_source, r.auc_target, r.pauc]
    return float(hmean(np.maximum(vals, np.finfo(float).eps)))


def jackknife_delta(pm, cfg_a, cfg_b, machines):
    full = omega_of(pm, cfg_a, machines) - omega_of(pm, cfg_b, machines)
    ds = []
    for leave in machines:
        keep = [m for m in machines if m != leave]
        ds.append(omega_of(pm, cfg_a, keep) - omega_of(pm, cfg_b, keep))
    ds = np.array(ds)
    nn = len(ds)
    se = float(np.sqrt((nn - 1) / nn * ((ds - ds.mean()) ** 2).sum()))
    return {"delta": round(full, 2), "se": round(se, 2),
            "ci95": [round(full - 1.96 * se, 2), round(full + 1.96 * se, 2)],
            "loo_range": [round(float(ds.min()), 2), round(float(ds.max()), 2)]}


def residuals(v, X):
    X1 = np.column_stack([np.ones(len(v)), X])
    beta, *_ = np.linalg.lstsq(X1, v, rcond=None)
    return v - X1 @ beta


out = {}
GRIDS = {"2023": "rep_2023_grid.csv", "2024": "rep_2024_grid.csv",
         "2025": "e45_beats_grid.csv"}
PMS = {"2023": "rep_2023_per_machine.csv", "2024": "rep_2024_per_machine.csv",
       "2025": "e45_beats_per_machine.csv"}

for year, fn in GRIDS.items():
    g = pd.read_csv(RESULTS / fn)
    g["family"] = g.config.map(family_of)

    yr = {}
    for label, xcol, sign in (("criterion", "crit_dev", -1),
                              ("dev_omega", "dev_omega", 1)):
        rho, p_naive = spearmanr(sign * g[xcol], g.eval_omega)
        yr[label] = {
            "rho_full_grid": round(float(rho), 3),
            "p_naive": round(float(p_naive), 6),
            "family_block_CI95": block_bootstrap_ci(g, xcol, sign),
            "perm": family_mean_perm_p(g, xcol, sign)}

    # decisive deltas with machine-level jackknife
    pm = pd.read_csv(RESULTS / PMS[year])
    ev = pm[pm.split == "eval"]
    machines = sorted(ev.machine.unique())
    floor = g.dev_omega.quantile(0.10)
    viable = g[g.dev_omega > floor]
    guarded_cfg = viable.loc[viable.crit_dev.idxmin()].config
    dev_cfg = g.loc[g.dev_omega.idxmax()].config
    yr["configs"] = {"guarded": guarded_cfg, "dev_pick": dev_cfg,
                     "fixed": "conf-soft-m0-k1"}
    yr["delta_guarded_vs_fixed"] = jackknife_delta(
        ev, guarded_cfg, "conf-soft-m0-k1", machines)
    yr["delta_guarded_vs_devpick"] = jackknife_delta(
        ev, guarded_cfg, dev_cfg, machines)

    # Per-machine blind (9-config pool, seed-0 criterion from CSV)
    pool = ev[ev.config.isin(AUTO_POOL)]
    vals = []
    choices = {}
    for m, grp in pool.groupby("machine"):
        best = grp.loc[grp.balance_ks.idxmin()]
        choices[m] = best.config
        vals += [best.auc_source, best.auc_target, best.pauc]
    yr["per_machine_blind"] = {"omega": round(float(hmean(vals)), 2),
                               "choices": choices}
    out[year] = yr

# Partial correlation on the 2023 grid
g23 = pd.read_csv(RESULTS / GRIDS["2023"])
g23["family"] = g23.config.map(family_of)
g23["logm"] = np.log1p(
    g23.config.str.extract(r"-m(\d+)-")[0].fillna(0).astype(float))
fam_dum = pd.get_dummies(g23.family, drop_first=True).astype(float)
X = np.column_stack([g23[["logm"]].values, fam_dum.values])
rx = residuals((-g23.crit_dev).values, X)
ry = residuals(g23.eval_omega.values, X)
out["partial_2023"] = {
    "pearson": [round(float(v), 4) for v in pearsonr(rx, ry)],
    "spearman": [round(float(spearmanr(rx, ry).statistic), 4),
                 round(float(spearmanr(rx, ry).pvalue), 5)]}

# unrounded eval omegas for the ldn-ratio K1 vs K16 tie on 2025 BEATs
g25 = pd.read_csv(RESULTS / GRIDS["2025"])
out["ldn_ratio_tie_check"] = {
    "K1_eval": float(g25[g25.config == "ldn-ratio-K1"].eval_omega.iloc[0]),
    "K16_eval": float(g25[g25.config == "ldn-ratio-K16"].eval_omega.iloc[0])}

RESULTS.joinpath("stats_yearly.json").write_text(
    json.dumps(out, indent=2))
print(json.dumps(out, indent=2))
