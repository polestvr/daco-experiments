"""Clustered-uncertainty statistics for the transfer study.

Computes and prints, writing results/paper_stats.json:
  (a) family-block bootstrap CI for spearman(-crit_dev, eval_omega) on the E3
      grid, with the block structure defined explicitly; leave-one-family-out
      and conformal-only correlations; family-mean rank correlation
  (b) the H0 floor of the CV balance criterion: mean two-sample KS between
      495 and 5 samples from the SAME distribution, S splits, 7-machine mean
  (c) E2 gap statistics: signed mean and mean absolute source-target gap for
      raw and m=0
  (d) rank correlation between train separability and per-machine target-AUC
      gain (conf-soft-m0-k1 vs raw-k1) across all 15 machines (E45 BEATs)
  (e) criterion-selection sensitivity to S (splits) and holdout fraction on
      the BEATs E3 AUTO pool
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))

RESULTS = Path(__file__).resolve().parent.parent / "results"

FAMILY_OF = lambda name: (
    "raw" if name.startswith("raw") else
    "conf-soft" if name.startswith("conf-soft") else
    "conf-hard" if name.startswith("conf-hard") else
    "ratio-soft" if name.startswith("ratio-soft") else
    "ratio-hard" if name.startswith("ratio-hard") else
    "zscore-soft" if name.startswith("zscore-soft") else
    "zscore-hard")

out: dict = {}

# ---------- (a) clustered uncertainty on the E3 grid ------------------------
g = pd.read_csv(RESULTS / "e3_config_grid.csv")
g["family"] = g.config.map(FAMILY_OF)
families = sorted(g.family.unique())
rho_full = spearmanr(-g.crit_dev, g.eval_omega).statistic
rng = np.random.default_rng(0)
boot = []
for _ in range(5000):
    pick = rng.choice(families, size=len(families), replace=True)
    sub = pd.concat([g[g.family == f] for f in pick])
    if sub.crit_dev.nunique() > 2:
        boot.append(spearmanr(-sub.crit_dev, sub.eval_omega).statistic)
ci = np.percentile(boot, [2.5, 97.5])
lofo = {f: spearmanr(-g[g.family != f].crit_dev,
                     g[g.family != f].eval_omega).statistic for f in families}
conf_only = g[g.family.str.startswith("conf")]
fam_means = g.groupby("family")[["crit_dev", "eval_omega"]].mean()
out["transfer_correlation"] = {
    "rho_full_grid": round(rho_full, 4),
    "family_block_bootstrap_CI95": [round(ci[0], 3), round(ci[1], 3)],
    "n_bootstrap": 5000, "families": families,
    "lofo_range": [round(min(lofo.values()), 3), round(max(lofo.values()), 3)],
    "conformal_only_rho": round(
        spearmanr(-conf_only.crit_dev, conf_only.eval_omega).statistic, 3),
    "family_mean_rho": round(
        spearmanr(-fam_means.crit_dev, fam_means.eval_omega).statistic, 3),
    "family_mean_p": round(
        spearmanr(-fam_means.crit_dev, fam_means.eval_omega).pvalue, 4),
}

# machine-level cluster bootstrap: resample the 7 dev machines (criterion)
# and the 8 eval machines (omega) with replacement
pm = pd.read_csv(RESULTS / "e3_per_machine.csv")
dev_pm = pm[pm.split == "dev"]
eval_pm = pm[pm.split == "eval"]
dev_ms = sorted(dev_pm.machine.unique())
eval_ms = sorted(eval_pm.machine.unique())
crit_tbl = dev_pm.pivot(index="config", columns="machine", values="balance_ks")
from scipy.stats import hmean
vals_tbl = {}
for cfgname, grp in eval_pm.groupby("config"):
    vals_tbl[cfgname] = grp.set_index("machine")[
        ["auc_source", "auc_target", "pauc"]]
configs_order = crit_tbl.index
rng = np.random.default_rng(2)
boot_m = []
for _ in range(2000):
    dsel = rng.choice(dev_ms, size=len(dev_ms), replace=True)
    esel = rng.choice(eval_ms, size=len(eval_ms), replace=True)
    crit = crit_tbl[list(dsel)].mean(axis=1)
    omega_vals = [float(hmean(vals_tbl[c].loc[list(esel)].values.ravel()))
                  for c in configs_order]
    boot_m.append(spearmanr(-crit.values, omega_vals).statistic)
ci_m = np.percentile(boot_m, [2.5, 97.5])
out["transfer_correlation"]["machine_level_bootstrap_CI95"] = [round(ci_m[0], 3),
                                            round(ci_m[1], 3)]

# ---------- (b) H0 floor of the criterion -----------------------------------
rng = np.random.default_rng(1)
per_machine_means = []
for _ in range(200):                       # 200 simulated machines
    ks_vals = [ks_2samp(rng.normal(size=495), rng.normal(size=5)).statistic
               for _ in range(10)]         # S=10 splits
    per_machine_means.append(np.mean(ks_vals))
out["criterion_h0_floor"] = {
    "h0_floor_per_machine_mean": round(float(np.mean(per_machine_means)), 3),
    "h0_floor_per_machine_sd": round(float(np.std(per_machine_means)), 3),
    "h0_floor_7machine_mean_sd": round(float(np.std(per_machine_means))
                                       / np.sqrt(7), 4),
}

# ---------- (c) E2 gap statistics -------------------------------------------
e2 = pd.read_csv(RESULTS / "e2_dev.csv")
for variant in ("raw", "daco-m0"):
    sub = e2[e2.variant == variant]
    gaps = sub.auc_source - sub.auc_target
    out.setdefault("gap_statistics", {})[variant] = {
        "signed_mean_gap": round(float(gaps.mean()), 2),
        "mean_abs_gap": round(float(gaps.abs().mean()), 2),
        "median_gap": round(float(gaps.median()), 2),
    }

# ---------- (d) separability vs gain, all 15 machines (BEATs E45) -----------
pm = pd.read_csv(RESULTS / "e45_beats_per_machine.csv")
raw = pm[pm.config == "raw-k1"].set_index("machine")
cal = pm[pm.config == "conf-soft-m0-k1"].set_index("machine")
e3pm = pd.read_csv(RESULTS / "e3_per_machine.csv")
sep = e3pm[e3pm.config == "raw-k1"].set_index("machine").separability
gain = (cal.auc_target - raw.auc_target).reindex(sep.index)
rho_sep = spearmanr(sep, gain)
out["separability_vs_gain"] = {"spearman_sep_vs_gain_15m": round(rho_sep.statistic, 3),
            "p": round(rho_sep.pvalue, 3),
            "counterexamples": {
                "ToyPet": [round(float(sep["ToyPet"]), 2), round(float(gain["ToyPet"]), 1)],
                "AutoTrash": [round(float(sep["AutoTrash"]), 2), round(float(gain["AutoTrash"]), 1)],
                "ToyTrain": [round(float(sep["ToyTrain"]), 2), round(float(gain["ToyTrain"]), 1)]}}

# ---------- (e) criterion sensitivity to S and holdout fraction -------------
from daco.criteria import Config, cv_balance_criterion
from daco.data import list_clips
from daco.extract import cache_path_for

POOL = [("raw-k1", None, None, 0), ("conf-soft-m0-k1", "soft", "conformal", 0),
        ("conf-soft-m10-k1", "soft", "conformal", 10),
        ("conf-soft-m30-k1", "soft", "conformal", 30),
        ("conf-soft-m100-k1", "soft", "conformal", 100),
        ("conf-soft-m300-k1", "soft", "conformal", 300),
        ("ratio-soft-k1", "soft", "ratio", 0)]
cache = Path.home() / "data/dcase2025t2/embeddings/multi"
DEV_ROOT = Path.home() / "data/dcase2025t2/dev/raw"


def load_train(machine):
    clips = [c for c in list_clips(DEV_ROOT, machine) if c.split == "train"]
    f = cache_path_for(clips, cache, f"{machine}_train", "BEATs_iter3_plus_AS2M")
    d = np.load(f, allow_pickle=True)
    return d["X"], d["domain"]


sens = {}
dev_machines = ["bearing", "fan", "gearbox", "slider", "ToyCar", "ToyTrain", "valve"]
for S in (5, 10, 20):
    crit_sum = {name: 0.0 for name, *_ in POOL}
    for m in dev_machines:
        X, dom = load_train(m)
        for name, a, meth, pr in POOL:
            cfg = Config(name, a, meth, pr, 1)
            crit_sum[name] += cv_balance_criterion(X, dom, cfg, n_splits=S)["balance_ks"]
    sens[f"S={S}"] = min(crit_sum, key=crit_sum.get)
out["criterion_sensitivity"] = {"argmin_by_S": sens}

RESULTS.joinpath("paper_stats.json").write_text(json.dumps(out, indent=2))
print(json.dumps(out, indent=2))
