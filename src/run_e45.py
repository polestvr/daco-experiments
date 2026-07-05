"""E4 + E5 — backbone ablation and head-to-head vs local-density normalization.

For each frozen backbone (BEATs / EAT / PANNs) this runs the E3 grid extended
with the LDN family (clean-room reimplementation of Wilkinghoff et al.'s
local-density score normalization, arXiv:2509.10951) and DACo-on-LDN stacks,
over the 7 dev + 8 eval machines, with the label-free criterion for every
config. Outputs per-backbone CSVs plus the E5 comparison and per-backbone
Q1/Q2 summaries.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))

from daco.calibrate import DACoCalibrator, loo_knn_scores
from daco.criteria import (Config, _knn_scores, cv_balance_criterion,
                           domain_separability)
from daco.data import DEV_MACHINES, EVAL_MACHINES, list_clips
from daco.embedders import load_embedder
from daco.extract import cached_embeddings
from daco.ldn import bank_densities, ldn_scores, loo_ldn_scores
from daco.metrics import machine_metrics, official_score
from run_e3 import build_configs

AUTO_POOL = ["raw-k1", "conf-soft-m0-k1", "conf-soft-m10-k1", "conf-soft-m30-k1",
             "conf-soft-m100-k1", "conf-soft-m300-k1", "ratio-soft-k1",
             "ldn-ratio-K1", "conf-soft-m0-ldnK1"]

LDN_VARIANTS = [("ratio", 1), ("ratio", 16), ("diff", 1), ("diff", 16)]


def build_extended_configs() -> list[Config]:
    cfgs = build_configs()
    for variant, K in LDN_VARIANTS:
        cfgs.append(Config(f"ldn-{variant}-K{K}", None, None, 0, 1,
                           base=f"ldn-{variant}", density_K=K))
    for m in (0, 30):
        cfgs.append(Config(f"conf-soft-m{m}-ldnK1", "soft", "conformal",
                           m, 1, base="ldn-ratio", density_K=1))
    return cfgs


def config_scores_ext(X_train, train_domains, X_test, cfg,
                      loo_knn_by_k, ldn_cache):
    if cfg.base.startswith("ldn"):
        variant = cfg.base.split("-")[1]
        base, loo = ldn_cache[(variant, cfg.density_K)]
    else:
        base = _knn_scores(X_test, X_train, cfg.k)
        loo = loo_knn_by_k[cfg.k]
    if cfg.assignment is None:
        return base
    cal = DACoCalibrator(cfg.assignment, cfg.method,
                         prior_strength=cfg.prior_strength).fit(
        X_train, train_domains, loo)
    return cal.transform(base, cal.domain_weights(X_test))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", required=True, choices=["beats", "eat", "panns"])
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--dev-root", type=Path, required=True)
    ap.add_argument("--eval-root", type=Path, required=True)
    ap.add_argument("--gt-root", type=Path, required=True)
    ap.add_argument("--cache-dir", type=Path, required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--n-splits", type=int, default=10)
    ap.add_argument("--out-dir", type=Path,
                    default=Path(__file__).resolve().parent.parent / "results")
    args = ap.parse_args()

    embedder = load_embedder(args.backbone, args.ckpt, args.device)
    configs = build_extended_configs()

    machines = ([("dev", m, args.dev_root.expanduser(), None) for m in DEV_MACHINES]
                + [("eval", m, args.eval_root.expanduser(),
                    args.gt_root.expanduser()) for m in EVAL_MACHINES])

    rows, metrics_by, crit_by = [], {}, {}
    for split, machine, root, gt in machines:
        t0 = time.time()
        clips = list_clips(root, machine, gt_root=gt)
        train = [c for c in clips if c.split == "train"]
        test = [c for c in clips if c.split == "test"]
        X_train = cached_embeddings(embedder, train, args.cache_dir.expanduser(),
                                    f"{machine}_train", args.device,
                                    model_tag=embedder.tag)
        X_test = cached_embeddings(embedder, test, args.cache_dir.expanduser(),
                                   f"{machine}_test", args.device,
                                   model_tag=embedder.tag)
        train_domains = np.array([c.domain for c in train])
        test_domains = np.array([c.domain for c in test])
        labels = np.array([c.label for c in test])
        loo_knn_by_k = {k: loo_knn_scores(X_train, k=k) for k in (1, 2, 4)}
        ldn_cache = {}
        for variant, K in LDN_VARIANTS:
            dens = bank_densities(X_train, K)
            ldn_cache[(variant, K)] = (
                ldn_scores(X_test, X_train, K, variant, dens),
                loo_ldn_scores(X_train, K, variant, dens))

        for cfg in configs:
            scores = config_scores_ext(X_train, train_domains, X_test, cfg,
                                       loo_knn_by_k, ldn_cache)
            res = machine_metrics(scores, labels, test_domains)
            crit = cv_balance_criterion(X_train, train_domains, cfg,
                                        n_splits=args.n_splits)
            metrics_by[(split, machine, cfg.name)] = res
            crit_by[(split, machine, cfg.name)] = crit["balance_ks"]
            rows.append({"split": split, "machine": machine, "config": cfg.name,
                         **{k: round(v * 100, 2) for k, v in res.items()},
                         "balance_ks": round(crit["balance_ks"], 4)})
        print(f"{machine} ({split}): {len(configs)} configs, "
              f"{time.time() - t0:.0f}s", flush=True)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(
        args.out_dir / f"e45_{args.backbone}_per_machine.csv", index=False)

    grid_rows = []
    for cfg in configs:
        dev_pm = {m: metrics_by[("dev", m, cfg.name)] for m in DEV_MACHINES}
        eval_pm = {m: metrics_by[("eval", m, cfg.name)] for m in EVAL_MACHINES}
        grid_rows.append({
            "config": cfg.name,
            "dev_omega": official_score(dev_pm) * 100,
            "eval_omega": official_score(eval_pm) * 100,
            "crit_dev": np.mean([crit_by[("dev", m, cfg.name)]
                                 for m in DEV_MACHINES]),
            "crit_eval": np.mean([crit_by[("eval", m, cfg.name)]
                                  for m in EVAL_MACHINES]),
        })
    grid = pd.DataFrame(grid_rows)
    grid.round(4).to_csv(args.out_dir / f"e45_{args.backbone}_grid.csv",
                         index=False)

    print(f"\n===== backbone: {args.backbone} =====")
    print("\n--- Q1 correlations (n=%d configs) ---" % len(grid))
    for pred, sign, label in (("dev_omega", 1, "dev omega"),
                              ("crit_dev", -1, "criterion (dev machines)"),
                              ("crit_eval", -1, "criterion (eval train)")):
        rho, p = spearmanr(sign * grid[pred], grid["eval_omega"])
        print(f"  spearman({label:26s}, eval omega) = {rho:+.3f} (p={p:.3f})")

    print("\n--- E5 head-to-head (eval omega / dev omega) ---")
    for name in ("raw-k1", "ldn-ratio-K1", "ldn-ratio-K16", "ldn-diff-K1",
                 "ldn-diff-K16", "conf-soft-m0-k1", "conf-soft-m0-ldnK1",
                 "conf-soft-m30-ldnK1", "ratio-soft-k1"):
        row = grid[grid.config == name]
        if len(row):
            print(f"  {name:22s} eval {row.eval_omega.iloc[0]:6.2f}   "
                  f"dev {row.dev_omega.iloc[0]:6.2f}")

    print("\n--- Q2 selections (eval omega) ---")
    best_dev = grid.loc[grid.dev_omega.idxmax()]
    best_cd = grid.loc[grid.crit_dev.idxmin()]
    best_ce = grid.loc[grid.crit_eval.idxmin()]
    best_ev = grid.loc[grid.eval_omega.idxmax()]
    for tag, r in (("by dev omega", best_dev), ("by crit_dev", best_cd),
                   ("by crit_eval", best_ce), ("oracle", best_ev)):
        print(f"  {tag:13s}: {r['config']:24s} -> {r['eval_omega']:.2f}")
    chosen = {m: min(AUTO_POOL, key=lambda c: crit_by[("eval", m, c)])
              for m in EVAL_MACHINES}
    auto_pm = {m: metrics_by[("eval", m, chosen[m])] for m in EVAL_MACHINES}
    print(f"  per-machine auto: {official_score(auto_pm) * 100:.2f}  choices: "
          + ", ".join(f"{m}:{c.replace('conf-soft-', '')}" for m, c in chosen.items()))
    print(f"\nresults -> {args.out_dir}/e45_{args.backbone}_*.csv")


if __name__ == "__main__":
    main()
