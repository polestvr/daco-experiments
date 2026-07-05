"""Replication of the transfer/selection study on DCASE 2023 and 2024 Task 2.

Same pipeline as the 2025 study (BEATs backbone, identical 51-configuration
grid, identical criterion with S=10 seed-paired splits): per year, compute
dev Omega, eval Omega, and the dev-machine criterion for every configuration,
then report rho(dev, eval), rho(-criterion, eval), and the selection rows
(dev-Omega pick, criterion-only, bottom-decile guarded, always-m0 default,
random baseline, oracle).

Usage: python src/run_replication.py --year 2023
Writes results/rep_<year>_grid.csv and rep_<year>_per_machine.csv.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))

from daco.calibrate import loo_knn_scores
from daco.criteria import cv_balance_criterion
from daco.data import list_clips
from daco.embedders import load_embedder
from daco.extract import cached_embeddings
from daco.ldn import bank_densities, ldn_scores, loo_ldn_scores
from daco.metrics import machine_metrics, official_score
from run_e45 import build_extended_configs, config_scores_ext

DEV7 = ["bearing", "fan", "gearbox", "slider", "ToyCar", "ToyTrain", "valve"]
YEARS = {
    "2023": {
        "eval_machines": ["Vacuum", "ToyTank", "ToyNscale", "ToyDrone",
                          "bandsaw", "grinder", "shaker"],
        "gt": "~/tools/dcase2023_task2_evaluator",
    },
    "2024": {
        "eval_machines": ["3DPrinter", "AirCompressor", "BrushlessMotor",
                          "HairDryer", "HoveringDrone", "RoboticArm",
                          "Scanner", "ToothBrush", "ToyCircuit"],
        "gt": "~/tools/dcase2024_task2_evaluator",
    },
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", required=True, choices=list(YEARS))
    ap.add_argument("--ckpt", default="~/models/beats/BEATs_iter3_plus_AS2M.pt")
    ap.add_argument("--cache-dir", type=Path,
                    default=Path.home() / "data/embeddings_multi_years")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--n-splits", type=int, default=10)
    ap.add_argument("--out-dir", type=Path,
                    default=Path(__file__).resolve().parent.parent / "results")
    args = ap.parse_args()

    yc = YEARS[args.year]
    dev_root = Path.home() / f"data/dcase{args.year}t2/dev/raw"
    eval_root = Path.home() / f"data/dcase{args.year}t2/eval/raw"
    gt_root = Path(yc["gt"]).expanduser()
    eval_machines = yc["eval_machines"]

    embedder = load_embedder("beats", args.ckpt, args.device)
    configs = build_extended_configs()

    machines = ([("dev", m, dev_root, None) for m in DEV7]
                + [("eval", m, eval_root, gt_root) for m in eval_machines])

    rows, metrics_by, crit_by = [], {}, {}
    for split, machine, root, gt in machines:
        t0 = time.time()
        clips = list_clips(root, machine, gt_root=gt)
        train = [c for c in clips if c.split == "train"]
        test = [c for c in clips if c.split == "test"]
        if any(c.label is None for c in test):
            raise RuntimeError(f"{machine}: unlabeled test clips")
        X_train = cached_embeddings(embedder, train, args.cache_dir,
                                    f"{args.year}_{machine}_train",
                                    args.device, model_tag=embedder.tag)
        X_test = cached_embeddings(embedder, test, args.cache_dir,
                                   f"{args.year}_{machine}_test",
                                   args.device, model_tag=embedder.tag)
        train_domains = np.array([c.domain for c in train])
        test_domains = np.array([c.domain for c in test])
        labels = np.array([c.label for c in test])
        loo_knn_by_k = {k: loo_knn_scores(X_train, k=k) for k in (1, 2, 4)}
        ldn_cache = {}
        for variant, K in [("ratio", 1), ("ratio", 16), ("diff", 1),
                           ("diff", 16)]:
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
            rows.append({"split": split, "machine": machine,
                         "config": cfg.name,
                         **{k: round(v * 100, 2) for k, v in res.items()},
                         "balance_ks": round(crit["balance_ks"], 4)})
        print(f"{machine} ({split}): {len(configs)} configs, "
              f"{time.time() - t0:.0f}s", flush=True)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(
        args.out_dir / f"rep_{args.year}_per_machine.csv", index=False)

    grid_rows = []
    for cfg in configs:
        dev_pm = {m: metrics_by[("dev", m, cfg.name)] for m in DEV7}
        ev_pm = {m: metrics_by[("eval", m, cfg.name)] for m in eval_machines}
        grid_rows.append({
            "config": cfg.name,
            "dev_omega": official_score(dev_pm) * 100,
            "eval_omega": official_score(ev_pm) * 100,
            "crit_dev": np.mean([crit_by[("dev", m, cfg.name)]
                                 for m in DEV7]),
            "crit_eval": np.mean([crit_by[("eval", m, cfg.name)]
                                  for m in eval_machines]),
        })
    grid = pd.DataFrame(grid_rows)
    grid.round(4).to_csv(args.out_dir / f"rep_{args.year}_grid.csv",
                         index=False)

    print(f"\n===== DCASE {args.year} (BEATs, n={len(grid)} configs) =====")
    for pred, sign, label in (("dev_omega", 1, "dev omega"),
                              ("crit_dev", -1, "criterion (dev machines)"),
                              ("crit_eval", -1, "criterion (eval train)")):
        rho, p = spearmanr(sign * grid[pred], grid["eval_omega"])
        print(f"  spearman({label:26s}, eval omega) = {rho:+.3f} (p={p:.4f})")

    best_dev = grid.loc[grid.dev_omega.idxmax()]
    crit_only = grid.loc[grid.crit_dev.idxmin()]
    floor = grid.dev_omega.quantile(0.10)
    viable = grid[grid.dev_omega > floor]
    guarded = viable.loc[viable.crit_dev.idxmin()]
    oracle = grid.loc[grid.eval_omega.idxmax()]
    raw = grid[grid.config == "raw-k1"].iloc[0]
    m0 = grid[grid.config == "conf-soft-m0-k1"].iloc[0]
    print(f"  raw-k1                : eval {raw.eval_omega:.2f}   "
          f"dev {raw.dev_omega:.2f}")
    print(f"  by dev omega          : {best_dev.config:22s} -> "
          f"{best_dev.eval_omega:.2f}")
    print(f"  criterion-only        : {crit_only.config:22s} -> "
          f"{crit_only.eval_omega:.2f}")
    print(f"  guarded (bottom-decile): {guarded.config:21s} -> "
          f"{guarded.eval_omega:.2f}")
    print(f"  always-m0-soft-k1     : -> {m0.eval_omega:.2f}")
    print(f"  random config         : {grid.eval_omega.mean():.2f} "
          f"+/- {grid.eval_omega.std(ddof=1):.2f}")
    print(f"  oracle                : {oracle.config:22s} -> "
          f"{oracle.eval_omega:.2f}")
    print(f"\nresults -> {args.out_dir}/rep_{args.year}_*.csv")


if __name__ == "__main__":
    main()
