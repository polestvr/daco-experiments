"""E8 --- DCASE 2026 Task 2 forward test (the pre-registered prospective point).

Scores the frozen 2026 selection (PREREGISTRATION.md) on the DCASE 2026
*evaluation* machines, now that the per-clip ground truth has been released in
the official ``nttcslab/dcase2026_task2_evaluator`` repository. No re-selection
and no re-tuning of any kind: the configuration, veto rule, criterion seed and
channel policies were all frozen on 2026-07-03, before the ground truth existed
(Zenodo snapshot v1.0.0, DOI 10.5281/zenodo.21210904).

For each channel policy (primary: channel 0; sensitivity: mean mixdown) this
runs the full 51-configuration grid over the five 2026 evaluation machines and,
joining the frozen development grid (``results/prereg2026_grid.csv``), reports
per policy:

  * eval Omega of the FROZEN pick (guarded criterion) vs the frozen dev-Omega
    pick and the fixed full-equalization default;
  * rho(criterion, eval Omega) and rho(dev Omega, eval Omega) over the grid ---
    the genuinely prospective fourth point of the 2023/2024/2025 transfer study.

Writes ``results/e8_2026_grid.csv`` and ``results/e8_2026_per_machine_<policy>.csv``.
The outcome is reported whatever it is; nothing here re-derives the frozen pick.
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

from daco.calibrate import loo_knn_scores
from daco.criteria import cv_balance_criterion
from daco.data import list_clips
from daco.embedders import load_embedder
from daco.extract import cached_embeddings
from daco.ldn import bank_densities, ldn_scores, loo_ldn_scores
from daco.metrics import machine_metrics, official_score
from run_e45 import LDN_VARIANTS, build_extended_configs, config_scores_ext

# The five novel 2026 evaluation machine types (disjoint from the 2026 dev set).
EVAL_MACHINES_2026 = [
    "BlowerDustCollector", "Sander", "SewingMachine", "ToothBrush", "ToyDrone",
]

# The fixed a-priori default (full per-domain equalization), frozen before E7.
FIXED_DEFAULT = "conf-soft-m0-k1"


def guarded_pick(dev: pd.DataFrame) -> str:
    """Bottom-decile dev-Omega viability veto, then minimum criterion --- the
    exact Stage-3 selection rule, applied to the frozen 2026 dev grid."""
    floor = dev.dev_omega.quantile(0.10)
    viable = dev[dev.dev_omega > floor]
    return viable.loc[viable.crit_dev.idxmin()].config


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="~/models/beats/BEATs_iter3_plus_AS2M.pt")
    ap.add_argument("--eval-root", type=Path,
                    default=Path.home() / "data/dcase2026t2/eval/raw",
                    help="dir with <machine>/{train,test}/*.wav for the "
                         "2026 eval machines (additional-train + eval-test zips)")
    ap.add_argument("--gt-root", type=Path,
                    default=Path.home() / "tools/dcase2026_task2_evaluator",
                    help="official 2026 evaluator checkout (ground_truth_data/, "
                         "ground_truth_domain/)")
    ap.add_argument("--dev-grid", type=Path,
                    default=Path(__file__).resolve().parent.parent
                    / "results/prereg2026_grid.csv")
    ap.add_argument("--cache-dir", type=Path,
                    default=Path.home() / "data/embeddings_2026")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--n-splits", type=int, default=10)
    ap.add_argument("--out-dir", type=Path,
                    default=Path(__file__).resolve().parent.parent / "results")
    args = ap.parse_args()

    eval_root = args.eval_root.expanduser()
    gt_root = args.gt_root.expanduser()
    machines = [m for m in EVAL_MACHINES_2026 if (eval_root / m).is_dir()]
    if not machines:
        raise FileNotFoundError(
            f"no 2026 eval machine dirs under {eval_root}; expected "
            f"{EVAL_MACHINES_2026} each with train/ and test/ subdirs")
    if machines != EVAL_MACHINES_2026:
        print(f"WARNING: only {machines} present (of {EVAL_MACHINES_2026})",
              flush=True)

    dev_all = pd.read_csv(args.dev_grid)
    embedder = load_embedder("beats", args.ckpt, args.device)
    configs = build_extended_configs()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    grid_frames = []
    for policy, channel in (("ch0", 0), ("mix", None)):
        rows, metrics_by, crit_by = [], {}, {}
        for machine in machines:
            t0 = time.time()
            clips = list_clips(eval_root, machine, gt_root=gt_root)
            train = [c for c in clips if c.split == "train"]
            test = [c for c in clips if c.split == "test"]
            if not train or not test:
                raise RuntimeError(
                    f"{machine}: {len(train)} train / {len(test)} test clips")
            if any(c.label is None for c in test):
                raise RuntimeError(f"{machine}: ground truth did not cover all "
                                   f"test clips (is --gt-root the 2026 evaluator?)")
            X_train = cached_embeddings(
                embedder, train, args.cache_dir,
                f"2026eval_{machine}_train_{policy}", args.device,
                model_tag=embedder.tag, channel=channel)
            X_test = cached_embeddings(
                embedder, test, args.cache_dir,
                f"2026eval_{machine}_test_{policy}", args.device,
                model_tag=embedder.tag, channel=channel)
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
                metrics_by[(machine, cfg.name)] = res
                crit_by[(machine, cfg.name)] = crit["balance_ks"]
                rows.append({"policy": policy, "machine": machine,
                             "config": cfg.name,
                             **{k: round(v * 100, 2) for k, v in res.items()},
                             "balance_ks": round(crit["balance_ks"], 4)})
            print(f"[{policy}] {machine}: {len(configs)} configs, "
                  f"{time.time() - t0:.0f}s", flush=True)

        pd.DataFrame(rows).to_csv(
            args.out_dir / f"e8_2026_per_machine_{policy}.csv", index=False)

        eval_grid = pd.DataFrame([{
            "policy": policy, "config": cfg.name,
            "eval_omega": official_score(
                {m: metrics_by[(m, cfg.name)] for m in machines}) * 100,
            "crit_eval": np.mean([crit_by[(m, cfg.name)] for m in machines]),
        } for cfg in configs])
        dev = dev_all[dev_all.policy == policy][["config", "dev_omega", "crit_dev"]]
        grid = eval_grid.merge(dev, on="config", how="left")
        grid_frames.append(grid)

        # --- selection rules (all read off the FROZEN dev grid; no eval peeking) ---
        g_pick = guarded_pick(dev)
        d_pick = dev.loc[dev.dev_omega.idxmax()].config
        ev = lambda name: grid.loc[grid.config == name, "eval_omega"].iloc[0]
        oracle = grid.loc[grid.eval_omega.idxmax()]
        rho_crit, p_crit = spearmanr(-grid.crit_dev, grid.eval_omega)
        rho_dev, p_dev = spearmanr(grid.dev_omega, grid.eval_omega)
        rho_ce, _ = spearmanr(-grid.crit_eval, grid.eval_omega)

        print(f"\n================ E8 DCASE 2026 forward test [{policy}] ================")
        print(f"  machines            : {machines}")
        print(f"  guarded pick (frozen): {g_pick:22s} eval Omega = {ev(g_pick):.2f}")
        print(f"  dev-Omega pick       : {d_pick:22s} eval Omega = {ev(d_pick):.2f}")
        print(f"  fixed default        : {FIXED_DEFAULT:22s} eval Omega = {ev(FIXED_DEFAULT):.2f}")
        print(f"  oracle best config   : {oracle.config:22s} eval Omega = {oracle.eval_omega:.2f}")
        print(f"  gain: guarded - dev  = {ev(g_pick) - ev(d_pick):+.2f}")
        print(f"  gain: guarded - fixed= {ev(g_pick) - ev(FIXED_DEFAULT):+.2f}")
        print(f"  rho(criterion , eval Omega) = {rho_crit:+.3f}  (p={p_crit:.3f})  [4th-year transfer point]")
        print(f"  rho(dev Omega , eval Omega) = {rho_dev:+.3f}  (p={p_dev:.3f})")
        print(f"  rho(crit_eval , eval Omega) = {rho_ce:+.3f}")
        # sanity: the guarded pick should reproduce the PREREGISTRATION frozen pick
        expected = {"ch0": "ratio-hard-k2", "mix": "conf-soft-m0-k1"}[policy]
        flag = "OK" if g_pick == expected else f"!! expected {expected}"
        print(f"  [check] guarded pick == PREREGISTRATION frozen pick: {flag}")

    pd.concat(grid_frames).round(4).to_csv(
        args.out_dir / "e8_2026_grid.csv", index=False)
    print(f"\nresults -> {args.out_dir}/e8_2026_grid.csv, e8_2026_per_machine_*.csv")


if __name__ == "__main__":
    main()
