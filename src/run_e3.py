"""E3 — dev->eval transfer study + Stage 3 label-free model selection.

Across a grid of ~45 configurations (kNN k x assignment x calibration method x
prior strength m) this script computes, per configuration:
    dev omega   official score on the 7 development machines
    eval omega  official score on the 8 evaluation machines (post-challenge
                ground truth from the official evaluator repo)
    crit_dev    label-free CV balance criterion, averaged over dev machines
    crit_eval   the same criterion computed on the EVAL machines' own training
                normals only (legal at deployment: no labels touched)

and answers two questions:
  Q1 (transfer): does the label-free criterion predict eval omega better than
     dev omega does?  [Spearman correlations across the grid]
  Q2 (Stage 3): per-machine blind selection — for each machine pick the
     candidate config minimizing the criterion, then score the composed system.

The eval ground truth is used ONLY inside metric computation, never in
selection: every selectable quantity is derived from training normals.
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
from daco.criteria import Config, _knn_scores, cv_balance_criterion, domain_separability
from daco.data import DEV_MACHINES, EVAL_MACHINES, list_clips
from daco.extract import cached_embeddings, load_beats
from daco.metrics import machine_metrics, official_score


def build_configs() -> list[Config]:
    cfgs = []
    for k in (1, 2, 4):
        cfgs.append(Config(f"raw-k{k}", None, None, 0, k))
        for assign in ("soft", "hard"):
            for m in (0, 10, 30, 100, 300):
                cfgs.append(Config(f"conf-{assign}-m{m}-k{k}", assign,
                                   "conformal", m, k))
            cfgs.append(Config(f"ratio-{assign}-k{k}", assign, "ratio", 0, k))
            cfgs.append(Config(f"zscore-{assign}-k{k}", assign, "zscore", 0, k))
    return cfgs


AUTO_POOL = ["raw-k1", "conf-soft-m0-k1", "conf-soft-m10-k1", "conf-soft-m30-k1",
             "conf-soft-m100-k1", "conf-soft-m300-k1", "ratio-soft-k1"]


def config_scores(X_train: np.ndarray, train_domains: np.ndarray,
                  X_test: np.ndarray, cfg: Config,
                  loo_by_k: dict[int, np.ndarray]) -> np.ndarray:
    base = _knn_scores(X_test, X_train, cfg.k)
    if cfg.assignment is None:
        return base
    cal = DACoCalibrator(cfg.assignment, cfg.method,
                         prior_strength=cfg.prior_strength).fit(
        X_train, train_domains, loo_by_k[cfg.k])
    return cal.transform(base, cal.domain_weights(X_test))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev-root", type=Path, required=True)
    ap.add_argument("--eval-root", type=Path, required=True)
    ap.add_argument("--gt-root", type=Path, required=True)
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--cache-dir", type=Path, required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--n-splits", type=int, default=10)
    ap.add_argument("--out-dir", type=Path,
                    default=Path(__file__).resolve().parent.parent / "results")
    args = ap.parse_args()

    model = load_beats(args.ckpt.expanduser(), args.device)
    model_tag = args.ckpt.expanduser().stem
    configs = build_configs()

    machines = ([("dev", m, args.dev_root.expanduser(), None) for m in DEV_MACHINES]
                + [("eval", m, args.eval_root.expanduser(),
                    args.gt_root.expanduser()) for m in EVAL_MACHINES])

    per_machine_rows = []
    metrics_by = {}     # (split, machine, cfg.name) -> metrics dict
    crit_by = {}        # (split, machine, cfg.name) -> balance_ks
    sep_by = {}

    for split, machine, root, gt in machines:
        t0 = time.time()
        clips = list_clips(root, machine, gt_root=gt)
        train = [c for c in clips if c.split == "train"]
        test = [c for c in clips if c.split == "test"]
        if any(c.label is None for c in test):
            raise RuntimeError(f"{machine}: unlabeled test clips")

        X_train = cached_embeddings(model, train, args.cache_dir.expanduser(),
                                    f"{machine}_train", args.device,
                                    model_tag=model_tag)
        X_test = cached_embeddings(model, test, args.cache_dir.expanduser(),
                                   f"{machine}_test", args.device,
                                   model_tag=model_tag)
        train_domains = np.array([c.domain for c in train])
        test_domains = np.array([c.domain for c in test])
        labels = np.array([c.label for c in test])
        loo_by_k = {k: loo_knn_scores(X_train, k=k) for k in (1, 2, 4)}
        sep_by[machine] = domain_separability(X_train, train_domains)

        for cfg in configs:
            scores = config_scores(X_train, train_domains, X_test, cfg, loo_by_k)
            res = machine_metrics(scores, labels, test_domains)
            crit = cv_balance_criterion(X_train, train_domains, cfg,
                                        n_splits=args.n_splits)
            metrics_by[(split, machine, cfg.name)] = res
            crit_by[(split, machine, cfg.name)] = crit["balance_ks"]
            per_machine_rows.append({
                "split": split, "machine": machine, "config": cfg.name,
                **{k: round(v * 100, 2) for k, v in res.items()},
                "balance_ks": round(crit["balance_ks"], 4),
                "uniform_ks": round(crit["uniform_ks"], 4),
                "separability": round(sep_by[machine], 3),
            })
        print(f"{machine} ({split}): {len(configs)} configs, "
              f"{time.time() - t0:.0f}s", flush=True)

    # ---- config-level aggregation --------------------------------------
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

    args.out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(per_machine_rows).to_csv(args.out_dir / "e3_per_machine.csv",
                                          index=False)
    grid.round(4).to_csv(args.out_dir / "e3_config_grid.csv", index=False)

    # ---- Q1: what predicts eval omega? ---------------------------------
    print("\n=== Q1: predictors of eval omega across the config grid "
          f"(n={len(grid)}) ===")
    for pred, sign, label in (("dev_omega", 1, "dev omega (standard practice)"),
                              ("crit_dev", -1, "criterion on dev machines"),
                              ("crit_eval", -1, "criterion on eval train normals")):
        rho, p = spearmanr(sign * grid[pred], grid["eval_omega"])
        print(f"  spearman({label:35s} , eval omega) = {rho:+.3f}  (p={p:.4f})")

    # ---- Q2: per-machine blind selection (Stage 3) ----------------------
    print("\n=== Q2: per-machine blind selection from label-free criterion ===")
    for split, mlist in (("dev", DEV_MACHINES), ("eval", EVAL_MACHINES)):
        chosen, auto_pm = {}, {}
        for m in mlist:
            best = min(AUTO_POOL, key=lambda c: crit_by[(split, m, c)])
            chosen[m] = best
            auto_pm[m] = metrics_by[(split, m, best)]
        omega_auto = official_score(auto_pm) * 100
        raw_pm = {m: metrics_by[(split, m, "raw-k1")] for m in mlist}
        print(f"  [{split}] daco-auto omega = {omega_auto:.2f}   "
              f"raw-k1 = {official_score(raw_pm) * 100:.2f}")
        print(f"  [{split}] choices: " + ", ".join(
            f"{m}:{chosen[m].replace('conf-soft-', '').replace('-k1', '')}"
            for m in mlist))

    best_dev = grid.loc[grid["dev_omega"].idxmax()]
    best_crit = grid.loc[grid["crit_eval"].idxmin()]
    best_eval = grid.loc[grid["eval_omega"].idxmax()]
    print("\n=== fixed-config selection comparison (eval omega) ===")
    print(f"  selected by dev omega : {best_dev['config']:22s} -> "
          f"{best_dev['eval_omega']:.2f}")
    print(f"  selected by criterion : {best_crit['config']:22s} -> "
          f"{best_crit['eval_omega']:.2f}")
    print(f"  oracle best on eval   : {best_eval['config']:22s} -> "
          f"{best_eval['eval_omega']:.2f}")

    print("\n=== machine separability (LOO 1-NN balanced acc on train) ===")
    print("  " + ", ".join(f"{m}:{sep_by[m]:.2f}" for _, m, _, _ in
                           [x for x in machines]))
    print(f"\nresults written to {args.out_dir}/e3_*.csv")


if __name__ == "__main__":
    main()
