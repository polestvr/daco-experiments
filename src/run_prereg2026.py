"""Pre-registration of the DCASE 2026 Task 2 forward test.

Runs the full Stage-3 selection (51-configuration grid, criterion on training
normals, bottom-decile dev-Omega veto) on the DCASE 2026 DEVELOPMENT set
only --- the evaluation machines do not exist yet, so the selected
configuration is frozen before any evaluation data. Writes
results/prereg2026_grid.csv and prints the frozen pick for both channel
policies (primary: channel 0; sensitivity: mean mixdown).
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from daco.calibrate import loo_knn_scores
from daco.criteria import cv_balance_criterion
from daco.data import list_clips
from daco.embedders import load_embedder
from daco.extract import cached_embeddings
from daco.ldn import bank_densities, ldn_scores, loo_ldn_scores
from daco.metrics import machine_metrics, official_score
from run_e45 import LDN_VARIANTS, build_extended_configs, config_scores_ext


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="~/models/beats/BEATs_iter3_plus_AS2M.pt")
    ap.add_argument("--cache-dir", type=Path,
                    default=Path.home() / "data/embeddings_2026")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--n-splits", type=int, default=10)
    ap.add_argument("--out-dir", type=Path,
                    default=Path(__file__).resolve().parent.parent / "results")
    args = ap.parse_args()

    root = Path.home() / "data/dcase2026t2/dev/raw"
    machines = sorted(p.name for p in root.iterdir() if p.is_dir())
    print(f"2026 dev machines: {machines}", flush=True)

    embedder = load_embedder("beats", args.ckpt, args.device)
    configs = build_extended_configs()

    for policy, channel in (("ch0", 0), ("mix", None)):
        rows, metrics_by, crit_by = [], {}, {}
        for machine in machines:
            t0 = time.time()
            clips = list_clips(root, machine)
            train = [c for c in clips if c.split == "train"]
            test = [c for c in clips if c.split == "test"]
            if any(c.label is None for c in test):
                raise RuntimeError(f"{machine}: unlabeled dev test clips")
            X_train = cached_embeddings(
                embedder, train, args.cache_dir,
                f"2026_{machine}_train_{policy}", args.device,
                model_tag=embedder.tag, channel=channel)
            X_test = cached_embeddings(
                embedder, test, args.cache_dir,
                f"2026_{machine}_test_{policy}", args.device,
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
                scores = config_scores_ext(X_train, train_domains, X_test,
                                           cfg, loo_knn_by_k, ldn_cache)
                res = machine_metrics(scores, labels, test_domains)
                crit = cv_balance_criterion(X_train, train_domains, cfg,
                                            n_splits=args.n_splits)
                metrics_by[(machine, cfg.name)] = res
                crit_by[(machine, cfg.name)] = crit["balance_ks"]
                rows.append({"policy": policy, "machine": machine,
                             "config": cfg.name,
                             **{k: round(v * 100, 2) for k, v in res.items()},
                             "balance_ks": round(crit["balance_ks"], 4)})
            print(f"[{policy}] {machine}: done, {time.time()-t0:.0f}s",
                  flush=True)

        grid = pd.DataFrame([{
            "policy": policy, "config": cfg.name,
            "dev_omega": official_score(
                {m: metrics_by[(m, cfg.name)] for m in machines}) * 100,
            "crit_dev": np.mean([crit_by[(m, cfg.name)] for m in machines]),
        } for cfg in configs])
        args.out_dir.mkdir(parents=True, exist_ok=True)
        mode = "w" if policy == "ch0" else "a"
        grid.round(4).to_csv(args.out_dir / "prereg2026_grid.csv",
                             mode=mode, header=(mode == "w"), index=False)
        pd.DataFrame(rows).to_csv(
            args.out_dir / f"prereg2026_per_machine_{policy}.csv", index=False)

        floor = grid.dev_omega.quantile(0.10)
        viable = grid[grid.dev_omega > floor]
        pick = viable.loc[viable.crit_dev.idxmin()]
        print(f"\n===== 2026 pre-registration [{policy}] =====")
        print(f"  dev-best        : "
              f"{grid.loc[grid.dev_omega.idxmax()].config} "
              f"({grid.dev_omega.max():.2f})")
        print(f"  FROZEN PICK     : {pick.config}  "
              f"(dev omega {pick.dev_omega:.2f}, crit {pick.crit_dev:.4f})")


if __name__ == "__main__":
    main()
