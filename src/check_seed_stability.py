"""Seed stability of per-machine blind selection.

The CV balance criterion holds out 5 of 10 target clips per split; with a
single RNG seed the per-machine argmin could be a seed artifact. This script
recomputes the criterion under 10 different seeds, reports how often each
machine's choice changes, and the spread of the composed eval omega.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from daco.criteria import cv_balance_criterion
from daco.data import EVAL_MACHINES, list_clips
from daco.extract import cached_embeddings, load_beats
from daco.metrics import official_score
from run_e3 import AUTO_POOL, build_configs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-root", type=Path, required=True)
    ap.add_argument("--gt-root", type=Path, required=True)
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--cache-dir", type=Path, required=True)
    ap.add_argument("--per-machine-csv", type=Path, required=True)
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    cfg_by_name = {c.name: c for c in build_configs()}
    pm = pd.read_csv(args.per_machine_csv)
    pm = pm[(pm["split"] == "eval") & (pm["config"].isin(AUTO_POOL))]
    metrics = {(r.machine, r.config): {"auc_source": r.auc_source / 100,
                                       "auc_target": r.auc_target / 100,
                                       "pauc": r.pauc / 100}
               for r in pm.itertuples()}

    model = load_beats(args.ckpt.expanduser(), args.device)
    model_tag = args.ckpt.expanduser().stem

    choices = {m: [] for m in EVAL_MACHINES}
    omegas = []
    for seed in range(args.seeds):
        auto_pm = {}
        for machine in EVAL_MACHINES:
            clips = list_clips(args.eval_root.expanduser(), machine,
                               gt_root=args.gt_root.expanduser())
            train = [c for c in clips if c.split == "train"]
            X_train = cached_embeddings(model, train, args.cache_dir.expanduser(),
                                        f"{machine}_train", args.device,
                                        model_tag=model_tag)
            dom = np.array([c.domain for c in train])
            crit = {name: cv_balance_criterion(X_train, dom, cfg_by_name[name],
                                               seed=seed)["balance_ks"]
                    for name in AUTO_POOL}
            best = min(crit, key=crit.get)
            choices[machine].append(best)
            auto_pm[machine] = metrics[(machine, best)]
        omega = official_score(auto_pm) * 100
        omegas.append(omega)
        print(f"seed {seed}: omega={omega:.2f}  " + " ".join(
            f"{m}:{choices[m][-1].replace('conf-soft-', '').replace('-k1', '')}"
            for m in EVAL_MACHINES), flush=True)

    print(f"\ncomposed eval omega over {args.seeds} seeds: "
          f"mean {np.mean(omegas):.2f}  std {np.std(omegas):.2f}  "
          f"min {np.min(omegas):.2f}  max {np.max(omegas):.2f}")
    for m in EVAL_MACHINES:
        uniq, counts = np.unique(choices[m], return_counts=True)
        mode_share = counts.max() / args.seeds
        print(f"  {m:14s} choice stability {mode_share * 100:3.0f}%  "
              f"({dict(zip(uniq.tolist(), counts.tolist()))})")


if __name__ == "__main__":
    main()
