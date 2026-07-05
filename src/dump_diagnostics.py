"""Release the per-machine diagnostics quoted in the paper.

Writes results/diagnostics.csv with, per machine (BEATs backbone):
  - assignment_acc: hard latent-domain assignment accuracy on test NORMALS
    (oracle diagnostic; uses test domain labels for measurement only)
  - separability: LOO 1-NN balanced accuracy on training normals
  - pct_anom_above_tgt_max: %% of anomalies whose base score exceeds the
    largest target LOO calibration score
  - pct_anom_above_src_q90: %% of anomalies above the source LOO 90th pct

Also prints the rank-based guard alternative: exclude the bottom decile of
configurations by development omega, then argmin criterion (per backbone).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from daco.calibrate import DACoCalibrator, loo_knn_scores
from daco.criteria import _knn_scores, domain_separability
from daco.data import DEV_MACHINES, EVAL_MACHINES, list_clips
from daco.extract import cache_path_for

CACHE = Path.home() / "data/dcase2025t2/embeddings/multi"
MODEL_TAG = "BEATs_iter3_plus_AS2M"
DEV_ROOT = Path.home() / "data/dcase2025t2/dev/raw"
EVAL_ROOT = Path.home() / "data/dcase2025t2/eval/raw"
GT_ROOT = Path.home() / "tools/dcase2025_task2_evaluator"
RESULTS = Path(__file__).resolve().parent.parent / "results"


def load(machine, split):
    root = DEV_ROOT if machine in DEV_MACHINES else EVAL_ROOT
    gt = None if machine in DEV_MACHINES else GT_ROOT
    clips = [c for c in list_clips(root, machine, gt_root=gt)
             if c.split == split]
    d = np.load(cache_path_for(clips, CACHE, f"{machine}_{split}", MODEL_TAG),
                allow_pickle=True)
    return d["X"], d["domain"], d["label"].astype(int)


rows = []
for machine in DEV_MACHINES + EVAL_MACHINES:
    X_tr, dom_tr, _ = load(machine, "train")
    X_te, dom_te, y_te = load(machine, "test")
    loo = loo_knn_scores(X_tr, k=1)
    base = _knn_scores(X_te, X_tr, 1)
    w = DACoCalibrator("hard").fit(X_tr, dom_tr, loo).domain_weights(X_te)
    normal = y_te == 0
    acc = float(((w[:, 1] > 0.5)[normal]
                 == (dom_te[normal] == "target")).mean())
    anom = base[y_te == 1]
    rows.append({
        "machine": machine,
        "split": "dev" if machine in DEV_MACHINES else "eval",
        "assignment_acc_test_normals": round(acc, 3),
        "separability_train": round(domain_separability(X_tr, dom_tr), 3),
        "pct_anom_above_tgt_max": round(float(
            (anom > loo[dom_tr == "target"].max()).mean()) * 100, 1),
        "pct_anom_above_src_q90": round(float(
            (anom > np.percentile(loo[dom_tr == "source"], 90)).mean()) * 100, 1),
    })

df = pd.DataFrame(rows)
df.to_csv(RESULTS / "diagnostics.csv", index=False)
print(df.to_string(index=False))

print("\n=== rank-based guard alternative (exclude bottom decile by dev "
      "omega, then argmin crit_dev) ===")
for bb in ("beats", "eat", "panns"):
    g = pd.read_csv(RESULTS / f"e45_{bb}_grid.csv")
    floor = g.dev_omega.quantile(0.10)
    viable = g[g.dev_omega > floor]
    pick = viable.loc[viable.crit_dev.idxmin()]
    print(f"  {bb:6s}: {pick.config:22s} -> eval {pick.eval_omega:.2f} "
          f"(excluded {len(g) - len(viable)}/{len(g)})")
