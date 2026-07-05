"""Differential test: our metric implementation vs the official evaluator's.

The reference logic below mirrors dcase2025_task2_evaluator.py from
github.com/nttcslab/dcase2025_task2_evaluator: per-domain AUC filters
predictions with (y_domain == domain_idx) | (y_true != 0) — i.e., normals of
the given domain against ALL anomalies — and pAUC is
sklearn.roc_auc_score(..., max_fpr=0.1) over all clips. We compare our
daco.metrics.machine_metrics against this reference on randomized inputs.

Run: python src/test_metrics_vs_evaluator.py  (prints max abs deviation)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))

from daco.metrics import machine_metrics


def reference_metrics(scores, labels, domains):
    """Official evaluator logic, transcribed."""
    out = {}
    for name, idx in (("auc_source", "source"), ("auc_target", "target")):
        keep = (domains == idx) | (labels != 0)
        out[name] = roc_auc_score(labels[keep], scores[keep])
    out["pauc"] = roc_auc_score(labels, scores, max_fpr=0.1)
    return out


def main(trials: int = 2000, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    worst = 0.0
    for _ in range(trials):
        n = rng.integers(40, 400)
        labels = rng.integers(0, 2, n)
        if labels.min() == labels.max():
            continue
        domains = np.where(rng.random(n) < rng.uniform(0.1, 0.9),
                           "source", "target")
        # ensure both domains have at least one normal clip
        for d in ("source", "target"):
            if not ((labels == 0) & (domains == d)).any():
                i = int(np.flatnonzero(labels == 0)[0])
                domains[i] = d
        scores = rng.normal(size=n) + labels * rng.uniform(0, 2)
        ours = machine_metrics(scores, labels, domains)
        ref = reference_metrics(scores, labels, domains)
        worst = max(worst, *(abs(ours[k] - ref[k]) for k in ref))
    print(f"{trials} randomized trials: max |ours - reference| = {worst:.2e}")
    assert worst < 1e-12, "metric implementations diverge"
    print("PASS")


if __name__ == "__main__":
    main()
