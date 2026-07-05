"""DCASE Task 2 metrics: per-domain AUC, pAUC, and the harmonic-mean score.

Official definitions (arXiv:2506.10097): the AUC of domain d uses the normal
test clips of domain d against ALL anomalous test clips (both domains); pAUC
is computed over all test clips with sklearn's max_fpr (p = 0.1), matching the
official baseline implementation; the aggregate score is the harmonic mean of
all per-machine {AUC_source, AUC_target, pAUC}.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import hmean
from sklearn.metrics import roc_auc_score

PAUC_FPR = 0.1


def machine_metrics(scores: np.ndarray, labels: np.ndarray,
                    domains: np.ndarray) -> dict[str, float]:
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)
    domains = np.asarray(domains)

    anomalies = labels == 1
    result = {}
    for name, dom in (("auc_source", "source"), ("auc_target", "target")):
        keep = anomalies | ((labels == 0) & (domains == dom))
        result[name] = roc_auc_score(labels[keep], scores[keep])
    result["pauc"] = roc_auc_score(labels, scores, max_fpr=PAUC_FPR)
    return result


def official_score(per_machine: dict[str, dict[str, float]]) -> float:
    values = [m[k] for m in per_machine.values()
              for k in ("auc_source", "auc_target", "pauc")]
    # epsilon clamp matches the official evaluator: hmean(np.maximum(...))
    return float(hmean(np.maximum(values, np.finfo(float).eps)))
