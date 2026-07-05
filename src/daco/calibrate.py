"""DACo Stages 1-2: latent-domain assignment + per-domain conformal calibration.

Stage 1: a test clip gets soft domain weights from its 1-NN cosine distance to
each domain's normal training embeddings (990 source / 10 target — the split is
known for TRAINING data under the DCASE protocol; only test-clip domains are
withheld).

Stage 2: each domain's leave-one-out kNN scores over the training bank form a
calibration set. The calibrated score is the domain-weighted interpolated
empirical CDF (split-conformal / quantile transform) of the base score, so a
single global threshold corresponds to the same normal-exceedance rate in
every latent domain.
"""
from __future__ import annotations

import numpy as np

from .backends import l2norm


def loo_knn_scores(X_train: np.ndarray, k: int = 1) -> np.ndarray:
    """Leave-one-out kNN scores of the training bank against itself.

    Mirrors test-time scoring (min cosine distance to the bank) while excluding
    the trivial self-match.
    """
    Xn = l2norm(X_train)
    D = 1.0 - Xn @ Xn.T
    np.fill_diagonal(D, np.inf)
    D.sort(axis=1)
    return D[:, :k].mean(axis=1)


def interp_ecdf(calib: np.ndarray):
    """Continuous empirical CDF: linear interpolation between order statistics
    at conformal plotting positions i/(n+1), linear toward 0 on the left, and
    a monotone exponential continuation on the right.

    Interpolation avoids massive ties when the target calibration set has only
    10 points (p-value granularity 1/11); the rational right tail
    q_n + (1-q_n) * x/(x+scale) keeps the raw-score ordering among anomalies
    that exceed every calibration score and never saturates to exactly 1.0 in
    float64 (an exponential tail collapses to 1.0 beyond ~37*scale, tying all
    extreme anomalies when the calibration scores are tightly clustered).

    NOTE: with n calibration points the map has data support only down to
    exceedance rate 1/(n+1); beyond the largest calibration score it is
    extrapolation, not calibration — for n=10 that covers exactly the
    pAUC(FPR<=0.1) region, so the per-domain FPR-equalization guarantee only
    holds for FPR >= 1/(n+1)."""
    s = np.sort(np.asarray(calib, dtype=float))
    n = len(s)
    q = np.arange(1, n + 1) / (n + 1)
    xs = np.concatenate([[min(0.0, s[0])], s])
    qs = np.concatenate([[0.0], q])
    scale = max(s[-1] - np.median(s), 1e-9)

    def cdf(a: np.ndarray) -> np.ndarray:
        a = np.asarray(a, dtype=float)
        p = np.interp(a, xs, qs)
        above = a > s[-1]
        if np.any(above):
            p = p.copy()
            x = a[above] - s[-1]
            p[above] = q[-1] + (1.0 - q[-1]) * (x / (x + scale))
        return p

    return cdf


class DACoCalibrator:
    """Per-latent-domain calibration of base anomaly scores.

    assignment: "soft" (softmax over 1-NN distances), "hard" (argmin distance),
                or "oracle" (true test domains — diagnostic upper bound only).
    method:     "conformal" (interpolated ECDF), "zscore" (per-domain
                standardization), or "ratio" (division by the per-domain
                median calibration score — unbounded tails).
    prior_strength m (conformal only): each domain's calibration map is shrunk
                toward the pooled map, c_k = (n_k * ecdf_k + m * ecdf_pool) /
                (n_k + m). m = 0 is pure per-domain FPR equalization; m -> inf
                recovers the raw-score ranking. m traces the source/target
                balance frontier and is the knob Stage 3 selects.
    """

    DOMAINS = ("source", "target")

    def __init__(self, assignment: str = "soft", method: str = "conformal",
                 prior_strength: float = 0.0):
        assert assignment in ("soft", "hard", "oracle")
        assert method in ("conformal", "zscore", "ratio")
        self.assignment = assignment
        self.method = method
        self.prior_strength = float(prior_strength)

    def fit(self, X_train: np.ndarray, train_domains: np.ndarray,
            loo_scores: np.ndarray) -> "DACoCalibrator":
        self._banks = {}
        self._cdfs = {}
        self._stats = {}
        self._medians = {}
        self._counts = {}
        pooled = interp_ecdf(loo_scores)
        for dom in self.DOMAINS:
            idx = np.flatnonzero(train_domains == dom)
            if len(idx) == 0:
                raise ValueError(f"no training clips for domain {dom!r}")
            self._banks[dom] = l2norm(X_train[idx])
            own = interp_ecdf(loo_scores[idx])
            n_k, m = float(len(idx)), self.prior_strength
            self._cdfs[dom] = (own if m == 0.0 else
                               (lambda a, own=own, n_k=n_k, m=m:
                                (n_k * own(a) + m * pooled(a)) / (n_k + m)))
            self._stats[dom] = (float(loo_scores[idx].mean()),
                                float(loo_scores[idx].std() + 1e-12))
            self._medians[dom] = max(float(np.median(loo_scores[idx])), 1e-12)
            self._counts[dom] = n_k
        # data-driven softmax temperature: the typical NN distance scale
        self._temp = max(float(np.median(loo_scores)), 1e-9)
        return self

    def domain_weights(self, X_test: np.ndarray,
                       oracle_domains: np.ndarray | None = None) -> np.ndarray:
        """Returns (N, 2) weights, columns ordered as self.DOMAINS."""
        if self.assignment == "oracle":
            if oracle_domains is None:
                raise ValueError("oracle assignment needs oracle_domains")
            w_t = (np.asarray(oracle_domains) == "target").astype(float)
            return np.stack([1.0 - w_t, w_t], axis=1)

        Xn = l2norm(X_test)
        d = np.stack([(1.0 - Xn @ self._banks[dom].T).min(axis=1)
                      for dom in self.DOMAINS], axis=1)      # (N, 2)
        if self.assignment == "hard":
            w_t = (d[:, 1] < d[:, 0]).astype(float)
        else:
            z = -d / self._temp
            z -= z.max(axis=1, keepdims=True)
            e = np.exp(z)
            w_t = e[:, 1] / e.sum(axis=1)
        return np.stack([1.0 - w_t, w_t], axis=1)

    def transform(self, base_scores: np.ndarray, weights: np.ndarray) -> np.ndarray:
        base_scores = np.asarray(base_scores, dtype=float)
        if self.method == "conformal":
            per_dom = np.stack([self._cdfs[dom](base_scores)
                                for dom in self.DOMAINS], axis=1)
        elif self.method == "zscore":
            per_dom = np.stack([(base_scores - mu) / sd
                                for mu, sd in (self._stats[dom] for dom in self.DOMAINS)],
                               axis=1)
        else:  # ratio
            per_dom = np.stack([base_scores / self._medians[dom]
                                for dom in self.DOMAINS], axis=1)
        return (weights * per_dom).sum(axis=1)
