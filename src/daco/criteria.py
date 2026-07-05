"""Stage 3: label-free criteria for transfer-robust model selection.

Everything here is computed from one machine's own NORMAL TRAINING clips only —
no anomalies, no test clips, no cross-machine statistics — so it is available
under the first-shot protocol at deployment time.

The primary criterion is CV domain balance: repeatedly hold out half of each
domain's training normals, refit the score pipeline (kNN bank + calibration) on
the rest, push the held-out normals through it, and measure the KS distance
between the calibrated held-out SOURCE scores and the calibrated held-out
TARGET scores. A perfectly domain-balanced operating point gives KS ~ 0: the
two domains' normal score distributions coincide, so one global threshold
yields the same false-positive rate in both. The raw (uncalibrated) pipeline is
scored by the same criterion, which makes configurations comparable.

A secondary diagnostic, conformal uniformity, measures each domain's calibrated
held-out scores against U(0,1) (only meaningful for conformal calibration).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import ks_2samp, kstest

from .backends import l2norm
from .calibrate import DACoCalibrator, loo_knn_scores


@dataclass(frozen=True)
class Config:
    name: str
    assignment: str | None      # None = raw (no calibration)
    method: str | None
    prior_strength: float
    k: int
    base: str = "knn"           # "knn" | "ldn-ratio" | "ldn-diff"
    density_K: int = 0          # LDN density neighborhood size


def _knn_scores(X_query: np.ndarray, X_bank: np.ndarray, k: int) -> np.ndarray:
    D = 1.0 - l2norm(X_query) @ l2norm(X_bank).T
    D.sort(axis=1)
    return D[:, :k].mean(axis=1)


def _base_and_loo(X_fit: np.ndarray, X_held: np.ndarray, cfg: Config):
    if cfg.base.startswith("ldn"):
        from .ldn import bank_densities, ldn_scores, loo_ldn_scores
        variant = cfg.base.split("-")[1]
        dens = bank_densities(X_fit, cfg.density_K)
        base = ldn_scores(X_held, X_fit, cfg.density_K, variant, dens)
        loo = loo_ldn_scores(X_fit, cfg.density_K, variant, dens)
    else:
        base = _knn_scores(X_held, X_fit, cfg.k)
        loo = loo_knn_scores(X_fit, k=cfg.k)
    return base, loo


def _calibrated_holdout_scores(X_fit: np.ndarray, dom_fit: np.ndarray,
                               X_held: np.ndarray, cfg: Config) -> np.ndarray:
    base, loo = _base_and_loo(X_fit, X_held, cfg)
    if cfg.assignment is None:
        return base
    cal = DACoCalibrator(cfg.assignment, cfg.method,
                         prior_strength=cfg.prior_strength).fit(X_fit, dom_fit, loo)
    return cal.transform(base, cal.domain_weights(X_held))


def cv_balance_criterion(X_train: np.ndarray, train_domains: np.ndarray,
                         cfg: Config, n_splits: int = 10,
                         seed: int = 0) -> dict[str, float]:
    """Returns the label-free criteria for one machine and configuration.

    balance_ks:  mean over splits of KS(calibrated held-out source normals,
                 calibrated held-out target normals). Lower = more balanced.
    uniform_ks:  mean over splits and domains of KS(held-out calibrated scores,
                 U(0,1)); NaN for non-conformal configs.
    """
    assert cfg.assignment != "oracle", \
        "oracle assignment is test-labeled; not selectable"
    rng = np.random.default_rng(seed)
    src = np.flatnonzero(train_domains == "source")
    tgt = np.flatnonzero(train_domains == "target")

    balance, uniform = [], []
    for _ in range(n_splits):
        held = np.concatenate([rng.permutation(src)[:len(src) // 2],
                               rng.permutation(tgt)[:len(tgt) // 2]])
        held_mask = np.zeros(len(train_domains), dtype=bool)
        held_mask[held] = True
        X_fit, dom_fit = X_train[~held_mask], train_domains[~held_mask]
        X_held, dom_held = X_train[held_mask], train_domains[held_mask]

        s = _calibrated_holdout_scores(X_fit, dom_fit, X_held, cfg)
        s_src = s[dom_held == "source"]
        s_tgt = s[dom_held == "target"]
        balance.append(ks_2samp(s_src, s_tgt).statistic)
        if cfg.method == "conformal":
            uniform.append(np.mean([kstest(s_src, "uniform").statistic,
                                    kstest(s_tgt, "uniform").statistic]))

    return {"balance_ks": float(np.mean(balance)),
            "uniform_ks": float(np.mean(uniform)) if uniform else float("nan")}


def domain_separability(X_train: np.ndarray, train_domains: np.ndarray) -> float:
    """LOO 1-NN domain-classification balanced accuracy on training normals.

    ~0.5 means source/target are indistinguishable in embedding space (latent
    domain discovery cannot work); ~1.0 means cleanly separable.
    """
    Xn = l2norm(X_train)
    D = 1.0 - Xn @ Xn.T
    np.fill_diagonal(D, np.inf)
    nn_dom = train_domains[D.argmin(axis=1)]
    recalls = [(nn_dom[train_domains == d] == d).mean()
               for d in ("source", "target")]
    return float(np.mean(recalls))
