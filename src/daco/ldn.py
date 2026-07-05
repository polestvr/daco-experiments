"""Clean-room reimplementation of local-density-based anomaly score
normalization (Wilkinghoff et al.: ICASSP 2025 "Keeping the Balance" and the
TASLP extension, arXiv:2509.10951 / MERL TR2026-010), written from the
papers' equations only — no code taken from the AGPL reference implementation.

Base distance: A_cos(x, y) = 0.5 * (1 - <x, y>) on L2-normalized embeddings.

Local density of a reference sample y (leave-one-out within the reference
set, precomputed): dens_K(y) = sum over y's K nearest other reference points
of A_cos(y, y_k).       [Eq. 1 denominator; K=1 recommended in TR2026-010]

Normalized score — the min is taken over the WHOLE ratio (the neighbor is
selected after normalization; the papers' stated difference from LOF):

    ratio variant:      A(x) = min_y A_cos(x, y) / dens_K(y)
    difference variant: A(x) = min_y [A_cos(x, y) - dens_K(y)]
"""
from __future__ import annotations

import numpy as np

from .backends import l2norm


def _cos_dist(Xa: np.ndarray, Xb: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 - l2norm(Xa) @ l2norm(Xb).T)


def bank_densities(X_bank: np.ndarray, K: int) -> np.ndarray:
    """dens_K(y) for every reference sample, leave-one-out, precomputed."""
    D = _cos_dist(X_bank, X_bank)
    np.fill_diagonal(D, np.inf)
    part = np.partition(D, K - 1, axis=1)[:, :K]
    return np.maximum(part.sum(axis=1), 1e-12)


def ldn_scores(X_query: np.ndarray, X_bank: np.ndarray, K: int,
               variant: str = "ratio",
               densities: np.ndarray | None = None) -> np.ndarray:
    dens = bank_densities(X_bank, K) if densities is None else densities
    D = _cos_dist(X_query, X_bank)
    if variant == "ratio":
        return (D / dens[None, :]).min(axis=1)
    return (D - dens[None, :]).min(axis=1)


def loo_ldn_scores(X_bank: np.ndarray, K: int, variant: str = "ratio",
                   densities: np.ndarray | None = None) -> np.ndarray:
    """LOO LDN scores of the bank against itself (calibration set): each bank
    point is scored as a fresh query over the other bank points, with the
    precomputed full-bank densities."""
    dens = bank_densities(X_bank, K) if densities is None else densities
    D = _cos_dist(X_bank, X_bank)
    np.fill_diagonal(D, np.inf)
    if variant == "ratio":
        return (D / dens[None, :]).min(axis=1)
    out = D - dens[None, :]
    np.fill_diagonal(out, np.inf)
    return out.min(axis=1)
