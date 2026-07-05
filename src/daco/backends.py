"""Training-free anomaly-score backends on frozen embeddings.

Each backend follows fit(train) / score(test) with higher score = more anomalous.
"""
from __future__ import annotations

import numpy as np
from sklearn.covariance import LedoitWolf
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors


def l2norm(X: np.ndarray) -> np.ndarray:
    return X / np.clip(np.linalg.norm(X, axis=1, keepdims=True), 1e-9, None)


class KNNBackend:
    """Mean cosine distance to the k nearest normal training embeddings."""

    def __init__(self, k: int = 1):
        self.k = k
        self._nn: NearestNeighbors | None = None

    def fit(self, X_train: np.ndarray) -> "KNNBackend":
        self._nn = NearestNeighbors(n_neighbors=self.k, metric="cosine")
        self._nn.fit(l2norm(X_train))
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        dist, _ = self._nn.kneighbors(l2norm(X))
        return dist.mean(axis=1)


class SelectiveMahalanobis:
    """Min of source- and target-mean Mahalanobis distances under a shared
    (Ledoit-Wolf shrunk) covariance fit on the source-domain normals —
    embedding-space analogue of the official Selective Mahalanobis mode."""

    def fit(self, X_source: np.ndarray, X_target: np.ndarray) -> "SelectiveMahalanobis":
        lw = LedoitWolf().fit(X_source)
        self._precision = lw.precision_
        self._mu_s = X_source.mean(axis=0)
        self._mu_t = X_target.mean(axis=0) if len(X_target) else self._mu_s
        return self

    def _maha(self, X: np.ndarray, mu: np.ndarray) -> np.ndarray:
        d = X - mu
        return np.einsum("ij,jk,ik->i", d, self._precision, d)

    def score(self, X: np.ndarray) -> np.ndarray:
        return np.minimum(self._maha(X, self._mu_s), self._maha(X, self._mu_t))


class GMMBackend:
    """Negative log-likelihood under a diagonal-covariance GMM."""

    def __init__(self, n_components: int = 2, seed: int = 0):
        self.n_components = n_components
        self.seed = seed
        self._gmm: GaussianMixture | None = None

    def fit(self, X_train: np.ndarray) -> "GMMBackend":
        self._gmm = GaussianMixture(
            n_components=self.n_components, covariance_type="diag",
            reg_covar=1e-4, random_state=self.seed,
        ).fit(X_train)
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        return -self._gmm.score_samples(X)
