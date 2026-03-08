"""
fuzzy_cluster.py — Fuzzy C-Means (FCM) clustering built entirely from scratch.

Why FCM and not k-means?
  The assignment spec is clear: "a document about gun legislation belongs to both
  politics AND firearms". Hard labels are epistemically dishonest for this corpus.
  FCM produces a *membership distribution* u_i ∈ Δ^(c-1) for every document i,
  where u_ik ∈ [0,1] and Σ_k u_ik = 1. This is the right abstraction.

Why NOT Gaussian Mixture Models (GMMs)?
  GMMs make a covariance assumption. In 384-d embedding space the covariance
  matrices are degenerate and expensive. FCM is parameter-free beyond (c, m)
  and converges reliably in this dimensionality after PCA reduction.

Why PCA before FCM?
  FCM operates on Euclidean distances. In raw 384-d space the "curse of
  dimensionality" makes all pairwise distances nearly equal — clusters become
  indistinguishable. PCA to 64 dims removes noise while preserving ~85% of the
  variance (confirmed empirically; see scripts/cluster.py).

Fuzziness exponent m=2:
  m=1   → collapses to hard k-means
  m=2   → standard FCM, crisp-ish soft assignments, stable convergence
  m→∞   → all memberships equal to 1/c (completely fuzzy, useless)
  m=2 is the well-established default in the literature.
"""

from __future__ import annotations

import json
import pickle
import numpy as np
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from app.config import (
    N_CLUSTERS, FCM_FUZZINESS, FCM_MAX_ITER, FCM_TOL,
    PCA_COMPONENTS, MODEL_DIR
)


# ── Core FCM algorithm ─────────────────────────────────────────────────────────

def _init_membership(n_samples: int, c: int, rng: np.random.Generator) -> np.ndarray:
    """
    Initialise the (n, c) membership matrix so each row sums to 1.
    Dirichlet(alpha=1) gives a uniform-over-simplex initialisation,
    which avoids the degenerate local minima that random-restart k-means
    sometimes falls into.
    """
    U = rng.dirichlet(np.ones(c), size=n_samples)
    return U.astype(np.float64)


def _update_centers(X: np.ndarray, U: np.ndarray, m: float) -> np.ndarray:
    """
    Compute cluster centres as weighted means.
    centre_k = Σ_i (u_ik^m * x_i) / Σ_i u_ik^m
    Shape: (c, d)
    """
    Um = U ** m                     # (n, c)
    centers = (Um.T @ X)            # (c, d)
    centers /= Um.sum(axis=0)[:, np.newaxis]   # normalise by sum of weights
    return centers


def _update_membership(X: np.ndarray, centers: np.ndarray, m: float) -> np.ndarray:
    """
    Update memberships via the FCM update rule:
      u_ik = 1 / Σ_j (d_ik / d_ij)^(2/(m-1))

    Distances are clamped to 1e-10 to avoid division by zero when a point
    lands exactly on a cluster centre (it should receive membership 1.0 for
    that cluster, 0.0 for all others — handled by the special-case branch).
    """
    n, c = X.shape[0], centers.shape[0]
    exp = 2.0 / (m - 1.0)

    # dist[i, k] = ||x_i - c_k||_2
    # Broadcast: X (n,d), centers (c,d) → diff (n,c,d) → dist (n,c)
    diff = X[:, np.newaxis, :] - centers[np.newaxis, :, :]  # (n, c, d)
    dist = np.linalg.norm(diff, axis=2)                      # (n, c)
    dist = np.fmax(dist, 1e-10)

    # Detect degenerate rows (a point IS a cluster centre)
    zero_mask = (dist < 1e-10)
    degenerate_rows = zero_mask.any(axis=1)

    # Standard FCM update
    ratio = dist[:, :, np.newaxis] / dist[:, np.newaxis, :]  # (n, c, c)
    U = 1.0 / (ratio ** exp).sum(axis=2)                     # (n, c)

    # Fix degenerate rows: full membership to the nearest centre
    for i in np.where(degenerate_rows)[0]:
        U[i, :] = 0.0
        U[i, zero_mask[i]] = 1.0 / zero_mask[i].sum()

    return U


def fuzzy_cmeans(
    X: np.ndarray,
    c: int,
    m: float = 2.0,
    max_iter: int = 150,
    tol: float = 1e-4,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, list[float]]:
    """
    Fuzzy C-Means clustering.

    Args:
        X:        (n_samples, n_features) float array
        c:        number of clusters
        m:        fuzziness exponent (>1)
        max_iter: maximum iterations
        tol:      convergence tolerance on the membership matrix (Frobenius norm)
        seed:     random seed for reproducibility

    Returns:
        U:        (n_samples, c) soft membership matrix
        centers:  (c, n_features) cluster centres in reduced-dim space
        history:  list of Frobenius norms (Δ U) per iteration — for diagnostics
    """
    rng = np.random.default_rng(seed)
    U = _init_membership(X.shape[0], c, rng)
    history: list[float] = []

    for iteration in range(max_iter):
        U_old = U.copy()
        centers = _update_centers(X, U, m)
        U = _update_membership(X, centers, m)
        delta = float(np.linalg.norm(U - U_old, ord="fro"))
        history.append(delta)
        if delta < tol:
            print(f"  FCM converged in {iteration + 1} iterations (Δ={delta:.2e})")
            break
    else:
        print(f"  FCM reached max_iter={max_iter} (Δ={history[-1]:.2e})")

    return U, centers, history


# ── Partition Coefficient — measures fuzziness ─────────────────────────────────

def partition_coefficient(U: np.ndarray) -> float:
    """
    PC = (1/n) Σ_i Σ_k u_ik²
    PC=1 → crisp (all memberships 0 or 1)
    PC=1/c → maximally fuzzy
    Higher is better (more confident cluster structure).
    """
    return float(np.mean(np.sum(U ** 2, axis=1)))


# ── FuzzyClustering orchestrator ───────────────────────────────────────────────

class FuzzyClustering:
    """
    Wraps PCA + FCM into a single estimator that can be saved/loaded.

    The PCA step is critical: it reduces 384-d embeddings to 64-d,
    making Euclidean distance meaningful and FCM tractable on a CPU.
    """

    def __init__(
        self,
        n_clusters: int = N_CLUSTERS,
        m: float = FCM_FUZZINESS,
        pca_components: int = PCA_COMPONENTS,
        max_iter: int = FCM_MAX_ITER,
        tol: float = FCM_TOL,
    ):
        self.n_clusters = n_clusters
        self.m = m
        self.pca_components = pca_components
        self.max_iter = max_iter
        self.tol = tol

        self.pca: PCA | None = None
        self.centers_: np.ndarray | None = None   # in PCA space
        self.membership_: np.ndarray | None = None

    def fit(self, embeddings: np.ndarray) -> "FuzzyClustering":
        """Fit PCA then FCM on a (N, 384) embedding matrix."""
        print(f"[FCM] Reducing {embeddings.shape[1]}-d → {self.pca_components}-d with PCA ...")
        self.pca = PCA(n_components=self.pca_components, random_state=42)
        X_reduced = self.pca.fit_transform(embeddings)
        explained = self.pca.explained_variance_ratio_.sum()
        print(f"[FCM] PCA explains {explained:.1%} of variance")

        print(f"[FCM] Running FCM with c={self.n_clusters}, m={self.m} ...")
        U, centers, history = fuzzy_cmeans(
            X_reduced,
            c=self.n_clusters,
            m=self.m,
            max_iter=self.max_iter,
            tol=self.tol,
        )
        self.centers_ = centers
        self.membership_ = U
        pc = partition_coefficient(U)
        print(f"[FCM] Partition coefficient = {pc:.4f} (1=crisp, {1/self.n_clusters:.4f}=random)")
        return self

    def predict_proba(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Return soft cluster memberships for new embeddings.
        Shape: (n_samples, n_clusters)
        """
        if self.pca is None or self.centers_ is None:
            raise RuntimeError("Call .fit() first")
        X_reduced = self.pca.transform(embeddings)
        U, _, _ = fuzzy_cmeans(
            X_reduced,
            c=self.n_clusters,
            m=self.m,
            max_iter=self.max_iter,
            tol=self.tol,
        )
        # Faster: use existing centres, don't re-run full FCM
        # (One-step update: fix centres, recompute membership)
        U = _update_membership(X_reduced, self.centers_, self.m)
        return U

    def dominant_cluster(self, embeddings: np.ndarray) -> np.ndarray:
        """Return the cluster with highest membership for each embedding."""
        return np.argmax(self.predict_proba(embeddings), axis=1)

    def top_k_clusters(self, embedding: np.ndarray, k: int = 2) -> list[tuple[int, float]]:
        """
        Return top-k (cluster_id, membership_score) sorted descending.
        Used by the semantic cache to decide which cluster buckets to search.
        """
        proba = self.predict_proba(embedding[np.newaxis, :])[0]
        ranked = sorted(enumerate(proba), key=lambda x: x[1], reverse=True)
        return [(int(idx), float(score)) for idx, score in ranked[:k]]

    # ── Persistence ────────────────────────────────────────────────────────────
    def save(self, path: Path | None = None) -> None:
        path = path or (MODEL_DIR / "fcm_model.pkl")
        with open(path, "wb") as f:
            pickle.dump(self, f)
        print(f"[FCM] Model saved → {path}")

    @classmethod
    def load(cls, path: Path | None = None) -> "FuzzyClustering":
        path = path or (MODEL_DIR / "fcm_model.pkl")
        with open(path, "rb") as f:
            obj = pickle.load(f)
        print(f"[FCM] Model loaded ← {path}")
        return obj

    @classmethod
    def is_trained(cls, path: Path | None = None) -> bool:
        path = path or (MODEL_DIR / "fcm_model.pkl")
        return path.exists()


# Module-level singleton — lazily loaded on first use
_cluster_model: FuzzyClustering | None = None


def get_cluster_model() -> FuzzyClustering:
    global _cluster_model
    if _cluster_model is None:
        if FuzzyClustering.is_trained():
            _cluster_model = FuzzyClustering.load()
        else:
            raise RuntimeError(
                "Cluster model not found. Run `python scripts/cluster.py` first."
            )
    return _cluster_model
