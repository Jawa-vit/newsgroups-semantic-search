"""
semantic_cache.py — A cluster-indexed semantic cache, built from scratch.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Architecture
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
A traditional cache uses exact string matching — "what is NASA?" and
"tell me about NASA" are treated as completely different keys.

Our cache uses cosine similarity of sentence embeddings:
  • Two queries q1, q2 are considered "the same" if cos(q1, q2) ≥ θ
  • θ (CACHE_SIMILARITY_THRESHOLD) is the single tunable that controls
    the precision/recall trade-off of the cache.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Why clusters make this faster
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
A flat scan of N cache entries is O(N). With K clusters and uniform
distribution, each cluster bucket holds ~N/K entries. We only search
the top-2 clusters for the incoming query (CACHE_CLUSTER_SEARCH_DEPTH).
That's O(2 × N/K) ≈ O(N/7) for K=15 — a 7× speedup for free.

The intuition: if a query is about space exploration it almost certainly
won't hit a cache entry filed under "baseball statistics". We skip those
buckets entirely.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Data structure
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  _buckets: dict[int, list[CacheEntry]]
    cluster_id → ordered list of CacheEntry objects

  CacheEntry:
    - embedding:  np.ndarray  (384-d, L2-normalised)
    - query:      str         original query text
    - result:     dict        the full API response payload
    - timestamp:  float       unix time of insertion
    - hit_count:  int         how many times this entry was returned

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
The tunable threshold — θ
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
θ=0.70: Broad paraphrase matching. "What is NASA?" matches "Tell me
        about space agencies." High cache efficiency, but may return
        slightly off-topic results. Good for FAQ bots.

θ=0.80: Balanced. Paraphrases match, topic shifts don't. "What are
        the fastest CPUs?" and "Which processors are quickest?" hit
        the cache; "What is the best GPU?" does not.

θ=0.85: Default. Near-identical phrasing required. Rewrites of the
        same sentence match. This is where you get reliable semantics
        without surprising cache hits.

θ=0.90: Strict. Only very close lexical paraphrases. Safe for domains
        where subtle wording changes carry distinct intent (legal, medical).

θ=0.95: Pedantic. Practically exact matches. Cache hit rate will be
        very low on real traffic — only useful for detecting exact
        duplicates submitted with minor typos.

The *interesting* observation: as θ rises, hit rate drops sharply at
each "paraphrase boundary". These boundaries reveal something real
about the semantic geometry of the embedding space.
"""

from __future__ import annotations

import time
import threading
from dataclasses import dataclass, field
from typing import Any
import numpy as np

from app.config import (
    CACHE_SIMILARITY_THRESHOLD,
    CACHE_CLUSTER_SEARCH_DEPTH,
    N_CLUSTERS,
)


@dataclass
class CacheEntry:
    embedding: np.ndarray      # L2-normalised query embedding
    query: str                  # original query text
    result: dict               # full response payload stored at miss time
    timestamp: float = field(default_factory=time.time)
    hit_count: int = 0


class SemanticCache:
    """
    Cluster-indexed semantic cache.

    All state lives in pure Python / NumPy — no Redis, no Memcached,
    no caching library of any kind.
    """

    def __init__(
        self,
        threshold: float = CACHE_SIMILARITY_THRESHOLD,
        search_depth: int = CACHE_CLUSTER_SEARCH_DEPTH,
        n_clusters: int = N_CLUSTERS,
    ):
        self.threshold = threshold
        self.search_depth = search_depth

        # One bucket per cluster: cluster_id → list[CacheEntry]
        self._buckets: dict[int, list[CacheEntry]] = {k: [] for k in range(n_clusters)}

        # Overflow bucket for entries whose cluster assignment isn't known yet
        # (shouldn't happen in production, but guards against race conditions
        #  if the cluster model hasn't loaded when the first query arrives)
        self._overflow: list[CacheEntry] = []

        # Statistics
        self._hit_count: int = 0
        self._miss_count: int = 0

        # Thread safety: FastAPI can serve concurrent requests
        self._lock = threading.RLock()

    # ── Lookup ──────────────────────────────────────────────────────────────────
    def lookup(
        self,
        query_embedding: np.ndarray,
        top_clusters: list[tuple[int, float]],
    ) -> tuple[CacheEntry | None, float]:
        """
        Search the cache for a semantically similar prior query.

        Args:
            query_embedding: L2-normalised (384,) float32 vector
            top_clusters:    [(cluster_id, membership), ...] sorted desc

        Returns:
            (best_entry, best_similarity) or (None, 0.0) on a miss
        """
        with self._lock:
            best_entry: CacheEntry | None = None
            best_sim: float = -1.0

            # Only search the top-K most probable clusters for this query
            clusters_to_search = [cid for cid, _ in top_clusters[:self.search_depth]]

            # Also search overflow bucket (safety net)
            candidates: list[CacheEntry] = list(self._overflow)
            for cid in clusters_to_search:
                candidates.extend(self._buckets.get(cid, []))

            for entry in candidates:
                # Cosine similarity: since both vectors are L2-normalised,
                # this is just the dot product — O(384) per entry.
                sim = float(np.dot(query_embedding, entry.embedding))
                if sim > best_sim:
                    best_sim = sim
                    best_entry = entry

            if best_sim >= self.threshold:
                # Cache hit
                best_entry.hit_count += 1
                self._hit_count += 1
                return best_entry, best_sim
            else:
                self._miss_count += 1
                return None, best_sim

    # ── Store ───────────────────────────────────────────────────────────────────
    def store(
        self,
        query: str,
        query_embedding: np.ndarray,
        result: dict,
        dominant_cluster: int | None,
    ) -> None:
        """
        Insert a new entry into the cache after a miss.

        The entry is filed under the dominant cluster so future lookups
        for semantically similar queries land in the same bucket.
        """
        entry = CacheEntry(
            embedding=query_embedding.copy(),
            query=query,
            result=result,
        )
        with self._lock:
            if dominant_cluster is not None and dominant_cluster in self._buckets:
                self._buckets[dominant_cluster].append(entry)
            else:
                self._overflow.append(entry)

    # ── Stats ───────────────────────────────────────────────────────────────────
    @property
    def total_entries(self) -> int:
        with self._lock:
            return sum(len(v) for v in self._buckets.values()) + len(self._overflow)

    @property
    def hit_count(self) -> int:
        return self._hit_count

    @property
    def miss_count(self) -> int:
        return self._miss_count

    @property
    def hit_rate(self) -> float:
        total = self._hit_count + self._miss_count
        return round(self._hit_count / total, 4) if total else 0.0

    def stats(self) -> dict:
        return {
            "total_entries": self.total_entries,
            "hit_count": self._hit_count,
            "miss_count": self._miss_count,
            "hit_rate": self.hit_rate,
            "current_threshold": self.threshold,
            "bucket_sizes": {
                str(k): len(v) for k, v in self._buckets.items() if v
            },
        }

    # ── Management ──────────────────────────────────────────────────────────────
    def flush(self) -> None:
        """Clear all entries and reset statistics."""
        with self._lock:
            for k in self._buckets:
                self._buckets[k] = []
            self._overflow.clear()
            self._hit_count = 0
            self._miss_count = 0

    def set_threshold(self, new_threshold: float) -> None:
        """Hot-swap the similarity threshold without flushing the cache."""
        if not (0.0 < new_threshold < 1.0):
            raise ValueError("Threshold must be in (0, 1)")
        self.threshold = new_threshold

    # ── Debug helpers ────────────────────────────────────────────────────────────
    def top_entries(self, n: int = 10) -> list[dict]:
        """Return the n most-hit cache entries — useful for diagnostics."""
        with self._lock:
            all_entries: list[CacheEntry] = []
            for bucket in self._buckets.values():
                all_entries.extend(bucket)
            all_entries.extend(self._overflow)
            top = sorted(all_entries, key=lambda e: e.hit_count, reverse=True)[:n]
            return [
                {
                    "query": e.query,
                    "hit_count": e.hit_count,
                    "age_seconds": round(time.time() - e.timestamp, 1),
                }
                for e in top
            ]


# Module-level singleton
semantic_cache = SemanticCache()
