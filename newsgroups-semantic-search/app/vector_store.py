"""
vector_store.py — ChromaDB wrapper for the newsgroups corpus.

Design decisions:
  • Cosine distance space: embeddings are L2-normalised, so cosine ≡ dot-product.
    Chromadb's cosine space natively handles this without extra normalisation.
  • Persistent storage: embeddings are expensive to recompute (~5 min on CPU
    for 18 k docs). Writing to disk means the ingestion script runs once.
  • Metadata schema: every document carries {source_category, doc_id, cluster_id,
    cluster_memberships_json} so we can filter by cluster at query time.
"""

from __future__ import annotations
import json
import numpy as np
import chromadb
from chromadb.config import Settings
from app.config import CHROMA_COLLECTION, DB_DIR, TOP_K_RESULTS


class VectorStore:
    """Thin wrapper around a ChromaDB collection."""

    def __init__(self):
        self._client = chromadb.PersistentClient(
            path=str(DB_DIR),
            settings=Settings(anonymized_telemetry=False),
        )
        self._col = self._client.get_or_create_collection(
            name=CHROMA_COLLECTION,
            metadata={"hnsw:space": "cosine"},  # cosine similarity
        )

    # ── Write ──────────────────────────────────────────────────────────────────
    def add_documents(
        self,
        ids: list[str],
        embeddings: np.ndarray,
        documents: list[str],
        metadatas: list[dict],
    ) -> None:
        """Batch upsert documents into the collection."""
        self._col.upsert(
            ids=ids,
            embeddings=embeddings.tolist(),
            documents=documents,
            metadatas=metadatas,
        )

    # ── Read ───────────────────────────────────────────────────────────────────
    def query(
        self,
        query_embedding: np.ndarray,
        n_results: int = TOP_K_RESULTS,
        where: dict | None = None,
    ) -> list[dict]:
        """
        Return the top-n most similar documents.

        Args:
            query_embedding: L2-normalised float32 vector
            n_results:       how many results to return
            where:           optional ChromaDB metadata filter
                             e.g. {"source_category": "sci.space"}

        Returns:
            List of dicts with keys: id, document, metadata, distance, similarity
        """
        kwargs = dict(
            query_embeddings=[query_embedding.tolist()],
            n_results=n_results,
            include=["documents", "metadatas", "distances"],
        )
        if where:
            kwargs["where"] = where

        res = self._col.query(**kwargs)

        results = []
        for doc, meta, dist, doc_id in zip(
            res["documents"][0],
            res["metadatas"][0],
            res["distances"][0],
            res["ids"][0],
        ):
            results.append(
                {
                    "id": doc_id,
                    "document": doc,
                    "metadata": meta,
                    # ChromaDB cosine distance ∈ [0, 2]; convert to similarity ∈ [-1, 1]
                    "distance": dist,
                    "similarity": 1.0 - dist,
                }
            )
        return results

    def query_by_cluster(
        self,
        query_embedding: np.ndarray,
        cluster_id: int,
        n_results: int = TOP_K_RESULTS,
    ) -> list[dict]:
        """
        Restrict semantic search to documents belonging to a specific cluster.

        This is the key efficiency win: instead of scanning all ~18 k documents
        we only scan the ~1.2 k docs in a cluster (assuming uniform distribution).
        For a large cache, the same principle applies — see semantic_cache.py.
        """
        return self.query(
            query_embedding,
            n_results=n_results,
            where={"dominant_cluster": cluster_id},
        )

    # ── Stats ──────────────────────────────────────────────────────────────────
    def count(self) -> int:
        return self._col.count()

    def is_empty(self) -> bool:
        return self.count() == 0


# Module-level singleton
vector_store = VectorStore()
