"""
search.py — Orchestrates embedding → cache lookup → vector search → cache store.

This is the main business logic layer that the FastAPI endpoints call.
It keeps the route handlers thin and the logic testable in isolation.
"""

from __future__ import annotations

import numpy as np
from app.embedder import embedder
from app.vector_store import vector_store
from app.fuzzy_cluster import get_cluster_model
from app.semantic_cache import semantic_cache
from app.config import TOP_K_RESULTS, CACHE_CLUSTER_SEARCH_DEPTH


def _format_results(raw_results: list[dict]) -> list[dict]:
    """Clean up vector store results for API consumption."""
    return [
        {
            "document": r["document"][:500] + ("..." if len(r["document"]) > 500 else ""),
            "source_category": r["metadata"].get("source_category", "unknown"),
            "similarity": round(r["similarity"], 4),
            "dominant_cluster": r["metadata"].get("dominant_cluster", -1),
        }
        for r in raw_results
    ]


def process_query(query: str) -> dict:
    """
    Full query pipeline:
      1. Embed the query
      2. Get soft cluster memberships from FCM model
      3. Check the semantic cache
      4. On hit  → return cached result
      5. On miss → run vector search, store in cache, return fresh result

    Returns the full API response dict.
    """
    # ── Step 1: Embed ────────────────────────────────────────────────────────
    query_embedding = embedder.embed(query)       # (384,) float32, L2-normalised

    # ── Step 2: Cluster membership ───────────────────────────────────────────
    try:
        model = get_cluster_model()
        top_clusters = model.top_k_clusters(query_embedding, k=CACHE_CLUSTER_SEARCH_DEPTH)
        dominant_cluster = top_clusters[0][0]
        cluster_memberships = model.predict_proba(query_embedding[np.newaxis, :])[0]
    except RuntimeError:
        # Cluster model not yet trained — degrade gracefully
        top_clusters = [(0, 1.0)]
        dominant_cluster = 0
        cluster_memberships = None

    # ── Step 3: Cache lookup ─────────────────────────────────────────────────
    cached_entry, similarity_score = semantic_cache.lookup(query_embedding, top_clusters)

    if cached_entry is not None:
        # ── Cache HIT ────────────────────────────────────────────────────────
        return {
            "query": query,
            "cache_hit": True,
            "matched_query": cached_entry.query,
            "similarity_score": round(similarity_score, 4),
            "result": cached_entry.result,
            "dominant_cluster": dominant_cluster,
            "cluster_distribution": (
                {str(i): round(float(v), 4) for i, v in enumerate(cluster_memberships)}
                if cluster_memberships is not None else {}
            ),
        }

    # ── Step 4: Cache MISS — run semantic search ─────────────────────────────
    raw_results = vector_store.query(query_embedding, n_results=TOP_K_RESULTS)
    formatted = _format_results(raw_results)

    result_payload = {
        "top_results": formatted,
        "result_count": len(formatted),
    }

    # ── Step 5: Store in cache ───────────────────────────────────────────────
    semantic_cache.store(
        query=query,
        query_embedding=query_embedding,
        result=result_payload,
        dominant_cluster=dominant_cluster,
    )

    return {
        "query": query,
        "cache_hit": False,
        "matched_query": None,
        "similarity_score": 0.0,
        "result": result_payload,
        "dominant_cluster": dominant_cluster,
        "cluster_distribution": (
            {str(i): round(float(v), 4) for i, v in enumerate(cluster_memberships)}
            if cluster_memberships is not None else {}
        ),
    }
