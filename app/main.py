"""
main.py — FastAPI application for the Newsgroups Semantic Search system.

Start with:
    uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

Endpoints
─────────
  POST   /query             → semantic search with cache
  GET    /cache/stats       → cache statistics
  DELETE /cache             → flush cache
  PUT    /cache/threshold   → hot-swap similarity threshold
  GET    /cache/top         → top hit entries (debug)
  GET    /cluster/info      → cluster model metadata
  GET    /health            → liveness check
  GET    /docs              → Swagger UI (auto-generated)
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from app.search import process_query
from app.semantic_cache import semantic_cache
from app.vector_store import vector_store
from app.fuzzy_cluster import get_cluster_model, FuzzyClustering
from app.config import N_CLUSTERS, CACHE_SIMILARITY_THRESHOLD


# ── Lifespan: warm up heavy objects before accepting traffic ──────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan context manager.
    Pre-loads the embedding model and cluster model so the first request
    doesn't pay the cold-start penalty (~2 s on CPU).
    """
    print("[startup] Loading embedding model ...")
    from app.embedder import embedder  # triggers model download/load
    _ = embedder.dim

    print("[startup] Loading cluster model ...")
    try:
        _ = get_cluster_model()
        print("[startup] Cluster model ready ✓")
    except RuntimeError as e:
        print(f"[startup] WARNING: {e}")
        print("[startup] Run `python scripts/ingest.py && python scripts/cluster.py` first.")

    print("[startup] Checking vector store ...")
    count = vector_store.count()
    if count == 0:
        print("[startup] WARNING: Vector store is empty. Run `python scripts/ingest.py` first.")
    else:
        print(f"[startup] Vector store contains {count:,} documents ✓")

    print("[startup] Service ready 🚀")
    yield
    print("[shutdown] Goodbye.")


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Newsgroups Semantic Search",
    description=(
        "Semantic search over the 20 Newsgroups corpus with fuzzy clustering "
        "and a cluster-indexed semantic cache. Built from scratch — no Redis, "
        "no Memcached, no caching middleware."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request / Response schemas ────────────────────────────────────────────────
class QueryRequest(BaseModel):
    query: str = Field(..., min_length=3, max_length=1000, example="What are the best rockets for space exploration?")


class QueryResponse(BaseModel):
    query: str
    cache_hit: bool
    matched_query: Optional[str]
    similarity_score: float
    result: dict
    dominant_cluster: int
    cluster_distribution: dict


class CacheStatsResponse(BaseModel):
    total_entries: int
    hit_count: int
    miss_count: int
    hit_rate: float
    current_threshold: float
    bucket_sizes: dict


class ThresholdRequest(BaseModel):
    threshold: float = Field(..., gt=0.0, lt=1.0, example=0.85)


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.post(
    "/query",
    response_model=QueryResponse,
    summary="Semantic search with caching",
    description=(
        "Embeds the query, checks the semantic cache, and returns results. "
        "On a cache miss the vector store is queried and the result is stored "
        "for future similar queries."
    ),
)
def query_endpoint(body: QueryRequest) -> dict:
    if vector_store.is_empty():
        raise HTTPException(
            status_code=503,
            detail="Vector store is empty. Run `python scripts/ingest.py` to index the corpus.",
        )
    return process_query(body.query)

@app.get("/")
def root():
    return {
        "message": "Newsgroups Semantic Search API",
        "docs": "http://localhost:8000/docs"
    }
@app.get(
    "/cache/stats",
    response_model=CacheStatsResponse,
    summary="Cache statistics",
)
def cache_stats() -> dict:
    return semantic_cache.stats()


@app.delete(
    "/cache",
    summary="Flush the semantic cache",
    description="Clears all cached entries and resets hit/miss counters.",
)
def flush_cache() -> dict:
    semantic_cache.flush()
    return {"status": "ok", "message": "Cache flushed successfully."}


@app.put(
    "/cache/threshold",
    summary="Update similarity threshold",
    description=(
        "Hot-swap the cosine similarity threshold without restarting the service. "
        "Lower values make the cache more permissive (more hits, less precision). "
        "Higher values make it more strict (fewer hits, higher precision)."
    ),
)
def update_threshold(body: ThresholdRequest) -> dict:
    semantic_cache.set_threshold(body.threshold)
    return {
        "status": "ok",
        "new_threshold": body.threshold,
        "message": f"Threshold updated to {body.threshold}. Cache entries preserved.",
    }


@app.get(
    "/cache/top",
    summary="Top cache entries by hit count (debug)",
)
def top_cache_entries(n: int = Query(10, ge=1, le=100)) -> dict:
    return {"entries": semantic_cache.top_entries(n)}


@app.get(
    "/cluster/info",
    summary="Cluster model information",
)
def cluster_info() -> dict:
    try:
        model = get_cluster_model()
        return {
            "status": "loaded",
            "n_clusters": model.n_clusters,
            "fuzziness_m": model.m,
            "pca_components": model.pca_components,
            "pca_explained_variance": (
                float(model.pca.explained_variance_ratio_.sum())
                if model.pca else None
            ),
        }
    except RuntimeError as e:
        return {"status": "not_loaded", "error": str(e)}


@app.get("/health", summary="Liveness check")
def health() -> dict:
    return {
        "status": "ok",
        "vector_store_count": vector_store.count(),
        "cache_entries": semantic_cache.total_entries,
        "cache_hit_rate": semantic_cache.hit_rate,
    }
