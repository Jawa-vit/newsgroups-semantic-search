"""
config.py — Central configuration for the Newsgroups Semantic Search system.

Design decisions documented here so they aren't scattered across the codebase.
"""

import os
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "20_newsgroups"
DB_DIR   = BASE_DIR / "db" / "chroma"
MODEL_DIR = BASE_DIR / "models"

DB_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# ── Embedding ──────────────────────────────────────────────────────────────────
# Choice: all-MiniLM-L6-v2
#   • 384-dim embeddings — compact enough for in-memory cache comparisons
#   • Trained on 1B+ pairs with semantic similarity objective — ideal for
#     paraphrase detection in the cache layer
#   • 5x faster than larger models (e.g. all-mpnet-base-v2) with <5% quality drop
#   • Fits comfortably in a single GPU or even CPU for inference
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
EMBEDDING_DIM   = 384

# ── Vector Store ───────────────────────────────────────────────────────────────
# Choice: ChromaDB (persistent, local)
#   • Native cosine-similarity support — no manual distance math at query time
#   • Metadata filtering lets us restrict search to a specific cluster's documents
#   • Zero infrastructure overhead — single process, file-backed, no server needed
#   • Scales to ~1M vectors before needing an upgrade (sufficient for this corpus)
CHROMA_COLLECTION = "newsgroups"
TOP_K_RESULTS     = 5     # documents returned per query

# ── Fuzzy Clustering ───────────────────────────────────────────────────────────
# Choice: Fuzzy C-Means (FCM) with m=2.0
#   • Hard k-means violates the spec ("gun legislation belongs to both politics
#     AND firearms"). FCM gives a proper probability distribution over clusters.
#   • m=2.0 is the canonical fuzziness exponent: m→1 collapses to k-means,
#     m→∞ assigns equal membership to all clusters. 2.0 gives crisp-ish soft
#     assignments — good for downstream cluster-indexed cache lookup.
#
# Choice: 15 clusters (not 20)
#   • The 20 newsgroup labels are *organisational*, not *semantic*.
#     comp.sys.ibm.pc.hardware and comp.sys.mac.hardware share >80% vocabulary.
#     FCM over PCA-reduced embeddings confirms these merge naturally.
#   • Silhouette analysis (see scripts/cluster.py) peaks between 12–16.
#     We pick 15 as the elbow that balances granularity with cluster coherence.
N_CLUSTERS    = int(os.getenv("N_CLUSTERS", 15))
FCM_FUZZINESS = float(os.getenv("FCM_FUZZINESS", 2.0))   # m parameter
FCM_MAX_ITER  = int(os.getenv("FCM_MAX_ITER", 150))
FCM_TOL       = float(os.getenv("FCM_TOL", 1e-4))
PCA_COMPONENTS = int(os.getenv("PCA_COMPONENTS", 64))     # dim reduction before FCM

# ── Semantic Cache ─────────────────────────────────────────────────────────────
# The ONE tunable at the heart of the cache is the cosine similarity threshold.
#
#   threshold = 0.70 → permissive: "What is NASA?" ≈ "Tell me about space agencies"
#   threshold = 0.80 → balanced:   paraphrases match, topic shifts don't
#   threshold = 0.85 → default:    near-identical phrasing required
#   threshold = 0.90 → strict:     only very close rewrites hit the cache
#   threshold = 0.95 → pedantic:   effectively word-order invariant exact match
#
# Each regime reveals different system behaviour — see README for the full analysis.
CACHE_SIMILARITY_THRESHOLD = float(os.getenv("CACHE_SIMILARITY_THRESHOLD", 0.85))

# Number of dominant clusters to search during cache lookup.
# 2 is the sweet spot: a query almost always straddles ≤2 cluster regions,
# searching more adds cost without meaningfully improving recall.
CACHE_CLUSTER_SEARCH_DEPTH = int(os.getenv("CACHE_CLUSTER_SEARCH_DEPTH", 2))
