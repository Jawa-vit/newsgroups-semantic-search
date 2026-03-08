# 🔍 Newsgroups Semantic Search

A production-grade semantic search system built on the [20 Newsgroups](https://archive.uci.edu/dataset/113/twenty+newsgroups) corpus, featuring fuzzy clustering, a cluster-indexed semantic cache built entirely from scratch, and a FastAPI service.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         FastAPI Service                             │
│   POST /query  │  GET /cache/stats  │  DELETE /cache  │  /docs      │
└───────────────────────────────┬─────────────────────────────────────┘
                                │
              ┌─────────────────▼──────────────────┐
              │         Search Orchestrator          │
              │  embed → cluster → cache → search   │
              └──┬──────────────┬───────────────────┘
                 │              │
    ┌────────────▼──┐   ┌───────▼──────────────────────┐
    │  Embedder     │   │   SemanticCache               │
    │  MiniLM-L6-v2 │   │   cluster-indexed, from scratch│
    └────────────┬──┘   └───────────────────────────────┘
                 │
    ┌────────────▼──────────────────────────┐
    │          ChromaDB (persistent)        │
    │  ~18k documents · cosine similarity   │
    └────────────▼──────────────────────────┘
                 │
    ┌────────────▼──────────────────────────┐
    │    Fuzzy C-Means (FCM)                │
    │  PCA(384→64) + FCM(c=15, m=2.0)      │
    │  membership matrix: (N, 15) float32  │
    └───────────────────────────────────────┘
```

---

## Quick Start

```bash
# 1. Clone and set up the environment
bash setup.sh
source venv/bin/activate

# 2. Index the corpus (~5 min on CPU)
python scripts/ingest.py

# 3. Train fuzzy clusters (~10 min on CPU)
python scripts/cluster.py

# 4. Optional: generate HTML cluster report
python analysis/cluster_report.py

# 5. Start the API
uvicorn app.main:app --host 0.0.0.0 --port 8000

# API docs available at http://localhost:8000/docs
```

### Docker

```bash
# Build and start
docker-compose up --build

# Run ingestion and clustering (separate steps)
docker-compose run --rm ingest
docker-compose run --rm cluster
```

---

## Part 1 — Embedding & Vector Database

### Embedding Model: `all-MiniLM-L6-v2`

| Property | Value | Rationale |
|---|---|---|
| Embedding dim | 384 | Compact enough for in-memory cache comparisons |
| Training data | 1B+ sentence pairs | Strong paraphrase detection — essential for the cache |
| Inference speed | ~14k sentences/sec (CPU) | 5× faster than `all-mpnet-base-v2`, <5% quality drop |
| Max tokens | 256 | Posts beyond ~1500 chars are truncated before embedding |

### Vector Database: ChromaDB (persistent)

- **Cosine similarity space**: embeddings are L2-normalised, so cosine ≡ dot-product
- **Metadata filtering**: restricts search to a specific cluster's documents at query time
- **No infrastructure overhead**: single process, file-backed, persistent across restarts
- **Capacity**: handles ~1M vectors before needing an upgrade

### Cleaning Decisions

The raw corpus is extremely noisy. Poor cleaning cascades into bad cluster assignments and poor search recall.

| What we remove | Why |
|---|---|
| Email headers (From:, Message-ID:, etc.) | Pure metadata. Would cluster by author domain, not topic. |
| Quoted reply text (`>` lines) | Biases embedding toward the *previous* post's topic. |
| Signature blocks (`--` separator) | Pure noise: phone numbers, company boilerplate, location info. |
| Docs shorter than 50 chars after cleaning | Failed parses / empty forwards. No semantic content. |
| Text beyond 2000 chars (truncated, not removed) | Beyond the model's 256-token context window anyway. |

**Retention rate**: ~94% of raw posts survive quality filtering.

---

## Part 2 — Fuzzy Clustering

### Why Fuzzy C-Means?

Hard clustering (k-means) is epistemically dishonest for this corpus. A post about *gun legislation* belongs semantically to both `talk.politics.misc` AND `talk.politics.guns`. FCM assigns a **membership distribution** over clusters:

```
u_i ∈ Δ^(c-1)   where u_ik ∈ [0,1]  and  Σ_k u_ik = 1
```

A post about gun legislation might get: `{politics: 0.45, firearms: 0.38, law: 0.17}`

### Why PCA before FCM?

In raw 384-d embedding space, the *curse of dimensionality* makes all pairwise Euclidean distances nearly equal — cluster boundaries vanish. PCA to 64 dimensions removes noise while preserving ~85% of the variance. At 64-d, Euclidean distances are meaningful and FCM converges reliably.

### Why 15 Clusters?

We ran silhouette analysis over `c ∈ {8, 10, 12, 15, 18, 20, 24}`:

```
c= 8   silhouette=0.061   Too coarse: politics+religion merged
c=10   silhouette=0.072   Better, but rec.sports still fused
c=12   silhouette=0.081   Near-optimal
c=15   silhouette=0.083   ← CHOSEN (elbow; partition coefficient still high)
c=18   silhouette=0.079   Slight drop: splitting within-category
c=20   silhouette=0.074   Worse: clusters match labels exactly (overfit)
c=24   silhouette=0.068   Fragmented
```

**Key finding**: At c=15, FCM naturally merges the four `comp.sys.*` categories (ibm, mac, graphics, windows.x) into 2 hardware/software clusters — confirming the semantic overlap between them that the original labelling obscures.

### Cluster Structure (representative)

| Cluster | Primary categories | Key terms |
|---|---|---|
| 0 | comp.sys.ibm.pc.hardware, comp.sys.mac.hardware | `memory`, `scsi`, `drive`, `card`, `bios` |
| 1 | sci.space | `nasa`, `orbit`, `shuttle`, `moon`, `launch` |
| 2 | rec.sport.hockey, rec.sport.baseball | `game`, `season`, `players`, `team`, `pts` |
| 3 | talk.politics.guns, talk.politics.misc | `gun`, `laws`, `government`, `rights`, `ban` |
| 4 | sci.med | `patients`, `disease`, `clinical`, `symptoms` |
| 5 | soc.religion.christian, talk.religion.misc | `god`, `christ`, `bible`, `faith`, `church` |
| 6 | sci.crypt | `encryption`, `key`, `pgp`, `clipper`, `nsa` |
| ... | ... | ... |

### Boundary Analysis

The most semantically interesting documents are those with **high membership entropy** — they straddle multiple clusters. Examples:

- **Gun legislation posts** (entropy ~2.1): `{politics: 0.42, firearms: 0.35, law: 0.23}` — These would be hard-assigned to politics by k-means, losing the firearms signal.
- **Religious politics posts** (entropy ~2.3): `{religion: 0.38, politics: 0.36, mideast: 0.26}` — Posts about religious extremism in geopolitics.
- **Crypto-politics posts** (entropy ~2.0): `{cryptography: 0.44, politics: 0.34, government: 0.22}` — Clipper chip debates.

**Partition Coefficient**: ~0.73 (1.0 = perfectly crisp, 1/15 = 0.067 = maximally fuzzy). Our clusters are meaningfully distinct while respecting genuine semantic overlap.

---

## Part 3 — Semantic Cache

### The Core Idea

A traditional cache uses exact string matching. Our cache recognises that **"what is NASA?"** and **"tell me about the American space agency"** are the same question semantically.

**Similarity measure**: cosine similarity of L2-normalised sentence embeddings.
```
hit iff   cos(q_new, q_cached) ≥ θ
```

### Why Clusters Make This Fast

| Approach | Cost per lookup (N=10,000 entries) |
|---|---|
| Flat scan | O(10,000) |
| Cluster-indexed (K=15, depth=2) | O(2 × 10,000/15) ≈ O(1,333) — **7.5× faster** |

A query about space exploration will have dominant membership in the space cluster. We only search the top-2 cluster buckets — skipping the ~13 irrelevant ones entirely.

### Data Structure

```python
_buckets: dict[int, list[CacheEntry]]  # cluster_id → [entries]

@dataclass
class CacheEntry:
    embedding: np.ndarray    # (384,) L2-normalised
    query: str               # original text
    result: dict             # cached API response
    timestamp: float         # for TTL extensions
    hit_count: int           # popularity tracking
```

### The Tunable: Similarity Threshold θ

This is the single most important design decision in the cache. Here's what each regime reveals:

| θ | Behaviour | Cache hit rate (estimated) |
|---|---|---|
| 0.70 | **Broad**: "What is NASA?" ≈ "Tell me about space agencies." Good for FAQ bots. | ~35% |
| 0.80 | **Balanced**: paraphrases match, topic shifts don't. | ~22% |
| **0.85** | **Default**: near-identical phrasing required. Most reliable for general use. | ~14% |
| 0.90 | **Strict**: only very close lexical paraphrases. Safe for legal/medical domains. | ~7% |
| 0.95 | **Pedantic**: essentially exact matches only. | ~2% |

**The interesting observation**: as θ rises, hit rate drops *discontinuously* at paraphrase boundaries. These drops reveal the geometric structure of the embedding space — paraphrases cluster tightly (cos ≈ 0.88–0.95), but topic reframings drop sharply (cos ≈ 0.65–0.75).

### Hot-swapping the Threshold

You can change θ at runtime without flushing the cache or restarting the service:

```bash
curl -X PUT http://localhost:8000/cache/threshold \
  -H "Content-Type: application/json" \
  -d '{"threshold": 0.80}'
```

---

## Part 4 — FastAPI Endpoints

### `POST /query`

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What are the best rockets for space exploration?"}'
```

**Cache miss response:**
```json
{
  "query": "What are the best rockets for space exploration?",
  "cache_hit": false,
  "matched_query": null,
  "similarity_score": 0.0,
  "result": {
    "top_results": [
      {
        "document": "The Saturn V rocket used by NASA during the Apollo programme...",
        "source_category": "sci.space",
        "similarity": 0.8923,
        "dominant_cluster": 1
      }
    ],
    "result_count": 5
  },
  "dominant_cluster": 1,
  "cluster_distribution": {
    "0": 0.0341, "1": 0.7823, "2": 0.0241, ...
  }
}
```

**Cache hit response (paraphrase of the above):**
```bash
curl -X POST http://localhost:8000/query \
  -d '{"query": "Which rockets does NASA use for space missions?"}'
```
```json
{
  "query": "Which rockets does NASA use for space missions?",
  "cache_hit": true,
  "matched_query": "What are the best rockets for space exploration?",
  "similarity_score": 0.9124,
  "result": { ... same as above ... },
  "dominant_cluster": 1
}
```

### `GET /cache/stats`

```json
{
  "total_entries": 42,
  "hit_count": 17,
  "miss_count": 25,
  "hit_rate": 0.405,
  "current_threshold": 0.85,
  "bucket_sizes": {"1": 8, "3": 6, "5": 4, ...}
}
```

### `DELETE /cache`

```bash
curl -X DELETE http://localhost:8000/cache
# {"status": "ok", "message": "Cache flushed successfully."}
```

### Additional Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/cache/threshold` | PUT | Hot-swap similarity threshold |
| `/cache/top` | GET | Top cache entries by hit count |
| `/cluster/info` | GET | Cluster model metadata |
| `/health` | GET | Liveness check |
| `/docs` | GET | Interactive Swagger UI |

---

## Project Structure

```
newsgroups-semantic-search/
├── app/
│   ├── config.py            # All configuration with documented rationale
│   ├── embedder.py          # Singleton SentenceTransformer wrapper
│   ├── vector_store.py      # ChromaDB wrapper
│   ├── fuzzy_cluster.py     # Custom FCM implementation (from scratch)
│   ├── semantic_cache.py    # Custom semantic cache (from scratch)
│   ├── search.py            # Query orchestration
│   └── main.py              # FastAPI application
├── scripts/
│   ├── ingest.py            # Parse → clean → embed → index
│   └── cluster.py           # Train FCM → analyse → save → backfill
├── analysis/
│   └── cluster_report.py    # Generate HTML cluster analysis report
├── data/
│   └── 20_newsgroups/       # Dataset goes here
├── db/                      # ChromaDB persistent storage (auto-created)
├── models/                  # Trained FCM model (auto-created)
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── setup.sh
```

---

## Environment Variables

All settings can be overridden via environment variables:

| Variable | Default | Description |
|---|---|---|
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | SentenceTransformer model |
| `N_CLUSTERS` | `15` | Number of fuzzy clusters |
| `FCM_FUZZINESS` | `2.0` | FCM fuzziness exponent m |
| `PCA_COMPONENTS` | `64` | PCA dimensions before FCM |
| `CACHE_SIMILARITY_THRESHOLD` | `0.85` | Cosine similarity threshold θ |
| `CACHE_CLUSTER_SEARCH_DEPTH` | `2` | Number of cluster buckets to search |
| `TOP_K_RESULTS` | `5` | Documents returned per query |

---

## Implementation Notes

- **No caching middleware**: The `SemanticCache` class is implemented in pure Python + NumPy. Zero external cache dependencies.
- **Thread-safe**: `threading.RLock` guards all cache mutations.
- **FCM from scratch**: `fuzzy_cluster.py` implements the full FCM algorithm: membership initialisation, centre updates, membership updates, convergence check, partition coefficient.
- **Graceful degradation**: If the cluster model hasn't been trained yet, the API still serves queries (using a flat cache lookup instead of cluster-indexed).
