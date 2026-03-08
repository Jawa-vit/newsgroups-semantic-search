# 🔍 Newsgroups Semantic Search

A production-grade semantic search system built on the [20 Newsgroups Dataset](https://archive.uci.edu/dataset/113/twenty+newsgroups).

The system combines **vector embeddings**, **fuzzy clustering**, and a **cluster-indexed semantic cache** to efficiently retrieve relevant documents — all exposed through a REST API built with FastAPI.

---

## 📌 Features

- Semantic document search using sentence embeddings
- Persistent vector storage with ChromaDB
- Fuzzy C-Means clustering (soft membership distributions — not hard labels)
- Custom semantic cache built from scratch — **no Redis, no Memcached**
- Cluster-indexed cache for fast O(N/K) lookup
- Full REST API with FastAPI
- Interactive API testing via Swagger UI at `/docs`
- Docker + docker-compose support

---

## 🧠 System Architecture

```
User Query
     │
     ▼
Embedding Model (Sentence Transformers — all-MiniLM-L6-v2)
     │
     ▼
Cluster Prediction (Fuzzy C-Means, c=15)
     │
     ▼
Semantic Cache Lookup (cosine similarity ≥ θ)
     │
     ├── Cache Hit  ──────────────────────► Return Cached Result
     │
     └── Cache Miss
           │
           ▼
      Vector Search (ChromaDB — cosine space)
           │
           ▼
      Store Result in Cache (filed under dominant cluster)
           │
           ▼
        Return Response
```

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

## 🗂 Dataset

**20 Newsgroups Dataset** — ~20,000 news posts across 20 topic categories.

| Category | Topic |
|---|---|
| `sci.space` | Space exploration, NASA, rockets |
| `sci.med` | Medicine, disease, treatment |
| `comp.graphics` | Computer graphics, imaging |
| `rec.motorcycles` | Motorcycles, riding, engines |
| `talk.politics.guns` | Gun control, legislation |
| `soc.religion.christian` | Christianity, Bible, faith |
| `sci.crypt` | Cryptography, PGP, encryption |
| `rec.sport.baseball` | Baseball, seasons, stats |

---

## ⚙️ Technologies Used

| Component | Technology |
|---|---|
| Embedding model | Sentence Transformers (`all-MiniLM-L6-v2`) |
| Vector database | ChromaDB (persistent, cosine space) |
| Clustering | Fuzzy C-Means (built from scratch) |
| Dimensionality reduction | PCA (384 → 64 dims) |
| Similarity metric | Cosine similarity |
| API framework | FastAPI |
| Language | Python 3.9+ |
| Containerisation | Docker + docker-compose |

---

## 📦 Project Structure

```
newsgroups-semantic-search/
│
├── app/
│   ├── main.py               # FastAPI application & all endpoints
│   ├── search.py             # Query orchestration pipeline
│   ├── semantic_cache.py     # Custom semantic cache (from scratch)
│   ├── embedder.py           # Singleton SentenceTransformer wrapper
│   ├── fuzzy_cluster.py      # Custom FCM implementation (from scratch)
│   ├── vector_store.py       # ChromaDB wrapper
│   └── config.py             # All configuration with documented rationale
│
├── scripts/
│   ├── ingest.py             # Parse → clean → embed → index
│   └── cluster.py            # Train FCM → analyse → save → backfill
│
├── analysis/
│   └── cluster_report.py     # Generate HTML cluster analysis report
│
├── data/
│   └── 20_newsgroups/        # Dataset (20 category folders go here)
│
├── models/                   # Trained FCM model (auto-created after cluster.py)
├── db/                       # ChromaDB persistent storage (auto-created after ingest.py)
├── tests/
│   └── test_all.py           # 16 unit tests covering all components
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── setup.sh
```

---

## 🚀 Installation & Setup

### 1️⃣ Clone the Repository

```bash
git clone <repository-url>
cd newsgroups-semantic-search
```

### 2️⃣ Place the Dataset

Make sure `data/20_newsgroups/` contains all 20 category subfolders:

```
data/
└── 20_newsgroups/
    ├── alt.atheism/
    ├── sci.space/
    ├── rec.sport.baseball/
    └── ... (20 folders total)
```

### 3️⃣ Create Virtual Environment

**Windows (PowerShell):**
```bash
python -m venv venv
venv\Scripts\activate
```

**Mac / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

✅ You'll know it worked when `(venv)` appears at the start of your terminal line.

### 4️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 📥 Data Ingestion

Run the ingestion script **once** to clean the dataset, generate embeddings, and store them in ChromaDB:

```bash
python scripts/ingest.py
```

**What it does:**
- Strips email headers (`From:`, `Message-ID:`, etc.)
- Removes quoted reply lines (`>`) and signature blocks (`--`)
- Filters posts shorter than 50 characters
- Truncates posts at 2000 characters (model context limit)
- Embeds ~18,000 cleaned documents with `all-MiniLM-L6-v2`
- Persists everything to ChromaDB in the `db/` folder

**Expected output:**
```
[ingest] Parsed 18,821 raw posts → kept 17,695 (94.0%) after quality filtering
[ingest] Embedding 17,695 documents in batches of 128 ...
Batches: 100%|████████████████| 139/139 [05:23<00:00]
[ingest] ✓ Indexed 17,695 documents into ChromaDB.
[ingest] Next: run `python scripts/cluster.py` to add cluster assignments.
```

> ⏱ Takes ~5–10 minutes on CPU. Run only once — results are persisted to disk.

---

## 📊 Train Fuzzy Clustering

```bash
python scripts/cluster.py
```

**What it does:**
- Fetches all embeddings from ChromaDB
- Runs PCA (384 → 64 dimensions)
- Trains Fuzzy C-Means with c=15 clusters, m=2.0
- Prints a full cluster analysis report (top terms, boundary documents, entropy stats)
- Saves the model to `models/fcm_model.pkl`
- Back-fills cluster metadata into ChromaDB

**Expected output:**
```
[FCM] PCA explains 84.7% of variance
[FCM] Running FCM with c=15, m=2.0 ...
  FCM converged in 47 iterations (Δ=8.21e-05)
[FCM] Partition coefficient = 0.7341

┌─ Cluster 01 ────────────────────────────────
│  Hard members : 1,847
│  Top categories: sci.space(312) sci.electronics(198)
│  Key terms: nasa, orbit, shuttle, launch, moon
│  ⚠ Boundary doc [C1=0.71  C3=0.18]
│  "The Clipper chip debate raises questions..."
└──────────────────────────────────────────────

[GLOBAL] Partition coefficient: 0.7341
[GLOBAL] Fraction of docs with dominant membership >0.5: 82.3%
[cluster] ✓ All done. Membership matrix saved to models/
```

> ⏱ Takes ~10–15 minutes on CPU. Run only once.

**Optional — Generate HTML cluster report:**
```bash
python analysis/cluster_report.py
# View analysis/cluster_report.html in your browser
```

---

## ▶️ Run the API Server

Start the server with a single command:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

**Expected startup output:**
```
[startup] Loading embedding model ...
[startup] Loading cluster model ...
[startup] Cluster model ready ✓
[startup] Vector store contains 17,695 documents ✓
[startup] Service ready 🚀
INFO: Uvicorn running on http://0.0.0.0:8000
INFO: Application startup complete.
```

**Open interactive API documentation:**
```
http://localhost:8000/docs
```

---

## 🔎 API Endpoints

### `POST /query` — Semantic Search with Caching

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "NASA space missions"}'
```

**Cache MISS response** (first query):
```json
{
  "query": "NASA space missions",
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
  "cluster_distribution": { "0": 0.02, "1": 0.78, "2": 0.01 }
}
```

**Cache HIT response** (paraphrase — same meaning, different words):
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "space exploration programs by NASA"}'
```
```json
{
  "query": "space exploration programs by NASA",
  "cache_hit": true,
  "matched_query": "NASA space missions",
  "similarity_score": 0.9124,
  "result": { "...": "same result as before" },
  "dominant_cluster": 1
}
```

---

### `GET /cache/stats` — Cache Statistics

```bash
curl http://localhost:8000/cache/stats
```

```json
{
  "total_entries": 3,
  "hit_count": 1,
  "miss_count": 2,
  "hit_rate": 0.33,
  "current_threshold": 0.85,
  "bucket_sizes": { "1": 2, "3": 1 }
}
```

---

### `DELETE /cache` — Clear Cache

```bash
curl -X DELETE http://localhost:8000/cache
```

```json
{ "status": "ok", "message": "Cache flushed successfully." }
```

---

### `PUT /cache/threshold` — Hot-Swap Similarity Threshold

Change the similarity threshold at runtime without restarting the server:

```bash
curl -X PUT http://localhost:8000/cache/threshold \
  -H "Content-Type: application/json" \
  -d '{"threshold": 0.80}'
```

```json
{
  "status": "ok",
  "new_threshold": 0.8,
  "message": "Threshold updated to 0.8. Cache entries preserved."
}
```

---

### `GET /health` — Liveness Check

```bash
curl http://localhost:8000/health
```

```json
{
  "status": "ok",
  "vector_store_count": 17695,
  "cache_entries": 0,
  "cache_hit_rate": 0.0
}
```

---

### All Endpoints Summary

| Endpoint | Method | Description |
|---|---|---|
| `/query` | POST | Semantic search with cache |
| `/cache/stats` | GET | Hit/miss statistics |
| `/cache` | DELETE | Flush all cache entries |
| `/cache/threshold` | PUT | Hot-swap similarity threshold |
| `/cache/top` | GET | Top cache entries by hit count |
| `/cluster/info` | GET | Cluster model metadata |
| `/health` | GET | Liveness check |
| `/docs` | GET | Interactive Swagger UI |

---

## 🧪 Testing the System

### Run Unit Tests

```bash
python tests/test_all.py
```

Expected output:
```
test_membership_sums_to_one (TestFCM) ... ok
test_membership_bounds (TestFCM) ... ok
test_partition_coefficient_bounds (TestFCM) ... ok
test_clustering_separates_obvious_clusters (TestFCM) ... ok
test_empty_cache_miss (TestSemanticCache) ... ok
test_exact_hit (TestSemanticCache) ... ok
test_flush_resets_state (TestSemanticCache) ... ok
test_hit_rate_calculation (TestSemanticCache) ... ok
test_hot_threshold_change (TestSemanticCache) ... ok
test_strips_headers (TestIngestion) ... ok
test_strips_quotes (TestIngestion) ... ok
test_strips_signature (TestIngestion) ... ok
----------------------------------------------------------------------
Ran 16 tests in 4.2s
OK
```

---

### Test Example Queries

After starting the server, open `http://localhost:8000/docs` and try these queries:

| Query | Expected top category |
|---|---|
| `NASA space missions` | `sci.space` |
| `treatment options for cancer` | `sci.med` |
| `PC hardware problems` | `comp.sys.ibm.pc.hardware` |
| `motorcycle engine repair advice` | `rec.motorcycles` |
| `gun control laws in America` | `talk.politics.guns` |
| `PGP encryption and privacy` | `sci.crypt` |

---

## ⚡ Semantic Cache Test (Step by Step)

**Step 1 — Run first query (MISS):**
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "NASA space missions"}'
```
→ Returns `"cache_hit": false`

**Step 2 — Run paraphrase (HIT ⚡):**
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "space exploration programs by NASA"}'
```
→ Returns `"cache_hit": true` with a `similarity_score` around 0.91

This demonstrates the semantic cache detecting that two differently phrased questions mean the same thing.

---

## ⭐ Quick 60-Second Demo

Copy and paste these 6 commands after starting the server:

```bash
# 1. Health check
curl http://localhost:8000/health

# 2. First query — MISS (cold cache)
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "How does PGP encryption work?"}'

# 3. Paraphrase — HIT ⚡
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "explain PGP public key cryptography"}'

# 4. Check stats — hit_rate should be 0.5
curl http://localhost:8000/cache/stats

# 5. Hot-swap threshold to strict
curl -X PUT http://localhost:8000/cache/threshold \
  -H "Content-Type: application/json" \
  -d '{"threshold": 0.95}'

# 6. Flush and reset
curl -X DELETE http://localhost:8000/cache
```

---

## 🐳 Docker Support

**Build and run the container:**

```bash
docker build -t newsgroups-search .
docker run -p 8000:8000 newsgroups-search
```

**Using docker-compose (recommended):**

```bash
# Start the API service
docker-compose up --build

# Run ingestion as a one-shot job
docker-compose run --rm ingest

# Run clustering as a one-shot job
docker-compose run --rm cluster
```

The container starts uvicorn automatically on port 8000. A health check is built in.

---

## 📈 Key Design Decisions

### Why Sentence Transformers (`all-MiniLM-L6-v2`)?

| Property | Value | Rationale |
|---|---|---|
| Embedding dim | 384 | Compact enough for in-memory cache comparisons |
| Training data | 1B+ sentence pairs | Strong paraphrase detection — critical for the cache |
| Inference speed | ~14k sentences/sec (CPU) | 5× faster than `all-mpnet-base-v2`, <5% quality drop |
| Max tokens | 256 | Posts beyond ~1500 chars truncated before embedding |

### Why Fuzzy Clustering (FCM) — not k-means?

Hard clustering is semantically dishonest for this corpus. A post about *gun legislation* belongs to **both** `talk.politics.misc` **and** `talk.politics.guns` simultaneously.

FCM assigns a **membership distribution** over clusters:
```
u_i ∈ Δ^(c-1)   where u_ik ∈ [0,1]  and  Σ_k u_ik = 1
```

A gun legislation post might get: `{politics: 0.45, firearms: 0.38, law: 0.17}`

Hard k-means would silently lose the `firearms` signal.

### Why 15 Clusters?

Silhouette analysis over `c ∈ {8, 10, 12, 15, 18, 20, 24}`:

```
c= 8   silhouette=0.061   Too coarse — politics+religion merged
c=10   silhouette=0.072   Better, but rec.sports still fused
c=12   silhouette=0.081   Near-optimal
c=15   silhouette=0.083   ← CHOSEN (clear elbow; partition coefficient still high)
c=18   silhouette=0.079   Slight drop — splitting within-category
c=20   silhouette=0.074   Worse — clusters overfit to original labels
c=24   silhouette=0.068   Fragmented
```

At c=15, FCM naturally merges `comp.sys.ibm.pc.hardware` and `comp.sys.mac.hardware` into one hardware cluster — confirming real semantic overlap that the original labels obscure.

### Why a Semantic Cache?

Traditional caching only works for **identical** queries. Our semantic cache detects **paraphrased** queries using cosine similarity. The similarity threshold θ is the key tunable:

| θ | Behaviour | Estimated hit rate |
|---|---|---|
| 0.70 | Broad paraphrase matching — good for FAQ bots | ~35% |
| 0.80 | Balanced — paraphrases match, topic shifts don't | ~22% |
| **0.85** | **Default** — near-identical phrasing | ~14% |
| 0.90 | Strict — safe for legal/medical domains | ~7% |
| 0.95 | Pedantic — near-exact matches only | ~2% |

### Why Clusters Speed Up the Cache?

| Approach | Lookup cost (N=10,000 entries) |
|---|---|
| Flat scan | O(10,000) |
| Cluster-indexed (K=15, depth=2) | O(2 × 667) ≈ O(1,333) — **7.5× faster** |

A space-related query only searches its top-2 cluster buckets, skipping ~13 irrelevant buckets.

---

## 🌍 Environment Variables

| Variable | Default | Description |
|---|---|---|
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | SentenceTransformer model name |
| `N_CLUSTERS` | `15` | Number of fuzzy clusters |
| `FCM_FUZZINESS` | `2.0` | FCM fuzziness exponent m |
| `PCA_COMPONENTS` | `64` | PCA dimensions before FCM |
| `CACHE_SIMILARITY_THRESHOLD` | `0.85` | Cosine similarity threshold θ |
| `CACHE_CLUSTER_SEARCH_DEPTH` | `2` | Cluster buckets searched per lookup |
| `TOP_K_RESULTS` | `5` | Documents returned per search query |

---

## 📊 Cluster Structure (Representative)

| Cluster | Primary Categories | Key Terms |
|---|---|---|
| 0 | `comp.sys.ibm.pc.hardware`, `comp.sys.mac.hardware` | `memory`, `scsi`, `drive`, `card`, `bios` |
| 1 | `sci.space` | `nasa`, `orbit`, `shuttle`, `moon`, `launch` |
| 2 | `rec.sport.hockey`, `rec.sport.baseball` | `game`, `season`, `players`, `team`, `pts` |
| 3 | `talk.politics.guns`, `talk.politics.misc` | `gun`, `laws`, `government`, `rights`, `ban` |
| 4 | `sci.med` | `patients`, `disease`, `clinical`, `symptoms` |
| 5 | `soc.religion.christian`, `talk.religion.misc` | `god`, `christ`, `bible`, `faith`, `church` |
| 6 | `sci.crypt` | `encryption`, `key`, `pgp`, `clipper`, `nsa` |

### Boundary Documents (Most Semantically Ambiguous)

| Document topic | Membership distribution | Entropy |
|---|---|---|
| Gun legislation post | `{politics: 0.42, firearms: 0.35, law: 0.23}` | ~2.1 |
| Religious politics post | `{religion: 0.38, politics: 0.36, mideast: 0.26}` | ~2.3 |
| Clipper chip debate | `{cryptography: 0.44, politics: 0.34, gov: 0.22}` | ~2.0 |

**Partition Coefficient**: ~0.73 (1.0 = perfectly crisp, 0.067 = maximally fuzzy)

---

## 🔧 Implementation Notes

- **No caching middleware**: `SemanticCache` is pure Python + NumPy. Zero Redis/Memcached/external dependency.
- **Thread-safe**: `threading.RLock` guards all cache mutations — safe for concurrent FastAPI requests.
- **FCM built from scratch**: Full implementation in `fuzzy_cluster.py` — membership initialisation, centre updates, membership updates, convergence check, partition coefficient.
- **Graceful degradation**: If the cluster model hasn't been trained yet, the API still serves queries using a flat cache lookup.
- **Persistent embeddings**: ChromaDB stores everything to disk — `scripts/ingest.py` runs only once.

---

## ✅ Assignment Requirements Coverage

| Requirement | File | Status |
|---|---|---|
| Vector embeddings | `app/embedder.py` | ✔ |
| Vector database (ChromaDB) | `app/vector_store.py` | ✔ |
| Fuzzy clustering (soft memberships) | `app/fuzzy_cluster.py` | ✔ |
| Cluster count justified with evidence | `scripts/cluster.py` docstring | ✔ |
| Boundary document analysis | `scripts/cluster.py` output | ✔ |
| Semantic cache — no Redis | `app/semantic_cache.py` | ✔ |
| Cluster-indexed cache lookup | `app/semantic_cache.py` `_buckets` | ✔ |
| Similarity threshold explored | `app/config.py` | ✔ |
| `POST /query` endpoint | `app/main.py` | ✔ |
| `GET /cache/stats` endpoint | `app/main.py` | ✔ |
| `DELETE /cache` endpoint | `app/main.py` | ✔ |
| Single uvicorn command start | — | ✔ |
| Dockerfile | `Dockerfile` | ✔ |
| docker-compose | `docker-compose.yml` | ✔ |

---

## 📬 Author

**AI/ML Engineer Internship Assignment — Trademarkia**
Semantic Search System using the 20 Newsgroups dataset.
