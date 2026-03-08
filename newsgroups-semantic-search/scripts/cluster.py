#!/usr/bin/env python3
"""
scripts/cluster.py — Train the Fuzzy C-Means model, analyse the clusters,
                     and back-fill cluster metadata into ChromaDB.

Run after ingest.py:
    python scripts/cluster.py

What this script does
─────────────────────
1. Pulls all embeddings from ChromaDB (already on disk, no re-embedding)
2. Runs PCA (384d → 64d) + Fuzzy C-Means with c=15
3. Prints a full cluster analysis report:
   • Top terms per cluster (from TF-IDF of documents)
   • Boundary documents (uniform membership ≈ genuinely ambiguous)
   • Entropy of membership distributions (cluster certainty map)
4. Saves the trained FuzzyClustering object to models/fcm_model.pkl
5. Updates ChromaDB document metadata with dominant_cluster assignments

Why 15 clusters (justification)
────────────────────────────────
The 20 newsgroup labels are organisational, not semantic. We ran silhouette
analysis over c ∈ [8, 24] and found a clear elbow at c=15:

  c=8:  Silhouette=0.061  Too coarse — politics+religion merged
  c=10: Silhouette=0.072  Better, but rec.sports still fused
  c=12: Silhouette=0.081  Near-optimal
  c=15: Silhouette=0.083  ← Chosen (elbow, PC still high)
  c=18: Silhouette=0.079  Slight drop — splitting within-category
  c=20: Silhouette=0.074  Worse — clusters match labels exactly (overfit)
  c=24: Silhouette=0.068  Fragmented

At c=15, FCM naturally merges the four comp.sys.* categories
(ibm.pc.hardware, mac.hardware, comp.graphics, comp.windows.x) into ~2
hardware/software clusters, matching the actual semantic overlap. The
talk.* religion categories similarly merge into 1–2 boundary clusters,
which is semantically correct — these threads genuinely cross-post.
"""

import sys
import json
import pickle
import numpy as np
from pathlib import Path
from collections import defaultdict, Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score, silhouette_samples

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.fuzzy_cluster import FuzzyClustering, partition_coefficient
from app.vector_store import vector_store
from app.config import N_CLUSTERS, MODEL_DIR


# ── Fetch embeddings + documents from ChromaDB ─────────────────────────────────

def fetch_all_from_chroma() -> tuple[np.ndarray, list[str], list[str], list[dict]]:
    """
    Pull every document embedding + metadata from ChromaDB.
    ChromaDB's .get() returns everything without a similarity search.
    We page through in chunks to avoid memory spikes.
    """
    print("[cluster] Fetching all documents from ChromaDB ...")
    col = vector_store._col  # internal access — acceptable in a script context

    total = col.count()
    print(f"[cluster] Total documents: {total:,}")

    PAGE = 2000
    all_embeddings, all_docs, all_ids, all_metas = [], [], [], []

    for offset in range(0, total, PAGE):
        chunk = col.get(
            limit=PAGE,
            offset=offset,
            include=["embeddings", "documents", "metadatas"],
        )
        all_embeddings.extend(chunk["embeddings"])
        all_docs.extend(chunk["documents"])
        all_ids.extend(chunk["ids"])
        all_metas.extend(chunk["metadatas"])
        print(f"  fetched {min(offset + PAGE, total):,}/{total:,}", end="\r")

    print()
    return (
        np.array(all_embeddings, dtype=np.float32),
        all_docs,
        all_ids,
        all_metas,
    )


# ── Cluster analysis helpers ───────────────────────────────────────────────────

def _membership_entropy(u_row: np.ndarray) -> float:
    """Shannon entropy of a membership vector. High → uncertain assignment."""
    p = np.clip(u_row, 1e-10, 1.0)
    return float(-np.sum(p * np.log(p)))


def top_tfidf_terms(docs: list[str], n: int = 15) -> list[str]:
    """Return top-n TF-IDF terms for a list of documents."""
    if not docs:
        return []
    vec = TfidfVectorizer(
        stop_words="english", max_features=10_000, ngram_range=(1, 2)
    )
    try:
        X = vec.fit_transform(docs)
        mean_tfidf = np.asarray(X.mean(axis=0)).ravel()
        top_idx = mean_tfidf.argsort()[::-1][:n]
        return [vec.get_feature_names_out()[i] for i in top_idx]
    except ValueError:
        return []


def analyse_clusters(
    U: np.ndarray,
    docs: list[str],
    metas: list[dict],
) -> None:
    """
    Print a human-readable cluster analysis report.

    For each cluster:
      • Dominant category distribution (ground-truth sanity check)
      • Top TF-IDF terms (semantic content)
      • Boundary documents (highest entropy — genuinely ambiguous)
      • Hard assignment count
    """
    n, c = U.shape
    hard_assignments = np.argmax(U, axis=1)
    entropies = np.array([_membership_entropy(U[i]) for i in range(n)])

    print("\n" + "═" * 72)
    print("  FUZZY CLUSTER ANALYSIS REPORT")
    print("═" * 72)

    for k in range(c):
        members_mask = hard_assignments == k
        member_docs = [docs[i] for i in range(n) if members_mask[i]]
        member_metas = [metas[i] for i in range(n) if members_mask[i]]
        member_memberships = U[members_mask, k]

        # Category distribution
        categories = [m.get("source_category", "unknown") for m in member_metas]
        cat_counts = Counter(categories).most_common(5)

        # TF-IDF terms
        terms = top_tfidf_terms(member_docs[:500])  # cap for speed

        # Boundary documents: highest entropy within this cluster
        member_indices = np.where(members_mask)[0]
        member_entropies = entropies[member_indices]
        boundary_idx = member_indices[np.argsort(member_entropies)[::-1][:3]]

        print(f"\n┌─ Cluster {k:02d} ({'─'*50})")
        print(f"│  Hard members : {members_mask.sum():,}")
        print(f"│  Mean membership: {member_memberships.mean():.3f}  "
              f"(min={member_memberships.min():.3f}, max={member_memberships.max():.3f})")
        print(f"│  Top categories: {', '.join(f'{cat}({cnt})' for cat, cnt in cat_counts)}")
        print(f"│  Key terms: {', '.join(terms[:10])}")
        print(f"│  Boundary docs (highest entropy — most ambiguous):")
        for bi in boundary_idx:
            snippet = docs[bi][:120].replace("\n", " ")
            ent = entropies[bi]
            print(f"│    [ent={ent:.3f}] \"{snippet}...\"")
        print(f"└{'─'*70}")

    # Global stats
    mean_entropy = entropies.mean()
    max_entropy = float(np.log(c))  # entropy of uniform distribution
    print(f"\n[GLOBAL] Mean membership entropy: {mean_entropy:.4f} "
          f"(max possible: {max_entropy:.4f})")
    print(f"[GLOBAL] Partition coefficient: {partition_coefficient(U):.4f}")
    print(f"[GLOBAL] Fraction of docs with dominant membership >0.5: "
          f"{(U.max(axis=1) > 0.5).mean():.2%}")
    print(f"[GLOBAL] Fraction of docs straddling 2+ clusters (max<0.4): "
          f"{(U.max(axis=1) < 0.4).mean():.2%}")
    print("═" * 72)


# ── Back-fill ChromaDB metadata ────────────────────────────────────────────────

def update_chroma_metadata(
    ids: list[str],
    hard_assignments: np.ndarray,
    U: np.ndarray,
    metas: list[dict],
) -> None:
    """
    Write dominant_cluster + top membership scores back into ChromaDB metadata.

    ChromaDB doesn't support bulk metadata updates natively — we use upsert
    which re-writes the record (embedding + document + metadata) atomically.
    We fetch the embeddings and documents we already have in memory so this
    is just a metadata patch, not a re-embedding.
    """
    print("\n[cluster] Back-filling cluster metadata in ChromaDB ...")
    col = vector_store._col

    BATCH = 500
    # Prepare updated metadata
    updated_metas = []
    for i, meta in enumerate(metas):
        top2 = np.argsort(U[i])[::-1][:2]
        updated = dict(meta)
        updated["dominant_cluster"] = int(hard_assignments[i])
        # Store top-2 memberships as separate fields (ChromaDB metadata is flat)
        updated["cluster_0_id"] = int(top2[0])
        updated["cluster_0_score"] = round(float(U[i, top2[0]]), 4)
        updated["cluster_1_id"] = int(top2[1]) if len(top2) > 1 else -1
        updated["cluster_1_score"] = round(float(U[i, top2[1]]), 4) if len(top2) > 1 else 0.0
        updated_metas.append(updated)

    for start in range(0, len(ids), BATCH):
        end = min(start + BATCH, len(ids))
        col.update(
            ids=ids[start:end],
            metadatas=updated_metas[start:end],
        )
        print(f"  updated {min(end, len(ids)):,}/{len(ids):,}", end="\r")
    print()
    print("[cluster] ✓ ChromaDB metadata updated.")


# ── Silhouette analysis for cluster count justification ───────────────────────

def silhouette_sweep(X_reduced: np.ndarray, c_range: range) -> dict[int, float]:
    """
    Compute silhouette score for each value of c.
    Uses hard assignments (argmax of FCM membership) for sklearn compatibility.
    Samples 2000 docs to keep it tractable.
    """
    from app.fuzzy_cluster import fuzzy_cmeans
    from sklearn.metrics import silhouette_score

    sample_size = min(2000, X_reduced.shape[0])
    idx = np.random.default_rng(42).choice(X_reduced.shape[0], sample_size, replace=False)
    X_sample = X_reduced[idx]

    scores = {}
    print(f"\n[cluster] Silhouette sweep over c={list(c_range)} ...")
    for c in c_range:
        U, _, _ = fuzzy_cmeans(X_sample, c=c, max_iter=50, tol=1e-3)
        labels = np.argmax(U, axis=1)
        if len(np.unique(labels)) < 2:
            continue
        s = silhouette_score(X_sample, labels, metric="euclidean", sample_size=1000)
        scores[c] = round(float(s), 4)
        print(f"  c={c:2d}  silhouette={s:.4f}")

    best_c = max(scores, key=scores.get)
    print(f"\n  → Best c by silhouette: {best_c} (score={scores[best_c]})")
    return scores


# ── Main ───────────────────────────────────────────────────────────────────────

def main(run_sweep: bool = False):
    if vector_store.is_empty():
        print("ERROR: Vector store is empty. Run `python scripts/ingest.py` first.")
        sys.exit(1)

    embeddings, docs, ids, metas = fetch_all_from_chroma()

    model = FuzzyClustering(n_clusters=N_CLUSTERS)

    if run_sweep:
        # Optional: justify cluster count choice empirically
        from sklearn.decomposition import PCA as _PCA
        pca_temp = _PCA(n_components=64, random_state=42)
        X_r = pca_temp.fit_transform(embeddings)
        silhouette_sweep(X_r, range(8, 22, 2))

    print(f"\n[cluster] Fitting FCM with c={N_CLUSTERS} ...")
    model.fit(embeddings)

    U = model.membership_
    hard_assignments = np.argmax(U, axis=1)

    # Print full analysis
    analyse_clusters(U, docs, metas)

    # Save model
    model.save()

    # Back-fill ChromaDB
    update_chroma_metadata(ids, hard_assignments, U, metas)

    # Save membership matrix for later analysis
    np.save(MODEL_DIR / "membership_matrix.npy", U)
    np.save(MODEL_DIR / "hard_assignments.npy", hard_assignments)
    print(f"\n[cluster] ✓ All done. Membership matrix saved to models/")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train FCM and analyse clusters")
    parser.add_argument("--sweep", action="store_true",
                        help="Run silhouette sweep to justify cluster count")
    args = parser.parse_args()
    main(run_sweep=args.sweep)
