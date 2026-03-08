#!/usr/bin/env python3
"""
scripts/cluster.py — Train the Fuzzy C-Means model, analyse the clusters,
                     and back-fill cluster metadata into ChromaDB.
"""

import sys
import json
import pickle
import numpy as np
from pathlib import Path
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.fuzzy_cluster import FuzzyClustering, partition_coefficient
from app.vector_store import vector_store
from app.config import N_CLUSTERS, MODEL_DIR


# ─────────────────────────────────────────────────────────────
# Fetch embeddings from ChromaDB
# ─────────────────────────────────────────────────────────────

def fetch_all_from_chroma():
    print("[cluster] Fetching all documents from ChromaDB ...")

    col = vector_store._col
    total = col.count()

    print(f"[cluster] Total documents: {total:,}")

    PAGE = 2000

    all_embeddings = []
    all_docs = []
    all_ids = []
    all_metas = []

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


# ─────────────────────────────────────────────────────────────
# Helper Functions
# ─────────────────────────────────────────────────────────────

def _membership_entropy(u_row):
    p = np.clip(u_row, 1e-10, 1.0)
    return float(-np.sum(p * np.log(p)))


def top_tfidf_terms(docs, n=15):

    if not docs:
        return []

    vec = TfidfVectorizer(
        stop_words="english",
        max_features=10000,
        ngram_range=(1, 2)
    )

    try:
        X = vec.fit_transform(docs)

        mean_tfidf = np.asarray(X.mean(axis=0)).ravel()

        top_idx = mean_tfidf.argsort()[::-1][:n]

        return [vec.get_feature_names_out()[i] for i in top_idx]

    except ValueError:
        return []


# ─────────────────────────────────────────────────────────────
# FIXED CLUSTER ANALYSIS
# ─────────────────────────────────────────────────────────────

def analyse_clusters(U, docs, metas):

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

        categories = [m.get("source_category", "unknown") for m in member_metas]

        cat_counts = Counter(categories).most_common(5)

        terms = top_tfidf_terms(member_docs[:500])

        member_indices = np.where(members_mask)[0]

        print(f"\n┌─ Cluster {k:02d} ({'─'*50})")

        print(f"│  Hard members : {members_mask.sum():,}")

        # SAFE MEMBERSHIP STATISTICS
        if len(member_memberships) > 0:

            print(
                f"│  Mean membership: {member_memberships.mean():.3f}  "
                f"(min={member_memberships.min():.3f}, max={member_memberships.max():.3f})"
            )

        else:

            print("│  Mean membership: N/A (no members)")

        print(
            f"│  Top categories: "
            f"{', '.join(f'{cat}({cnt})' for cat, cnt in cat_counts)}"
        )

        print(f"│  Key terms: {', '.join(terms[:10])}")

        print(f"│  Boundary docs (highest entropy — most ambiguous):")

        # SAFE BOUNDARY DOCUMENTS
        if len(member_indices) > 0:

            member_entropies = entropies[member_indices]

            boundary_idx = member_indices[
                np.argsort(member_entropies)[::-1][:3]
            ]

            for bi in boundary_idx:

                snippet = docs[bi][:120].replace("\n", " ")

                ent = entropies[bi]

                print(f"│    [ent={ent:.3f}] \"{snippet}...\"")

        else:

            print("│    No boundary documents (cluster empty)")

        print(f"└{'─'*70}")

    print(f"\n[GLOBAL] Mean entropy: {entropies.mean():.4f}")

    print(f"[GLOBAL] Partition coefficient: {partition_coefficient(U):.4f}")

    print(
        f"[GLOBAL] Fraction dominant >0.5: "
        f"{(U.max(axis=1) > 0.5).mean():.2%}"
    )

    print(
        f"[GLOBAL] Fraction ambiguous (max<0.4): "
        f"{(U.max(axis=1) < 0.4).mean():.2%}"
    )

    print("═" * 72)


# ─────────────────────────────────────────────────────────────
# Update ChromaDB metadata
# ─────────────────────────────────────────────────────────────

def update_chroma_metadata(ids, hard_assignments, U, metas):

    print("\n[cluster] Back-filling cluster metadata in ChromaDB ...")

    col = vector_store._col

    BATCH = 500

    updated_metas = []

    for i, meta in enumerate(metas):

        top2 = np.argsort(U[i])[::-1][:2]

        updated = dict(meta)

        updated["dominant_cluster"] = int(hard_assignments[i])

        updated["cluster_0_id"] = int(top2[0])
        updated["cluster_0_score"] = round(float(U[i, top2[0]]), 4)

        updated["cluster_1_id"] = int(top2[1]) if len(top2) > 1 else -1
        updated["cluster_1_score"] = (
            round(float(U[i, top2[1]]), 4) if len(top2) > 1 else 0.0
        )

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


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main():

    if vector_store.is_empty():

        print("ERROR: Vector store is empty. Run `python scripts/ingest.py` first.")

        sys.exit(1)

    embeddings, docs, ids, metas = fetch_all_from_chroma()

    model = FuzzyClustering(n_clusters=N_CLUSTERS)

    print(f"\n[cluster] Fitting FCM with c={N_CLUSTERS} ...")

    model.fit(embeddings)

    U = model.membership_

    hard_assignments = np.argmax(U, axis=1)

    analyse_clusters(U, docs, metas)

    model.save()

    update_chroma_metadata(ids, hard_assignments, U, metas)

    np.save(MODEL_DIR / "membership_matrix.npy", U)

    np.save(MODEL_DIR / "hard_assignments.npy", hard_assignments)

    print("\n[cluster] ✓ All done. Membership matrix saved to models/")


if __name__ == "__main__":
    main()