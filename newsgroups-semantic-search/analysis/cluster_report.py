#!/usr/bin/env python3
"""
analysis/cluster_report.py — Generate a rich HTML cluster analysis report.

Run after cluster.py:
    python analysis/cluster_report.py

Produces analysis/cluster_report.html — open in any browser.
"""

import sys
import json
import numpy as np
from pathlib import Path
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.fuzzy_cluster import get_cluster_model, partition_coefficient
from app.vector_store import vector_store
from app.config import MODEL_DIR


def fetch_data():
    col = vector_store._col
    total = col.count()
    PAGE = 2000
    docs, ids, metas = [], [], []
    for offset in range(0, total, PAGE):
        chunk = col.get(limit=PAGE, offset=offset, include=["documents", "metadatas"])
        docs.extend(chunk["documents"])
        ids.extend(chunk["ids"])
        metas.extend(chunk["metadatas"])
    return docs, ids, metas


def top_terms(docs, n=12):
    if not docs:
        return []
    vec = TfidfVectorizer(stop_words="english", max_features=5000, ngram_range=(1, 2))
    try:
        X = vec.fit_transform(docs)
        mean_tfidf = np.asarray(X.mean(axis=0)).ravel()
        top_idx = mean_tfidf.argsort()[::-1][:n]
        return list(vec.get_feature_names_out()[top_idx])
    except Exception:
        return []


def generate_report():
    print("[report] Loading cluster model and data ...")
    model = get_cluster_model()
    U_path = MODEL_DIR / "membership_matrix.npy"
    if not U_path.exists():
        print("ERROR: Run scripts/cluster.py first.")
        sys.exit(1)

    U = np.load(U_path)
    docs, ids, metas = fetch_data()
    n, c = U.shape
    hard_assignments = np.argmax(U, axis=1)
    entropies = np.array([-np.sum(np.clip(U[i], 1e-10, 1) * np.log(np.clip(U[i], 1e-10, 1)))
                          for i in range(n)])

    pc = partition_coefficient(U)
    max_ent = float(np.log(c))
    mean_ent = float(entropies.mean())

    # Build cluster summaries
    clusters = []
    for k in range(c):
        mask = hard_assignments == k
        k_docs = [docs[i] for i in range(n) if mask[i]]
        k_metas = [metas[i] for i in range(n) if mask[i]]
        k_memberships = U[mask, k]
        k_entropies = entropies[mask]
        cats = Counter(m.get("source_category", "?") for m in k_metas).most_common(4)

        # Most ambiguous docs in this cluster
        k_indices = np.where(mask)[0]
        sorted_by_entropy = k_indices[np.argsort(entropies[k_indices])[::-1]]
        boundary = []
        for bi in sorted_by_entropy[:3]:
            snippet = docs[bi][:200].replace("\n", " ").replace("<", "&lt;").replace(">", "&gt;")
            # Get top-2 memberships for this doc
            top2_clusters = np.argsort(U[bi])[::-1][:2]
            top2_str = ", ".join(f"C{j}={U[bi,j]:.2f}" for j in top2_clusters)
            boundary.append({
                "snippet": snippet,
                "entropy": round(float(entropies[bi]), 3),
                "memberships": top2_str,
            })

        clusters.append({
            "id": k,
            "count": int(mask.sum()),
            "mean_membership": round(float(k_memberships.mean()), 3),
            "mean_entropy": round(float(k_entropies.mean()), 3),
            "top_cats": cats,
            "terms": top_terms(k_docs[:500]),
            "boundary_docs": boundary,
        })

    # Sort clusters by size descending
    clusters.sort(key=lambda x: x["count"], reverse=True)

    # Generate HTML
    html = _generate_html(clusters, pc, max_ent, mean_ent, n, c, U)

    out = Path(__file__).parent / "cluster_report.html"
    out.write_text(html, encoding="utf-8")
    print(f"[report] ✓ Report written to {out}")
    return out


def _generate_html(clusters, pc, max_ent, mean_ent, n, c, U):
    cluster_cards = ""
    for cl in clusters:
        terms_html = " ".join(
            f'<span class="term">{t}</span>' for t in cl["terms"]
        )
        cats_html = " | ".join(
            f'<span class="cat">{cat} <b>{cnt}</b></span>'
            for cat, cnt in cl["top_cats"]
        )
        boundary_html = ""
        for bd in cl["boundary_docs"]:
            boundary_html += f"""
            <div class="boundary-doc">
              <span class="ent-badge">entropy={bd['entropy']}</span>
              <span class="mem-badge">{bd['memberships']}</span>
              <p>"{bd['snippet']}..."</p>
            </div>"""

        # Membership histogram
        memberships_k = U[:, cl["id"]]
        hist_bins = np.histogram(memberships_k, bins=10, range=(0, 1))[0]
        hist_html = "".join(
            f'<div class="bar" style="height:{int(h/hist_bins.max()*60)}px" title="{h} docs"></div>'
            for h in hist_bins
        )

        cluster_cards += f"""
        <div class="cluster-card">
          <div class="cluster-header">
            <span class="cluster-id">Cluster {cl['id']:02d}</span>
            <span class="cluster-count">{cl['count']:,} docs</span>
            <span class="cluster-entropy">mean entropy: {cl['mean_entropy']}</span>
            <span class="cluster-membership">mean membership: {cl['mean_membership']}</span>
          </div>
          <div class="categories">{cats_html}</div>
          <div class="terms">{terms_html}</div>
          <div class="hist-container">
            <div class="hist-label">Membership distribution (0→1)</div>
            <div class="hist">{hist_html}</div>
          </div>
          <div class="boundary-section">
            <h4>⚠ Boundary Documents (most ambiguous)</h4>
            {boundary_html}
          </div>
        </div>"""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Fuzzy Cluster Analysis — 20 Newsgroups</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: 'Segoe UI', system-ui, sans-serif; background: #0f1117; color: #e2e8f0; padding: 2rem; }}
  h1 {{ font-size: 2rem; font-weight: 700; color: #7dd3fc; margin-bottom: 0.5rem; }}
  h2 {{ font-size: 1.2rem; color: #94a3b8; margin-bottom: 2rem; }}
  .global-stats {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px,1fr)); gap: 1rem; margin-bottom: 2.5rem; }}
  .stat-card {{ background: #1e2130; border: 1px solid #2d3748; border-radius: 12px; padding: 1.2rem; }}
  .stat-label {{ font-size: 0.75rem; color: #64748b; text-transform: uppercase; letter-spacing: .05em; }}
  .stat-value {{ font-size: 2rem; font-weight: 700; color: #7dd3fc; margin-top: 0.3rem; }}
  .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(540px,1fr)); gap: 1.5rem; }}
  .cluster-card {{ background: #1a1d2e; border: 1px solid #2d3748; border-radius: 14px; padding: 1.5rem; transition: transform .15s; }}
  .cluster-card:hover {{ transform: translateY(-2px); border-color: #4a90d9; }}
  .cluster-header {{ display: flex; align-items: center; gap: 1rem; flex-wrap: wrap; margin-bottom: 1rem; }}
  .cluster-id {{ font-size: 1.4rem; font-weight: 800; color: #f0abfc; }}
  .cluster-count {{ background: #2d3748; color: #7dd3fc; padding: .2rem .7rem; border-radius: 20px; font-size: .85rem; }}
  .cluster-entropy, .cluster-membership {{ color: #64748b; font-size: .8rem; }}
  .categories {{ margin-bottom: 0.8rem; }}
  .cat {{ background: #1e3a5f; color: #93c5fd; padding: .15rem .5rem; border-radius: 6px; margin: .2rem; display: inline-block; font-size: .8rem; }}
  .terms {{ margin-bottom: 1rem; }}
  .term {{ background: #1a2e1a; color: #86efac; padding: .15rem .5rem; border-radius: 6px; margin: .2rem; display: inline-block; font-size: .8rem; cursor: default; }}
  .hist-container {{ margin-bottom: 1rem; }}
  .hist-label {{ font-size: .7rem; color: #64748b; margin-bottom: .3rem; }}
  .hist {{ display: flex; align-items: flex-end; gap: 2px; height: 64px; }}
  .bar {{ background: linear-gradient(to top, #3b82f6, #7dd3fc); border-radius: 2px 2px 0 0; flex: 1; min-height: 2px; }}
  .boundary-section h4 {{ font-size: .85rem; color: #fb923c; margin-bottom: .6rem; }}
  .boundary-doc {{ background: #111827; border-left: 3px solid #fb923c; padding: .7rem; margin-bottom: .5rem; border-radius: 0 6px 6px 0; }}
  .boundary-doc p {{ font-size: .82rem; color: #94a3b8; margin-top: .3rem; font-style: italic; }}
  .ent-badge {{ background: #7c2d12; color: #fed7aa; padding: .1rem .4rem; border-radius: 4px; font-size: .75rem; margin-right: .4rem; }}
  .mem-badge {{ background: #1e3a5f; color: #93c5fd; padding: .1rem .4rem; border-radius: 4px; font-size: .75rem; }}
  footer {{ margin-top: 3rem; color: #374151; font-size: .8rem; text-align: center; }}
</style>
</head>
<body>
<h1>🔬 Fuzzy Cluster Analysis</h1>
<h2>20 Newsgroups corpus · FCM with c={c} clusters · Sentence embeddings (all-MiniLM-L6-v2)</h2>

<div class="global-stats">
  <div class="stat-card">
    <div class="stat-label">Documents</div>
    <div class="stat-value">{n:,}</div>
  </div>
  <div class="stat-card">
    <div class="stat-label">Clusters</div>
    <div class="stat-value">{c}</div>
  </div>
  <div class="stat-card">
    <div class="stat-label">Partition Coefficient</div>
    <div class="stat-value">{pc:.3f}</div>
  </div>
  <div class="stat-card">
    <div class="stat-label">Mean Entropy (max={max_ent:.2f})</div>
    <div class="stat-value">{mean_ent:.3f}</div>
  </div>
  <div class="stat-card">
    <div class="stat-label">Docs with dominant > 0.5</div>
    <div class="stat-value">{(U.max(axis=1)>0.5).mean():.1%}</div>
  </div>
  <div class="stat-card">
    <div class="stat-label">Boundary docs (max < 0.4)</div>
    <div class="stat-value">{(U.max(axis=1)<0.4).mean():.1%}</div>
  </div>
</div>

<div class="grid">
{cluster_cards}
</div>

<footer>Generated by scripts/cluster.py · Trademarkia ML Assignment</footer>
</body>
</html>"""


if __name__ == "__main__":
    generate_report()
