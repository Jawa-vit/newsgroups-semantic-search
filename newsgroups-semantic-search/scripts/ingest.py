#!/usr/bin/env python3
"""
scripts/ingest.py — Parse the 20 Newsgroups corpus, clean it, embed it,
                    and persist it in ChromaDB.

Run once before starting the API:
    python scripts/ingest.py

Design decisions on cleaning
─────────────────────────────
The raw newsgroup posts are extremely noisy. Poor cleaning degrades embedding
quality and cascades into bad cluster assignments and poor search recall.

We deliberately REMOVE:
  1. Email headers (From:, Subject:, Message-ID:, etc.)
     — These are metadata, not semantic content. Including them would cause
       the embedding model to cluster posts by author domain, not topic.

  2. Quoted reply text (lines starting with ">")
     — Quoted text biases the embedding toward the *previous* post's topic,
       not the current one. A reply to a baseball thread in the politics
       group should cluster with politics.

  3. Signature blocks ("-- " separator and everything after)
     — Pure noise. Signatures often contain location info, phone numbers,
       and company boilerplate that would pollute cluster centroids.

  4. Documents shorter than 50 characters after cleaning
     — These are almost always failed parses or empty forwards. They add
       no semantic content and hurt cluster quality.

  5. Documents longer than 2000 characters (truncated, not removed)
     — SentenceTransformer models have a 256-token context window. Text
       beyond ~1500 chars is silently truncated by the tokeniser anyway.
       We take the first 2000 chars explicitly so users know what was embedded.

We deliberately KEEP:
  • Subject lines — the most information-dense part of the post
  • Body text after stripping the above artifacts
  • Original category labels — stored as metadata for filtered retrieval
"""

import sys
import os
import re
import json
from pathlib import Path
from tqdm import tqdm
import numpy as np

# Ensure project root is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.embedder import embedder
from app.vector_store import vector_store
from app.config import DATA_DIR

# ── Newsgroup categories ───────────────────────────────────────────────────────
CATEGORIES = [
    "alt.atheism",
    "comp.graphics",
    "comp.os.ms-windows.misc",
    "comp.sys.ibm.pc.hardware",
    "comp.sys.mac.hardware",
    "comp.windows.x",
    "misc.forsale",
    "rec.autos",
    "rec.motorcycles",
    "rec.sport.baseball",
    "rec.sport.hockey",
    "sci.crypt",
    "sci.electronics",
    "sci.med",
    "sci.space",
    "soc.religion.christian",
    "talk.politics.guns",
    "talk.politics.mideast",
    "talk.politics.misc",
    "talk.religion.misc",
]

# Header fields that carry zero semantic content
_HEADER_PATTERN = re.compile(
    r"^(Xref|Path|From|Newsgroups|Subject|Message-ID|Date|References|"
    r"Organization|Lines|Followup-To|Distribution|Keywords|Summary|"
    r"Approved|Expires|Supersedes|Reply-To|Archive-name|"
    r"Alt-atheism-archive-name|Last-modified|Version)\s*:",
    re.IGNORECASE,
)


def _parse_post(raw: str) -> dict:
    """
    Parse a single raw newsgroup post into subject + clean body.

    Returns a dict with keys: subject, body, text (combined).
    """
    lines = raw.splitlines()
    subject = ""
    body_lines = []
    in_header = True
    in_signature = False

    for line in lines:
        # Header parsing phase
        if in_header:
            if line.strip() == "":
                # First blank line ends the header
                in_header = False
                continue
            m = re.match(r"^Subject:\s*(.+)", line, re.IGNORECASE)
            if m:
                subject = m.group(1).strip()
            # Skip all other header lines
            continue

        # Body parsing phase
        if line.strip() == "--":
            # Signature separator — stop here
            in_signature = True
            continue
        if in_signature:
            continue
        if line.startswith(">"):
            # Quoted reply — skip
            continue
        body_lines.append(line)

    body = "\n".join(body_lines).strip()
    # Collapse excessive blank lines
    body = re.sub(r"\n{3,}", "\n\n", body)
    # Remove lines that are just punctuation/whitespace
    body = "\n".join(
        l for l in body.splitlines() if len(l.strip()) > 2
    )

    combined = f"{subject}\n\n{body}".strip() if subject else body
    # Truncate to 2000 chars (model context limit; see module docstring)
    combined = combined[:2000]

    return {"subject": subject, "body": body[:1500], "text": combined}


def load_corpus() -> list[dict]:
    """
    Walk the data directory, parse every post, and return clean records.

    Returns list of:
      {doc_id, category, subject, body, text}
    """
    records = []
    total_raw = 0
    total_kept = 0

    for category in CATEGORIES:
        cat_dir = DATA_DIR / category
        if not cat_dir.exists():
            print(f"  WARNING: {cat_dir} not found, skipping.")
            continue

        files = sorted(cat_dir.iterdir())
        for fpath in files:
            if not fpath.is_file():
                continue
            total_raw += 1
            try:
                raw = fpath.read_text(encoding="latin-1", errors="replace")
            except Exception:
                continue

            parsed = _parse_post(raw)

            # Quality gate: discard very short documents
            if len(parsed["text"]) < 50:
                continue

            total_kept += 1
            records.append(
                {
                    "doc_id": f"{category}_{fpath.name}",
                    "category": category,
                    "subject": parsed["subject"],
                    "body": parsed["body"],
                    "text": parsed["text"],
                }
            )

    print(f"\n[ingest] Parsed {total_raw} raw posts → kept {total_kept} "
          f"({100*total_kept/total_raw:.1f}%) after quality filtering")
    return records


def ingest(batch_size: int = 128, force: bool = False) -> None:
    """Embed corpus and upsert into ChromaDB."""

    # Skip if already indexed
    if not force and not vector_store.is_empty():
        count = vector_store.count()
        print(f"[ingest] Vector store already contains {count:,} documents.")
        print("[ingest] Pass --force to re-index.")
        return

    print("[ingest] Loading and cleaning corpus ...")
    records = load_corpus()
    texts = [r["text"] for r in records]
    print(f"[ingest] Embedding {len(texts):,} documents in batches of {batch_size} ...")

    embeddings = embedder.embed_batch(texts, batch_size=batch_size)

    # Upsert to ChromaDB in batches (ChromaDB has a ~5461 item batch limit)
    chroma_batch = 500
    print("[ingest] Writing to ChromaDB ...")
    for start in tqdm(range(0, len(records), chroma_batch), desc="ChromaDB upsert"):
        end = min(start + chroma_batch, len(records))
        batch_records = records[start:end]
        batch_embeddings = embeddings[start:end]

        ids = [r["doc_id"] for r in batch_records]
        documents = [r["text"] for r in batch_records]
        metadatas = [
            {
                "source_category": r["category"],
                # dominant_cluster will be set after clustering in scripts/cluster.py
                # We initialise to -1 as a sentinel so the vector store is usable
                # even before clustering runs.
                "dominant_cluster": -1,
            }
            for r in batch_records
        ]
        vector_store.add_documents(ids, batch_embeddings, documents, metadatas)

    print(f"\n[ingest] ✓ Indexed {len(records):,} documents into ChromaDB.")
    print(f"[ingest] Next: run `python scripts/cluster.py` to add cluster assignments.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Ingest 20 Newsgroups into ChromaDB")
    parser.add_argument("--force", action="store_true", help="Re-index even if DB is populated")
    parser.add_argument("--batch-size", type=int, default=128, help="Embedding batch size")
    args = parser.parse_args()
    ingest(batch_size=args.batch_size, force=args.force)
