"""
embedder.py — Singleton wrapper around SentenceTransformer.

Keeps a single model instance in memory across the full FastAPI lifecycle.
Loading SentenceTransformer is expensive (~1–2 s); we never want to do it
per-request.
"""

from __future__ import annotations
import numpy as np
from sentence_transformers import SentenceTransformer
from app.config import EMBEDDING_MODEL, EMBEDDING_DIM


class Embedder:
    """Thread-safe singleton embedding model."""

    _instance: Embedder | None = None
    _model: SentenceTransformer | None = None

    def __new__(cls) -> "Embedder":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._model = SentenceTransformer(EMBEDDING_MODEL)
        return cls._instance

    def embed(self, text: str) -> np.ndarray:
        """Embed a single string → L2-normalised 384-d vector."""
        vec = self._model.encode(text, normalize_embeddings=True, show_progress_bar=False)
        return vec.astype(np.float32)

    def embed_batch(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        """
        Embed a list of strings → (N, 384) float32 array.

        normalize_embeddings=True means cosine similarity == dot product,
        which is what ChromaDB's cosine space and our cache comparisons use.
        """
        vecs = self._model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=True,
        )
        return vecs.astype(np.float32)

    @property
    def dim(self) -> int:
        return EMBEDDING_DIM


# Module-level singleton — import this everywhere
embedder = Embedder()
