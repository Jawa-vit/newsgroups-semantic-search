#!/usr/bin/env python3
"""
tests/test_all.py — Test suite covering all major components.

Run with:
    pytest tests/test_all.py -v

Or without pytest:
    python tests/test_all.py
"""

import sys
import unittest
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ── Test Fuzzy C-Means ─────────────────────────────────────────────────────────

class TestFCM(unittest.TestCase):

    def setUp(self):
        from app.fuzzy_cluster import fuzzy_cmeans, partition_coefficient, FuzzyClustering
        self.fuzzy_cmeans = fuzzy_cmeans
        self.partition_coefficient = partition_coefficient
        self.FuzzyClustering = FuzzyClustering

    def test_membership_sums_to_one(self):
        """Each document's membership vector must sum to 1."""
        rng = np.random.default_rng(42)
        X = rng.standard_normal((100, 10))
        U, centers, _ = self.fuzzy_cmeans(X, c=5, m=2.0, max_iter=50)
        np.testing.assert_allclose(U.sum(axis=1), np.ones(100), atol=1e-5,
                                   err_msg="Memberships don't sum to 1")

    def test_membership_bounds(self):
        """All memberships must be in [0, 1]."""
        rng = np.random.default_rng(0)
        X = rng.standard_normal((50, 8))
        U, _, _ = self.fuzzy_cmeans(X, c=4, m=2.0, max_iter=30)
        self.assertTrue((U >= 0).all(), "Negative membership found")
        self.assertTrue((U <= 1 + 1e-6).all(), "Membership > 1 found")

    def test_partition_coefficient_bounds(self):
        """PC must be in [1/c, 1]."""
        rng = np.random.default_rng(1)
        X = rng.standard_normal((80, 6))
        c = 5
        U, _, _ = self.fuzzy_cmeans(X, c=c, max_iter=50)
        pc = self.partition_coefficient(U)
        self.assertGreaterEqual(pc, 1.0/c - 1e-6)
        self.assertLessEqual(pc, 1.0 + 1e-6)

    def test_center_count(self):
        """Number of centres must equal c."""
        rng = np.random.default_rng(2)
        X = rng.standard_normal((60, 5))
        c = 7
        _, centers, _ = self.fuzzy_cmeans(X, c=c, max_iter=30)
        self.assertEqual(centers.shape[0], c)
        self.assertEqual(centers.shape[1], X.shape[1])

    def test_clustering_separates_obvious_clusters(self):
        """FCM should assign high membership to the correct cluster for well-separated data."""
        rng = np.random.default_rng(42)
        # Two very clearly separated 2D clusters
        X1 = rng.standard_normal((50, 2)) + np.array([10.0, 0.0])
        X2 = rng.standard_normal((50, 2)) + np.array([-10.0, 0.0])
        X = np.vstack([X1, X2])
        U, _, _ = self.fuzzy_cmeans(X, c=2, m=2.0, max_iter=100)
        # Each point should have dominant membership > 0.9
        dominant = U.max(axis=1)
        self.assertTrue((dominant > 0.9).mean() > 0.9,
                        "Expected high-confidence assignments for well-separated clusters")


# ── Test Semantic Cache ────────────────────────────────────────────────────────

class TestSemanticCache(unittest.TestCase):

    def setUp(self):
        from app.semantic_cache import SemanticCache
        self.SemanticCache = SemanticCache

    def _make_embedding(self, seed: int) -> np.ndarray:
        """Make a deterministic unit-norm embedding."""
        rng = np.random.default_rng(seed)
        v = rng.standard_normal(384).astype(np.float32)
        return v / np.linalg.norm(v)

    def test_empty_cache_miss(self):
        """Fresh cache should always miss."""
        cache = self.SemanticCache(threshold=0.85, n_clusters=5)
        emb = self._make_embedding(0)
        entry, sim = cache.lookup(emb, [(0, 1.0)])
        self.assertIsNone(entry)
        self.assertEqual(cache.miss_count, 1)
        self.assertEqual(cache.hit_count, 0)

    def test_exact_hit(self):
        """Looking up the exact same embedding should always hit."""
        cache = self.SemanticCache(threshold=0.85, n_clusters=5)
        emb = self._make_embedding(1)
        cache.store("test query", emb, {"result": "ok"}, dominant_cluster=0)
        entry, sim = cache.lookup(emb, [(0, 1.0)])
        self.assertIsNotNone(entry)
        self.assertAlmostEqual(sim, 1.0, places=4)
        self.assertEqual(cache.hit_count, 1)

    def test_threshold_respected(self):
        """Below-threshold similarity should be a miss."""
        cache = self.SemanticCache(threshold=0.99, n_clusters=5)  # very strict
        emb1 = self._make_embedding(10)
        emb2 = self._make_embedding(11)  # different random vector
        cache.store("query 1", emb1, {"result": "a"}, dominant_cluster=0)
        entry, sim = cache.lookup(emb2, [(0, 1.0)])
        # Two random unit vectors in 384-d space have cos ≈ 0 (very unlikely to be > 0.99)
        self.assertIsNone(entry, f"Expected miss but got hit with sim={sim}")

    def test_flush_resets_state(self):
        """After flush, cache should be empty with zero stats."""
        cache = self.SemanticCache(threshold=0.85, n_clusters=5)
        emb = self._make_embedding(20)
        cache.store("q", emb, {}, dominant_cluster=0)
        cache.lookup(emb, [(0, 1.0)])  # hit
        cache.flush()
        self.assertEqual(cache.total_entries, 0)
        self.assertEqual(cache.hit_count, 0)
        self.assertEqual(cache.miss_count, 0)

    def test_stats_consistency(self):
        """hit_count + miss_count must equal total lookups."""
        cache = self.SemanticCache(threshold=0.85, n_clusters=5)
        emb = self._make_embedding(30)
        cache.store("q", emb, {}, dominant_cluster=0)
        for _ in range(3):
            cache.lookup(emb, [(0, 1.0)])   # hits
        for seed in range(40, 45):
            cache.lookup(self._make_embedding(seed), [(0, 1.0)])  # misses
        total_lookups = cache.hit_count + cache.miss_count
        self.assertEqual(total_lookups, 8)

    def test_hot_threshold_change(self):
        """Changing threshold should affect subsequent lookups."""
        cache = self.SemanticCache(threshold=0.99, n_clusters=5)
        emb1 = self._make_embedding(50)
        emb2 = self._make_embedding(50)  # same seed = same embedding = cos=1.0
        # small perturbation
        emb2 = emb1 + np.random.default_rng(99).standard_normal(384).astype(np.float32) * 0.02
        emb2 = emb2 / np.linalg.norm(emb2)
        cache.store("q", emb1, {}, dominant_cluster=0)
        entry, _ = cache.lookup(emb2, [(0, 1.0)])
        # Might miss at 0.99 threshold
        cache.set_threshold(0.70)  # relax threshold
        cache.flush()
        cache.store("q", emb1, {}, dominant_cluster=0)
        entry2, sim2 = cache.lookup(emb2, [(0, 1.0)])
        # Should hit now (cos of slightly perturbed vector is very close to 1)
        self.assertIsNotNone(entry2, f"Expected hit at threshold=0.70, got miss with sim={sim2}")

    def test_hit_rate_calculation(self):
        """hit_rate = hits / (hits + misses)."""
        cache = self.SemanticCache(threshold=0.85, n_clusters=5)
        emb = self._make_embedding(60)
        cache.store("q", emb, {}, dominant_cluster=0)
        cache.lookup(emb, [(0, 1.0)])   # hit
        cache.lookup(emb, [(0, 1.0)])   # hit
        cache.lookup(self._make_embedding(999), [(0, 1.0)])  # miss
        self.assertAlmostEqual(cache.hit_rate, 2/3, places=3)


# ── Test text cleaning (ingest) ────────────────────────────────────────────────

class TestIngestion(unittest.TestCase):

    def setUp(self):
        from scripts.ingest import _parse_post
        self._parse_post = _parse_post

    def test_strips_headers(self):
        """From:, Message-ID: etc. should not appear in cleaned text."""
        raw = (
            "From: user@example.com\n"
            "Message-ID: <abc123>\n"
            "Subject: Space exploration news\n"
            "\n"
            "NASA announced a new mission to Mars today.\n"
        )
        result = self._parse_post(raw)
        self.assertNotIn("user@example.com", result["text"])
        self.assertNotIn("abc123", result["text"])
        self.assertIn("NASA", result["text"])

    def test_strips_quotes(self):
        """Lines starting with '>' should be removed."""
        raw = (
            "Subject: Re: Rockets\n"
            "\n"
            "> The old rocket was bad\n"
            "Actually the new one is great.\n"
        )
        result = self._parse_post(raw)
        self.assertNotIn("The old rocket was bad", result["text"])
        self.assertIn("great", result["text"])

    def test_strips_signature(self):
        """Content after '-- ' should be stripped."""
        raw = (
            "Subject: Test\n"
            "\n"
            "Some content here.\n"
            "--\n"
            "John Doe | Company | +1-555-0000\n"
        )
        result = self._parse_post(raw)
        self.assertNotIn("John Doe", result["text"])
        self.assertIn("content", result["text"])

    def test_truncates_long_text(self):
        """Text longer than 2000 chars should be truncated."""
        raw = "Subject: Long\n\n" + "x" * 3000
        result = self._parse_post(raw)
        self.assertLessEqual(len(result["text"]), 2001)


# ── Runner ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Running test suite ...\n")
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    for cls in [TestFCM, TestSemanticCache, TestIngestion]:
        suite.addTests(loader.loadTestsFromTestCase(cls))
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
