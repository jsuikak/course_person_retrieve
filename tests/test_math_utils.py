import unittest

import numpy as np

from retrieval.math_utils import l2_normalize, mean_topk, topk_indices


class MathUtilsTest(unittest.TestCase):
    def test_l2_normalize_norm_is_one(self):
        vec = np.array([3.0, 4.0], dtype=np.float32)
        out = l2_normalize(vec)
        self.assertAlmostEqual(float(np.linalg.norm(out)), 1.0, places=6)

    def test_topk_indices_monotonic(self):
        scores = np.array([0.3, 0.9, 0.5, 0.1], dtype=np.float32)
        idx = topk_indices(scores, k=3)
        ranked_scores = scores[idx]
        self.assertTrue(np.all(ranked_scores[:-1] >= ranked_scores[1:]))
        self.assertListEqual(idx.tolist(), [1, 2, 0])

    def test_mean_topk_short_track(self):
        scores = [0.9, 0.8]
        self.assertAlmostEqual(mean_topk(scores, k=3), 0.85, places=6)


if __name__ == "__main__":
    unittest.main()
