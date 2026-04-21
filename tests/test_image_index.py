import csv
import os
import tempfile
import unittest

import numpy as np

from retrieval.image_index import ImageIndex
from retrieval.math_utils import l2_normalize_rows


class ImageIndexTest(unittest.TestCase):
    def test_search_ranking_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            feats = np.array(
                [
                    [1.0, 0.0, 0.0],
                    [0.8, 0.6, 0.0],
                    [-1.0, 0.0, 0.0],
                ],
                dtype=np.float32,
            )
            np.save(os.path.join(tmp, "features.npy"), l2_normalize_rows(feats))

            meta_path = os.path.join(tmp, "meta.csv")
            with open(meta_path, "w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=["image_id", "image_path", "identity", "x", "y", "w", "h"],
                )
                writer.writeheader()
                writer.writerows(
                    [
                        {"image_id": "a", "image_path": "a.jpg", "identity": "p1", "x": "", "y": "", "w": "", "h": ""},
                        {"image_id": "b", "image_path": "b.jpg", "identity": "p2", "x": "", "y": "", "w": "", "h": ""},
                        {"image_id": "c", "image_path": "c.jpg", "identity": "p3", "x": "", "y": "", "w": "", "h": ""},
                    ]
                )

            idx = ImageIndex(tmp)
            query = np.array([1.0, 0.0, 0.0], dtype=np.float32)
            res = idx.search_by_feature(query_feat=query, topk=2, threshold=0.0)
            self.assertEqual([r["image_id"] for r in res], ["a", "b"])


if __name__ == "__main__":
    unittest.main()
