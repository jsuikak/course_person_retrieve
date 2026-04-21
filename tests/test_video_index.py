import csv
import os
import tempfile
import unittest

import numpy as np

from retrieval.math_utils import l2_normalize_rows
from retrieval.video_index import VideoIndex


class VideoIndexTest(unittest.TestCase):
    def test_empty_index_search_returns_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            np.save(os.path.join(tmp, "features.npy"), np.empty((0, 512), dtype=np.float32))
            with open(os.path.join(tmp, "meta.csv"), "w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=[
                        "video_id",
                        "video_path",
                        "identity",
                        "frame_idx",
                        "timestamp",
                        "track_id",
                        "x",
                        "y",
                        "w",
                        "h",
                    ],
                )
                writer.writeheader()

            idx = VideoIndex(tmp)
            query = np.zeros((512,), dtype=np.float32)
            res = idx.search_by_feature(query_feat=query, topk=5, threshold=0.3)
            self.assertEqual(res, [])

    def test_video_rank_uses_track_top3_and_video_max(self):
        with tempfile.TemporaryDirectory() as tmp:
            feats = np.array(
                [
                    [0.9, 0.1],
                    [0.8, 0.2],
                    [0.4, 0.6],
                    [0.95, 0.05],
                ],
                dtype=np.float32,
            )
            np.save(os.path.join(tmp, "features.npy"), l2_normalize_rows(feats))

            with open(os.path.join(tmp, "meta.csv"), "w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=[
                        "video_id",
                        "video_path",
                        "identity",
                        "frame_idx",
                        "timestamp",
                        "track_id",
                        "x",
                        "y",
                        "w",
                        "h",
                    ],
                )
                writer.writeheader()
                writer.writerows(
                    [
                        {"video_id": "v1", "video_path": "v1.mp4", "identity": "p1", "frame_idx": 0, "timestamp": 0.0, "track_id": "0", "x": 0, "y": 0, "w": 1, "h": 1},
                        {"video_id": "v1", "video_path": "v1.mp4", "identity": "p1", "frame_idx": 1, "timestamp": 1.0, "track_id": "0", "x": 0, "y": 0, "w": 1, "h": 1},
                        {"video_id": "v1", "video_path": "v1.mp4", "identity": "p1", "frame_idx": 2, "timestamp": 2.0, "track_id": "1", "x": 0, "y": 0, "w": 1, "h": 1},
                        {"video_id": "v2", "video_path": "v2.mp4", "identity": "p2", "frame_idx": 0, "timestamp": 0.0, "track_id": "0", "x": 0, "y": 0, "w": 1, "h": 1},
                    ]
                )

            idx = VideoIndex(tmp)
            query = np.array([1.0, 0.0], dtype=np.float32)
            res = idx.search_by_feature(query_feat=query, topk=2, threshold=0.0)
            self.assertEqual(res[0]["video_id"], "v2")
            self.assertEqual(res[1]["video_id"], "v1")


if __name__ == "__main__":
    unittest.main()
