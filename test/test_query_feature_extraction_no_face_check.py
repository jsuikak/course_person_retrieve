"""query 特征提取不做人脸存在性判断的回归测试。"""

from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import cv2
import csv
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import retrieval
from src.tools.feature_extractor import FeatureMode


class _FakeExtractor:
    configs: list[object] = []

    def __init__(self, config: object):
        self.config = config
        _FakeExtractor.configs.append(config)

    def extract(self, mode: FeatureMode, image_bgr: np.ndarray) -> np.ndarray | None:
        if image_bgr is None or image_bgr.size == 0:
            return None
        if mode == FeatureMode.FACE:
            return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        if mode == FeatureMode.PERSON:
            return np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
        return None


class QueryFeatureExtractionNoFaceCheckTest(unittest.TestCase):
    def _write_query_image(self, tmp_dir: Path, name: str = "query.jpg") -> Path:
        path = tmp_dir / name
        image = np.zeros((32, 48, 3), dtype=np.uint8)
        ok = cv2.imwrite(str(path), image)
        self.assertTrue(ok)
        return path

    def test_face_query_feature_uses_whole_image_without_face_detection(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            query_file = self._write_query_image(tmp_dir)
            _FakeExtractor.configs.clear()

            with mock.patch("src.retrieval.FeatureExtractor", _FakeExtractor):
                feat, bbox = retrieval._extract_face_query_feature(
                    query_file=query_file,
                    arcface_weight_path="./models/weights/arcface.pt",
                    device="cpu",
                )

            self.assertEqual(feat.shape, (1, 4))
            self.assertEqual(bbox, {"x": 0, "y": 0, "w": 48, "h": 32})
            self.assertGreaterEqual(len(_FakeExtractor.configs), 1)
            self.assertFalse(bool(_FakeExtractor.configs[-1].detect_face))

    def test_person_query_feature_keeps_detect_face_disabled(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            query_file = self._write_query_image(tmp_dir, "person_query.jpg")
            _FakeExtractor.configs.clear()

            with mock.patch("src.retrieval.FeatureExtractor", _FakeExtractor):
                feat = retrieval._extract_person_query_feature(
                    query_file=query_file,
                    arcface_weight_path="./models/weights/arcface.pt",
                    device="cpu",
                    person_model="resnet",
                    resnet_backbone="resnet18",
                    resnet_pretrained=False,
                    resnet_weight_path=None,
                    person_input_size=224,
                )

            self.assertEqual(feat.shape, (1, 4))
            self.assertGreaterEqual(len(_FakeExtractor.configs), 1)
            self.assertFalse(bool(_FakeExtractor.configs[-1].detect_face))

    def test_search_rejects_mismatched_person_feature_dimensions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            query_file = self._write_query_image(tmp_dir, "person_query.jpg")
            features_path = tmp_dir / "demo_person_features.npy"
            meta_path = tmp_dir / "demo_person_meta.csv"
            np.save(features_path, np.zeros((1, 8), dtype=np.float32))
            with meta_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["row_id", "source_type", "source_name", "frame_index", "person_index", "x", "y", "w", "h"])
                writer.writerow([0, "image", "dummy.jpg", -1, 0, 0, 0, 10, 10])

            with mock.patch("src.retrieval.FeatureExtractor", _FakeExtractor):
                with self.assertRaisesRegex(ValueError, "feature dimension mismatch"):
                    retrieval.search_query_in_index(
                        query_path=str(query_file),
                        index_name="demo",
                        indexes_root=str(tmp_dir),
                        retrieval_output_root=str(tmp_dir / "out"),
                        feature_mode="person",
                        person_model="resnet",
                        resnet_backbone="resnet18",
                    )


if __name__ == "__main__":
    unittest.main()
