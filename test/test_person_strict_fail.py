"""person 模式严格失败策略测试。"""

from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.face_index_builder import build_feature_index
from src.person_feature_pipeline import PersonFeatureBundle


class PersonStrictFailTest(unittest.TestCase):
    def test_person_mode_raises_when_gallery_has_no_valid_person_features(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            image_path = tmp_path / "dummy.jpg"
            cv2.imwrite(str(image_path), np.zeros((32, 32, 3), dtype=np.uint8))

            with mock.patch("src.face_index_builder._extract_person_bundle", return_value=PersonFeatureBundle()):
                with self.assertRaises(ValueError):
                    build_feature_index(
                        library_path=str(image_path),
                        output_dir=str(tmp_path / "index"),
                        arcface_weight_path="./models/weights/arcface.pt",
                        feature_mode="person",
                        library_type="image",
                        prefix="demo",
                        device="cpu",
                    )


if __name__ == "__main__":
    unittest.main()

