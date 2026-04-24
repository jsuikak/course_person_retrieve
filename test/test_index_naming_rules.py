"""索引命名规则测试：仅支持 *_face/*_person 新命名。"""

from __future__ import annotations

import csv
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.retrieval import _find_index_files
from src.app_retrieval import resolve_effective_index_name
from src.tools.feature_extractor import FeatureMode


def _write_meta(meta_path: Path) -> None:
    with meta_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["row_id", "source_type", "source_name", "frame_index", "face_index", "x", "y", "w", "h"])
        writer.writerow([0, "image", "dummy.jpg", -1, 0, 0, 0, 10, 10])


class IndexNamingRulesTest(unittest.TestCase):
    def test_find_face_and_person_suffix_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            face_feat = root / "demo_face_features.npy"
            face_meta = root / "demo_face_meta.csv"
            person_feat = root / "demo_person_features.npy"
            person_meta = root / "demo_person_meta.csv"

            np.save(face_feat, np.zeros((1, 4), dtype=np.float32))
            np.save(person_feat, np.zeros((1, 8), dtype=np.float32))
            _write_meta(face_meta)
            _write_meta(person_meta)

            f_path, f_meta, _ = _find_index_files("demo", str(root), FeatureMode.FACE)
            p_path, p_meta, _ = _find_index_files("demo", str(root), FeatureMode.PERSON)

            self.assertEqual(f_path, face_feat)
            self.assertEqual(f_meta, face_meta)
            self.assertEqual(p_path, person_feat)
            self.assertEqual(p_meta, person_meta)

    def test_old_naming_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            old_feat = root / "demo_features.npy"
            old_meta = root / "demo_meta.csv"
            np.save(old_feat, np.zeros((1, 4), dtype=np.float32))
            _write_meta(old_meta)

            with self.assertRaises(FileNotFoundError):
                _find_index_files("demo", str(root), FeatureMode.FACE)

    def test_person_index_names_include_model_key(self) -> None:
        self.assertEqual(
            resolve_effective_index_name("demo", "person", "resnet", "resnet18"),
            "demo_resnet18",
        )
        self.assertEqual(
            resolve_effective_index_name("demo", "person", "osnet", "resnet18"),
            "demo_osnet_x1_0",
        )
        self.assertEqual(
            resolve_effective_index_name("demo_osnet_x1_0", "person", "osnet", "resnet18"),
            "demo_osnet_x1_0",
        )

    def test_face_index_name_does_not_include_person_model_key(self) -> None:
        self.assertEqual(
            resolve_effective_index_name("demo", "face", "osnet", "resnet18"),
            "demo",
        )


if __name__ == "__main__":
    unittest.main()
