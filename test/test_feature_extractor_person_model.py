from __future__ import annotations

from pathlib import Path
import sys
import types
import unittest
from unittest import mock

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.tools import feature_extractor as feature_extractor_mod
from src.tools.feature_extractor import FeatureExtractorConfig, FeatureMode


class _FakeArcFace:
    def __init__(self, weight_path: str, device: str):
        self.weight_path = weight_path
        self.device = device


class _FakeOSNet(torch.nn.Module):
    feature_dim = 512

    def __init__(self) -> None:
        super().__init__()
        self.input_shape: tuple[int, ...] | None = None
        self.eval_called = False

    def eval(self) -> "_FakeOSNet":
        self.eval_called = True
        return super().eval()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.input_shape = tuple(x.shape)
        return torch.ones((x.shape[0], self.feature_dim), dtype=torch.float32, device=x.device)


class FeatureExtractorPersonModelTest(unittest.TestCase):
    def test_osnet_x1_0_uses_torchreid_pretrained_and_reid_input_shape(self) -> None:
        captured: dict[str, object] = {}

        def build_model(**kwargs: object) -> _FakeOSNet:
            captured.update(kwargs)
            return _FakeOSNet()

        fake_torchreid = types.SimpleNamespace(models=types.SimpleNamespace(build_model=build_model))

        with mock.patch.object(feature_extractor_mod, "ArcFace", _FakeArcFace):
            with mock.patch.dict(sys.modules, {"torchreid": fake_torchreid}):
                extractor = feature_extractor_mod.FeatureExtractor(
                    FeatureExtractorConfig(
                        arcface_weight_path="./models/weights/arcface.pt",
                        device="cpu",
                        person_model="osnet",
                    )
                )
                feat = extractor.extract(FeatureMode.PERSON, np.zeros((300, 180, 3), dtype=np.uint8))

        self.assertEqual(captured["name"], "osnet_x1_0")
        self.assertEqual(captured["num_classes"], 1000)
        self.assertEqual(captured["loss"], "softmax")
        self.assertIs(captured["pretrained"], True)
        self.assertIs(captured["use_gpu"], False)
        self.assertIsNotNone(feat)
        self.assertEqual(feat.shape, (512,))
        self.assertAlmostEqual(float(np.linalg.norm(feat)), 1.0, places=5)
        self.assertIsInstance(extractor.person_model, _FakeOSNet)
        self.assertTrue(extractor.person_model.eval_called)
        self.assertEqual(extractor.person_model.input_shape, (1, 3, 256, 128))


if __name__ == "__main__":
    unittest.main()
