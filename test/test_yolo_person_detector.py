"""YOLO 人体检测接口单测（通过 fake ultralytics 模块隔离真实依赖）。"""

from __future__ import annotations

import importlib
from pathlib import Path
import sys
import types
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class _FakeTensor:
    def __init__(self, value: object):
        self._array = np.asarray(value, dtype=np.float32)

    def cpu(self) -> "_FakeTensor":
        return self

    def numpy(self) -> np.ndarray:
        return self._array


class _FakeBoxes:
    def __init__(self, xyxy: object, conf: object, cls: object):
        self.xyxy = _FakeTensor(xyxy)
        self.conf = _FakeTensor(conf)
        self.cls = _FakeTensor(cls)


class _FakeResult:
    def __init__(self, boxes: _FakeBoxes):
        self.boxes = boxes


class _FakeYOLO:
    predictions_per_call: list[dict[str, object]] = []
    last_predict_kwargs: dict[str, object] = {}

    def __init__(self, weights: str):
        self.weights = weights

    def predict(self, **kwargs: object) -> list[_FakeResult]:
        _FakeYOLO.last_predict_kwargs = dict(kwargs)
        if _FakeYOLO.predictions_per_call:
            pred = _FakeYOLO.predictions_per_call.pop(0)
        else:
            pred = {"xyxy": [], "conf": [], "cls": []}
        return [
            _FakeResult(
                _FakeBoxes(
                    xyxy=pred.get("xyxy", []),
                    conf=pred.get("conf", []),
                    cls=pred.get("cls", []),
                )
            )
        ]


class YOLOPersonDetectorTest(unittest.TestCase):
    def setUp(self) -> None:
        self._orig_ultralytics = sys.modules.get("ultralytics")
        fake_module = types.ModuleType("ultralytics")
        fake_module.YOLO = _FakeYOLO
        sys.modules["ultralytics"] = fake_module
        sys.modules.pop("src.tools.yolo_person_detector", None)
        self.mod = importlib.import_module("src.tools.yolo_person_detector")

    def tearDown(self) -> None:
        sys.modules.pop("src.tools.yolo_person_detector", None)
        if self._orig_ultralytics is None:
            sys.modules.pop("ultralytics", None)
        else:
            sys.modules["ultralytics"] = self._orig_ultralytics

    def test_empty_image_returns_empty_list(self) -> None:
        _FakeYOLO.predictions_per_call = [
            {"xyxy": [[1, 1, 20, 20]], "conf": [0.9], "cls": [0]},
        ]
        detector = self.mod.YOLOPersonDetector(self.mod.YOLOPersonDetectorConfig())
        image = np.empty((0, 0, 3), dtype=np.uint8)
        boxes = detector.detect(image)
        self.assertEqual(boxes, [])

    def test_no_detection_returns_empty_list(self) -> None:
        _FakeYOLO.predictions_per_call = [{"xyxy": [], "conf": [], "cls": []}]
        detector = self.mod.YOLOPersonDetector(self.mod.YOLOPersonDetectorConfig())
        image = np.zeros((80, 120, 3), dtype=np.uint8)
        boxes = detector.detect(image)
        self.assertEqual(boxes, [])

    def test_multi_detection_and_bbox_clipping(self) -> None:
        _FakeYOLO.predictions_per_call = [
            {
                "xyxy": [
                    [-10, -5, 50, 60],   # 左上越界，需裁剪
                    [30, 40, 180, 140],  # 下边越界，需裁剪
                    [20, 20, 20, 30],    # 无效宽度，需过滤
                ],
                "conf": [0.95, 0.88, 0.5],
                "cls": [0, 0, 0],
            }
        ]
        detector = self.mod.YOLOPersonDetector(self.mod.YOLOPersonDetectorConfig(device="cpu"))
        image = np.zeros((100, 200, 3), dtype=np.uint8)
        boxes = detector.detect(image)

        self.assertEqual(len(boxes), 2)
        self.assertEqual((boxes[0].x, boxes[0].y, boxes[0].w, boxes[0].h), (0, 0, 50, 60))
        self.assertEqual((boxes[1].x, boxes[1].y, boxes[1].w, boxes[1].h), (30, 40, 150, 59))
        self.assertEqual(_FakeYOLO.last_predict_kwargs.get("classes"), [0])

    def test_detect_batch_keeps_input_length(self) -> None:
        _FakeYOLO.predictions_per_call = [
            {"xyxy": [[0, 0, 20, 20]], "conf": [0.9], "cls": [0]},
            {"xyxy": [], "conf": [], "cls": []},
        ]
        detector = self.mod.YOLOPersonDetector(self.mod.YOLOPersonDetectorConfig())
        image = np.zeros((64, 64, 3), dtype=np.uint8)
        outputs = detector.detect_batch([image, image])
        self.assertEqual(len(outputs), 2)
        self.assertEqual(len(outputs[0]), 1)
        self.assertEqual(len(outputs[1]), 0)


if __name__ == "__main__":
    unittest.main()

