"""YOLO 人体检测接口（Ultralytics，内存输入版）。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

import numpy as np


@dataclass(slots=True)
class PersonBox:
    """人体检测框数据。"""

    x: int
    y: int
    w: int
    h: int
    conf: float
    cls: int


@dataclass(slots=True)
class YOLOPersonDetectorConfig:
    """YOLO 检测器配置。"""

    weights: str = "./models/weights/yolo11n.pt"
    conf: float = 0.25
    iou: float = 0.7
    max_det: int = 100
    device: str = "cpu"
    person_class_id: int = 0


class YOLOPersonDetector:
    """最简 YOLO 人体检测接口。

    输入：未裁剪的内存 BGR 图像（np.ndarray）。
    输出：检测框列表（List[PersonBox]）。
    """

    def __init__(self, config: YOLOPersonDetectorConfig | None = None):
        self.config = config or YOLOPersonDetectorConfig()
        try:
            from ultralytics import YOLO
        except Exception as exc:
            raise ImportError(
                "ultralytics is required for YOLOPersonDetector. Install it first, "
                'or run in face mode. Example: `uv pip install ultralytics`.'
            ) from exc

        weights_path = self._resolve_weights_path(self.config.weights)
        try:
            self.model = YOLO(weights_path)
        except Exception as exc:
            raise RuntimeError(
                f"failed to load YOLO weights: {weights_path}. "
                "If network is unavailable, provide a local weights path via --yolo-weights."
            ) from exc

    @staticmethod
    def _resolve_weights_path(weights: str) -> str:
        """Resolve YOLO weights path with backward-compatible fallbacks."""
        raw = (weights or "").strip()
        if not raw:
            raise ValueError("YOLO weights path is empty.")

        base = Path(raw).expanduser()
        if base.is_absolute():
            return str(base)

        project_root = Path(__file__).resolve().parents[2]
        candidates: list[Path] = [
            (Path.cwd() / base).resolve(),
            (project_root / base).resolve(),
        ]
        if base.name == "yolo11n.pt":
            candidates.extend(
                [
                    (project_root / "models" / "weights" / "yolo11n.pt").resolve(),
                    (project_root / "yolo11n.pt").resolve(),
                ]
            )

        for candidate in candidates:
            if candidate.exists():
                return str(candidate)
        return str(candidates[0])

    @staticmethod
    def _to_numpy(value: object) -> np.ndarray:
        """将 YOLO 输出对象统一转换为 numpy 数组。"""
        if value is None:
            return np.empty((0,), dtype=np.float32)
        if hasattr(value, "cpu") and callable(value.cpu):
            value = value.cpu()
        if hasattr(value, "numpy") and callable(value.numpy):
            return np.asarray(value.numpy())
        return np.asarray(value)

    def detect(self, image_bgr: np.ndarray) -> List[PersonBox]:
        """单张图检测，返回人体检测框列表。"""
        if image_bgr is None or image_bgr.size == 0:
            return []

        outputs = self.model.predict(
            source=image_bgr,
            conf=float(self.config.conf),
            iou=float(self.config.iou),
            max_det=int(self.config.max_det),
            device=self.config.device,
            classes=[int(self.config.person_class_id)],
            verbose=False,
        )
        if not outputs:
            return []

        boxes_data = getattr(outputs[0], "boxes", None)
        if boxes_data is None:
            return []

        xyxy = self._to_numpy(getattr(boxes_data, "xyxy", None))
        confs = self._to_numpy(getattr(boxes_data, "conf", None)).reshape(-1)
        clss = self._to_numpy(getattr(boxes_data, "cls", None)).reshape(-1)
        if xyxy.size == 0 or confs.size == 0 or clss.size == 0:
            return []

        h_img, w_img = image_bgr.shape[:2]
        out: List[PersonBox] = []
        for i in range(min(xyxy.shape[0], confs.shape[0], clss.shape[0])):
            x1, y1, x2, y2 = [int(v) for v in xyxy[i][:4]]
            x1 = max(0, min(w_img - 1, x1))
            y1 = max(0, min(h_img - 1, y1))
            x2 = max(0, min(w_img - 1, x2))
            y2 = max(0, min(h_img - 1, y2))
            w = x2 - x1
            h = y2 - y1
            if w <= 1 or h <= 1:
                continue
            out.append(
                PersonBox(
                    x=x1,
                    y=y1,
                    w=w,
                    h=h,
                    conf=float(confs[i]),
                    cls=int(clss[i]),
                )
            )
        return out

    def detect_batch(self, images_bgr: Sequence[np.ndarray]) -> List[List[PersonBox]]:
        """批量检测，返回与输入等长的检测框列表。"""
        return [self.detect(image_bgr) for image_bgr in images_bgr]
