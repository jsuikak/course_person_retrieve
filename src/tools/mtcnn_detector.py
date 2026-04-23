"""MTCNN 人脸检测接口（内存输入版）。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

import cv2
import numpy as np

from models import MTCNNFaceDetector


@dataclass(slots=True)
class FaceBox:
    """检测框数据。"""

    x: int
    y: int
    w: int
    h: int


@dataclass(slots=True)
class MTCNNDetectorConfig:
    """MTCNN 检测器配置。"""

    min_face_size: float = 20.0
    thresholds: tuple[float, float, float] = (0.6, 0.7, 0.8)
    nms_thresholds: tuple[float, float, float] = (0.7, 0.7, 0.7)


class MTCNNDetector:
    """最简 MTCNN 检测接口。

    输入：未裁剪的内存 BGR 图像（np.ndarray）。
    输出：检测框列表（List[FaceBox]）。
    """

    def __init__(self, config: MTCNNDetectorConfig | None = None):
        self.config = config or MTCNNDetectorConfig()
        self.detector = MTCNNFaceDetector(
            min_face_size=self.config.min_face_size,
            thresholds=self.config.thresholds,
            nms_thresholds=self.config.nms_thresholds,
        )

    def detect(self, image_bgr: np.ndarray) -> List[FaceBox]:
        """单张图检测，返回检测框列表。"""
        if image_bgr is None or image_bgr.size == 0:
            return []

        raw_boxes = self.detector.detect(image_bgr)
        boxes: List[FaceBox] = []
        for box in raw_boxes:
            if hasattr(box, "x") and hasattr(box, "y") and hasattr(box, "w") and hasattr(box, "h"):
                boxes.append(FaceBox(x=int(box.x), y=int(box.y), w=int(box.w), h=int(box.h)))
            else:
                x, y, w, h = box
                boxes.append(FaceBox(x=int(x), y=int(y), w=int(w), h=int(h)))
        return boxes

    def detect_batch(self, images_bgr: Sequence[np.ndarray]) -> List[List[FaceBox]]:
        """批量检测，返回与输入等长的检测框列表。"""
        return [self.detect(image_bgr) for image_bgr in images_bgr]

    def annotate_image(self, image_path: str, output_dir: str = "outputs/mtcnn_annotated") -> str:
        """读取图片并输出包含人脸框的标记图片。

        输出文件名规则：`<输入文件名>__mtcnn<后缀>`，确保与输入文件名关联。
        """
        image = cv2.imread(image_path)
        if image is None or image.size == 0:
            raise FileNotFoundError(f"Failed to read image: {image_path}")

        boxes = self.detect(image)
        annotated = image.copy()
        for box in boxes:
            x1, y1 = int(box.x), int(box.y)
            x2, y2 = int(box.x + box.w), int(box.y + box.h)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)

        in_path = Path(image_path)
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{in_path.stem}__mtcnn{in_path.suffix}"
        ok = cv2.imwrite(str(out_path), annotated)
        if not ok:
            raise RuntimeError(f"Failed to write annotated image: {out_path}")
        return str(out_path)
