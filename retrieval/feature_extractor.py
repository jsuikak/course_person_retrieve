"""ArcFace 特征提取层。

作用：
- 统一管理图像预处理、MTCNN 检测、质量过滤与特征提取。
- 对外提供 `extract` / `extract_batch` 接口，供索引与检索模块复用。

典型用法：
```python
from retrieval.feature_extractor import ArcFaceFeatureExtractor, ExtractorConfig

extractor = ArcFaceFeatureExtractor(ExtractorConfig(weight_path="./models/weights/arcface.pt"))
feat = extractor.extract("query.jpg")
```
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch

from models import ArcFace, MTCNNFaceDetector

from .face_utils import crop_with_bbox, largest_bbox
from .math_utils import l2_normalize
from .quality import passes_bbox_size, passes_blur_quality
from .types import BBox


@dataclass(slots=True)
class ExtractorConfig:
    """特征提取配置。"""

    weight_path: str
    device: str = "cpu"
    flip_test: bool = True
    detect_face: bool = False
    blur_threshold: float = 0.0
    min_face_size: int = 0


class ArcFaceFeatureExtractor:
    """ArcFace 特征提取器。

    主要流程：
    1) 可选 MTCNN 检测人脸并选最大框
    2) 按阈值做尺寸与清晰度过滤
    3) ArcFace 前向 + 可选 flip test
    4) L2 归一化后输出 512 维向量
    """

    def __init__(self, config: ExtractorConfig):
        self.config = config
        self.arcface = ArcFace(weight_path=config.weight_path, device=config.device)
        if not config.detect_face:
            self.detector = None
        else:
            self.detector = MTCNNFaceDetector(min_face_size=max(float(config.min_face_size), 20.0))

    def _preprocess_rgb(self, image_rgb: np.ndarray) -> torch.Tensor:
        """ArcFace 输入预处理：resize + 归一化 + CHW + batch。"""
        img = cv2.resize(image_rgb, (112, 112)).astype(np.float32)
        img = (img - 127.5) / 128.0
        img = np.transpose(img, (2, 0, 1))
        return torch.from_numpy(img).unsqueeze(0)

    @torch.no_grad()
    def _forward(self, image_bgr: np.ndarray) -> np.ndarray:
        """对单张 BGR 图做 ArcFace 前向，返回未融合的单路特征。"""
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        x = self._preprocess_rgb(image_rgb).to(self.config.device)
        feat = self.arcface.model(x)
        feat = torch.nn.functional.normalize(feat, dim=1)
        return feat[0].detach().cpu().numpy().astype(np.float32)

    def _resolve_bbox(self, image_bgr: np.ndarray, bbox: Optional[BBox]) -> Optional[BBox]:
        """优先使用外部 bbox；否则调用 MTCNN 并取最大人脸框。"""
        if bbox is not None:
            return bbox
        if self.detector is None:
            return None
        detected_raw = self.detector.detect(image_bgr)
        if not detected_raw:
            return None
        first = detected_raw[0]
        if hasattr(first, "x") and hasattr(first, "y") and hasattr(first, "w") and hasattr(first, "h"):
            if isinstance(first, BBox):
                detected: List[BBox] = detected_raw
            else:
                detected = [BBox(x=int(b.x), y=int(b.y), w=int(b.w), h=int(b.h)) for b in detected_raw]
        else:
            detected = [BBox(x=x, y=y, w=w, h=h) for x, y, w, h in detected_raw]
        return largest_bbox(detected)

    def extract_from_bgr(self, image_bgr: np.ndarray, bbox: Optional[BBox] = None) -> Optional[np.ndarray]:
        """从内存中的 BGR 图提取特征（检索核心入口）。"""
        if image_bgr is None or image_bgr.size == 0:
            return None

        bbox = self._resolve_bbox(image_bgr, bbox)
        if not passes_bbox_size(bbox, self.config.min_face_size):
            return None

        crop = crop_with_bbox(image_bgr, bbox)
        if crop.size == 0:
            return None

        if not passes_blur_quality(crop, self.config.blur_threshold):
            return None

        feat = self._forward(crop)
        if self.config.flip_test:
            flipped = cv2.flip(crop, 1)
            feat_flip = self._forward(flipped)
            feat = l2_normalize((feat + feat_flip) / 2.0)
        else:
            feat = l2_normalize(feat)

        return feat

    def extract(self, image_path: str, bbox: Optional[BBox] = None) -> Optional[np.ndarray]:
        """从磁盘图片路径提取特征。"""
        image = cv2.imread(image_path)
        return self.extract_from_bgr(image, bbox=bbox)

    def extract_batch(
        self,
        image_paths: Sequence[str],
        bboxes: Optional[Sequence[Optional[BBox]]] = None,
    ) -> Tuple[np.ndarray, List[int]]:
        """批量提特征并返回有效样本下标（用于建库时过滤失败样本）。"""
        feats: List[np.ndarray] = []
        valid_indices: List[int] = []
        for idx, image_path in enumerate(image_paths):
            bbox = bboxes[idx] if bboxes is not None else None
            feat = self.extract(image_path, bbox=bbox)
            if feat is None:
                continue
            feats.append(feat)
            valid_indices.append(idx)

        if not feats:
            return np.empty((0, 512), dtype=np.float32), valid_indices
        return np.vstack(feats).astype(np.float32), valid_indices
