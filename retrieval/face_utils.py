"""人脸框几何工具。

作用：
- 提供检索流程中通用的人脸框运算（选最大框、按框裁剪、IoU 计算）。

典型用法：
```python
from retrieval.face_utils import largest_bbox, crop_with_bbox, iou
```
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from .types import BBox


def largest_bbox(bboxes: List[BBox]) -> Optional[BBox]:
    """从候选框中选择面积最大的框。"""
    if not bboxes:
        return None
    return max(bboxes, key=lambda b: b.w * b.h)


def crop_with_bbox(image_bgr: np.ndarray, bbox: Optional[BBox]) -> np.ndarray:
    """按人脸框裁剪图像，若框为空或非法则回退原图。"""
    if bbox is None:
        return image_bgr
    h, w = image_bgr.shape[:2]
    x1 = max(0, bbox.x)
    y1 = max(0, bbox.y)
    x2 = min(w, bbox.x + bbox.w)
    y2 = min(h, bbox.y + bbox.h)
    if x2 <= x1 or y2 <= y1:
        return image_bgr
    return image_bgr[y1:y2, x1:x2]


def iou(a: BBox, b: BBox) -> float:
    """计算两个人脸框的 IoU，用于视频轨迹关联。"""
    ax1, ay1 = a.x, a.y
    ax2, ay2 = a.x + a.w, a.y + a.h
    bx1, by1 = b.x, b.y
    bx2, by2 = b.x + b.w, b.y + b.h

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0, a.w) * max(0, a.h)
    area_b = max(0, b.w) * max(0, b.h)
    denom = area_a + area_b - inter_area
    if denom <= 0:
        return 0.0
    return inter_area / denom
