"""图像质量过滤工具。

作用：
- 提供模糊度与最小人脸尺寸阈值判断，减少低质量样本对检索的干扰。

典型用法：
```python
ok = passes_blur_quality(face_crop, blur_threshold=100.0)
```
"""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from .types import BBox


def laplacian_variance(image_bgr: np.ndarray) -> float:
    """使用拉普拉斯方差估计图像清晰度。"""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def passes_blur_quality(image_bgr: np.ndarray, blur_threshold: float) -> bool:
    """根据阈值判断是否通过清晰度过滤。"""
    if blur_threshold <= 0:
        return True
    return laplacian_variance(image_bgr) >= blur_threshold


def passes_bbox_size(bbox: Optional[BBox], min_face_size: int) -> bool:
    """根据最小宽高阈值判断人脸框是否有效。"""
    if bbox is None or min_face_size <= 0:
        return True
    return bbox.w >= min_face_size and bbox.h >= min_face_size
