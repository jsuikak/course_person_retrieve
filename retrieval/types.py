"""检索系统共享数据结构。

作用：
- 定义跨模块传递的统一类型（人脸框、图像样本、视频样本）。

典型用法：
```python
record = ImageRecord(image_path="a.jpg", image_id="a", identity="p001")
```
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(slots=True)
class BBox:
    """人脸框（左上角 + 宽高）。"""
    x: int
    y: int
    w: int
    h: int


@dataclass(slots=True)
class ImageRecord:
    """图像样本记录。"""
    image_path: str
    image_id: str
    identity: Optional[str] = None
    bbox: Optional[BBox] = None


@dataclass(slots=True)
class VideoRecord:
    """视频样本记录。"""
    video_path: str
    video_id: str
    identity: Optional[str] = None
