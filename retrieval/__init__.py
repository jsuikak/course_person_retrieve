"""检索包统一导出。

作用：
- 暴露项目对外最常用的检索 API，避免上层脚本直接依赖内部细节模块。

典型用法：
```python
from retrieval import ArcFaceFeatureExtractor, build_image_index, ImageIndex
```
"""

from .feature_extractor import ArcFaceFeatureExtractor
from .image_index import ImageIndex, build_image_index
from .video_index import VideoIndex, build_video_index
from .video_annotator import annotate_video_with_retrieval

__all__ = [
    "ArcFaceFeatureExtractor",
    "ImageIndex",
    "VideoIndex",
    "annotate_video_with_retrieval",
    "build_image_index",
    "build_video_index",
]
