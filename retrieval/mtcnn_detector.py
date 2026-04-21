"""MTCNN 检测器兼容导出。

作用：
- 为历史代码保留 `retrieval.mtcnn_detector.MTCNNFaceDetector` 导入路径。
- 实际实现已迁移到 `models.mtcnn`。

典型用法：
```python
from retrieval.mtcnn_detector import MTCNNFaceDetector
```
"""

from models import MTCNNFaceDetector

__all__ = ["MTCNNFaceDetector"]
