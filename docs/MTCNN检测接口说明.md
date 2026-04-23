# MTCNN 检测接口说明

## 1. 目标

提供一个最简人脸检测接口：

1. 输入：未裁剪的内存图像（`np.ndarray`，BGR）。
2. 处理：调用 `models` 目录中的 MTCNN 模型做检测。
3. 输出：检测框数据列表。

---

## 2. 代码位置

- `src/tools/mtcnn_detector.py`

---

## 3. 核心类型

### 3.1 检测框

`FaceBox`

1. `x: int`
2. `y: int`
3. `w: int`
4. `h: int`

### 3.2 配置

`MTCNNDetectorConfig`

1. `min_face_size: float = 20.0`
2. `thresholds: tuple[float, float, float] = (0.6, 0.7, 0.8)`
3. `nms_thresholds: tuple[float, float, float] = (0.7, 0.7, 0.7)`

---

## 4. 接口定义

### 4.1 单张检测

```python
detect(image_bgr: np.ndarray) -> List[FaceBox]
```

说明：

1. 输入必须是内存 BGR 图像。
2. 空图或无脸时返回空列表 `[]`。

### 4.2 批量检测

```python
detect_batch(images_bgr: Sequence[np.ndarray]) -> List[List[FaceBox]]
```

说明：

1. 返回与输入等长。
2. 每个元素是对应图像的人脸框列表。

---

## 5. 输入输出汇总表

| 接口 | 输入 | 输出 | 说明 |
|---|---|---|---|
| `detect(image_bgr)` | 单张 `np.ndarray(BGR)` | `List[FaceBox]` | 未检测到人脸时返回 `[]` |
| `detect_batch(images_bgr)` | `Sequence[np.ndarray(BGR)]` | `List[List[FaceBox]]` | 与输入等长，逐图返回检测框 |

---

## 6. 最小示例

```python
import cv2
from src.tools.mtcnn_detector import MTCNNDetector

detector = MTCNNDetector()
image = cv2.imread("data/example.jpg")
boxes = detector.detect(image)
print(boxes)
```
