"""视频检索可视化模块。

作用：
- 对视频逐帧做人脸检测与检索，输出带框、姓名、置信度的标注视频。

典型用法：
```python
stats = annotate_video_with_retrieval(
    video_path="short.mp4",
    output_video_path="short_annotated.mp4",
    image_index=index,
    extractor=extractor,
)
```
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Optional

import cv2
import numpy as np

from .feature_extractor import ArcFaceFeatureExtractor
from .image_index import ImageIndex
from .types import BBox


@dataclass(slots=True)
class VideoAnnotateStats:
    """视频标注统计结果。"""

    total_frames: int
    faces_detected: int
    matched_faces: int
    output_video: str


def _draw_label(frame: np.ndarray, text: str, x: int, y: int, color: tuple[int, int, int]) -> None:
    """绘制文本背景和标签，保证视频上可读性。"""
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.5
    thickness = 1
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    y1 = max(0, y - th - baseline - 6)
    y2 = max(0, y)
    x2 = min(frame.shape[1] - 1, x + tw + 6)
    cv2.rectangle(frame, (x, y1), (x2, y2), color, -1)
    cv2.putText(frame, text, (x + 3, y2 - 4), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)


def _best_label_from_result(result: Optional[Dict[str, object]], threshold: float) -> tuple[str, float, bool]:
    """从检索结果中提取最终显示的人名和置信度。"""
    if not result:
        return "unknown", 0.0, False
    score = float(result.get("score", 0.0))
    identity = str(result.get("identity") or "")
    image_id = str(result.get("image_id") or "")
    matched = bool(result.get("matched", score >= threshold))
    name = identity if identity else image_id
    if not matched or not name:
        return "unknown", score, False
    return name, score, True


def annotate_video_with_retrieval(
    *,
    video_path: str,
    output_video_path: str,
    image_index: ImageIndex,
    extractor: ArcFaceFeatureExtractor,
    threshold: float = 0.3,
    topk: int = 1,
) -> VideoAnnotateStats:
    """执行视频标注主流程。

    主要流程：
    1) 逐帧检测人脸
    2) 每个人脸提特征并检索图库
    3) 将预测标签与分数绘制到帧上
    4) 写出标注视频并返回统计信息
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if fps <= 0:
        fps = 25.0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    os.makedirs(os.path.dirname(output_video_path) or ".", exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    total_frames = 0
    faces_detected = 0
    matched_faces = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        total_frames += 1

        if extractor.detector is None:
            detections = []
        else:
            raw_boxes = extractor.detector.detect(frame)
            detections = [BBox(x=int(b.x), y=int(b.y), w=int(b.w), h=int(b.h)) for b in raw_boxes]

        for bbox in detections:
            feat = extractor.extract_from_bgr(frame, bbox=bbox)
            if feat is None:
                continue

            faces_detected += 1
            results = image_index.search_by_feature(feat, topk=topk, threshold=threshold)
            best = results[0] if results else None
            name, score, matched = _best_label_from_result(best, threshold=threshold)
            if matched:
                matched_faces += 1

            x1, y1 = bbox.x, bbox.y
            x2, y2 = bbox.x + bbox.w, bbox.y + bbox.h
            color = (0, 180, 0) if matched else (0, 0, 220)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            _draw_label(frame, f"{name} {score:.3f}", x1, y1, color)

        writer.write(frame)

    writer.release()
    cap.release()

    return VideoAnnotateStats(
        total_frames=total_frames,
        faces_detected=faces_detected,
        matched_faces=matched_faces,
        output_video=output_video_path,
    )
