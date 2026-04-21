"""视频库索引与检索。

作用：
- 从视频采样帧提取人脸特征并构建视频索引。
- 基于轨迹聚合分数实现 image->video 检索。

典型用法：
```python
stats = build_video_index(video_records, extractor, "outputs/video_index")
index = VideoIndex("outputs/video_index")
results = index.search_video("query.jpg", extractor)
```
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .face_utils import iou
from .feature_extractor import ArcFaceFeatureExtractor
from .io_utils import read_csv_rows, write_csv
from .math_utils import cosine_scores, l2_normalize_rows, mean_topk
from .types import BBox, VideoRecord


@dataclass(slots=True)
class VideoIndexBuildStats:
    """视频索引构建统计。"""

    total_videos: int
    indexed_videos: int
    indexed_frame_features: int
    skipped_videos: int
    elapsed_sec: float


class VideoIndex:
    """视频索引读取与检索接口。"""

    def __init__(self, index_dir: str):
        self.index_dir = index_dir
        self.features_path = os.path.join(index_dir, "features.npy")
        self.meta_path = os.path.join(index_dir, "meta.csv")

        if not os.path.exists(self.features_path):
            raise FileNotFoundError(f"Index not found: {self.features_path}")
        if not os.path.exists(self.meta_path):
            raise FileNotFoundError(f"Index not found: {self.meta_path}")

        self.features = np.load(self.features_path).astype(np.float32)
        self.features = l2_normalize_rows(self.features)
        self.meta = read_csv_rows(self.meta_path)

        if self.features.shape[0] != len(self.meta):
            raise ValueError(
                f"Index corrupted: features ({self.features.shape[0]}) != meta ({len(self.meta)})"
            )

    def __len__(self) -> int:
        return self.features.shape[0]

    def _rank_all_by_feature(self, query_feat: np.ndarray, threshold: float) -> List[Dict[str, object]]:
        """按视频维度输出全排序结果。

        主要流程：
        1) 计算查询与每个帧特征的分数
        2) 轨迹内取 top3 均值
        3) 视频层取最佳轨迹分数
        """
        if self.features.size == 0:
            return []

        scores = cosine_scores(query_feat, self.features)
        track_scores: Dict[Tuple[str, str], List[float]] = {}
        video_info: Dict[str, Dict[str, str]] = {}
        video_frame_counts: Dict[str, int] = {}

        for row, score in zip(self.meta, scores):
            video_id = row["video_id"]
            track_id = row["track_id"]
            key = (video_id, track_id)
            track_scores.setdefault(key, []).append(float(score))
            video_frame_counts[video_id] = video_frame_counts.get(video_id, 0) + 1
            if video_id not in video_info:
                video_info[video_id] = {
                    "video_id": video_id,
                    "video_path": row.get("video_path", ""),
                    "identity": row.get("identity", ""),
                }

        video_to_tracks: Dict[str, List[Tuple[str, float]]] = {}
        for (video_id, track_id), score_list in track_scores.items():
            ts = mean_topk(score_list, k=3)
            video_to_tracks.setdefault(video_id, []).append((track_id, ts))

        ranked: List[Dict[str, object]] = []
        for video_id, tracks in video_to_tracks.items():
            tracks = sorted(tracks, key=lambda x: x[1], reverse=True)
            best_track_id, video_score = tracks[0]
            row: Dict[str, object] = dict(video_info[video_id])
            row["score"] = float(video_score)
            row["best_track_id"] = best_track_id
            row["num_tracks"] = len(tracks)
            row["num_frame_features"] = video_frame_counts.get(video_id, 0)
            row["matched"] = float(video_score) >= threshold
            ranked.append(row)

        ranked.sort(key=lambda x: float(x["score"]), reverse=True)
        return ranked

    def rank_all_by_feature(self, query_feat: np.ndarray, threshold: float = 0.3) -> List[Dict[str, object]]:
        """返回全量视频排序（用于评估）。"""
        return self._rank_all_by_feature(query_feat, threshold=threshold)

    def search_by_feature(self, query_feat: np.ndarray, topk: int = 5, threshold: float = 0.3) -> List[Dict[str, object]]:
        """返回 Top-K 视频检索结果。"""
        ranked = self._rank_all_by_feature(query_feat, threshold=threshold)
        if not ranked:
            return []
        return ranked[: max(1, min(topk, len(ranked)))]

    def search_video(
        self,
        query_image_path: str,
        extractor: ArcFaceFeatureExtractor,
        topk: int = 5,
        threshold: float = 0.3,
        bbox: Optional[BBox] = None,
    ) -> List[Dict[str, object]]:
        """从查询图片直接检索视频库。"""
        feat = extractor.extract(query_image_path, bbox=bbox)
        if feat is None:
            return []
        return self.search_by_feature(feat, topk=topk, threshold=threshold)


def _match_track(
    bbox: BBox,
    frame_idx: int,
    active_tracks: Dict[str, Dict[str, object]],
    used_track_ids: set[str],
    iou_threshold: float,
    max_gap_frames: int,
) -> Optional[str]:
    """将当前框匹配到已存在轨迹（基于 IoU + 帧间间隔约束）。"""
    best_track: Optional[str] = None
    best_iou = -1.0
    for track_id, info in active_tracks.items():
        if track_id in used_track_ids:
            continue
        last_frame = int(info["last_frame"])
        if frame_idx - last_frame > max_gap_frames:
            continue
        last_bbox = info["last_bbox"]
        cur_iou = iou(bbox, last_bbox)
        if cur_iou >= iou_threshold and cur_iou > best_iou:
            best_iou = cur_iou
            best_track = track_id
    return best_track


def _sample_step(native_fps: float, sample_fps: float) -> int:
    """根据原始 FPS 与目标采样率计算步长。"""
    if native_fps <= 0:
        native_fps = 25.0
    if sample_fps <= 0:
        sample_fps = 1.0
    return max(1, int(round(native_fps / sample_fps)))


def build_video_index(
    records: Sequence[VideoRecord],
    extractor: ArcFaceFeatureExtractor,
    index_dir: str,
    sample_fps: float = 1.0,
    iou_threshold: float = 0.3,
    max_track_gap: int = 1,
) -> VideoIndexBuildStats:
    """构建视频索引。

    主要流程：
    1) 读取视频并按固定步长采样
    2) 对每帧检测人脸并提取特征
    3) 用 IoU 做简单轨迹关联
    4) 保存帧级特征与元信息
    """
    os.makedirs(index_dir, exist_ok=True)

    t0 = time.time()
    features: List[np.ndarray] = []
    meta_rows: List[Dict[str, object]] = []
    indexed_videos = 0

    for rec in records:
        cap = cv2.VideoCapture(rec.video_path)
        if not cap.isOpened():
            continue

        native_fps = float(cap.get(cv2.CAP_PROP_FPS))
        step = _sample_step(native_fps, sample_fps)
        max_gap_frames = step * max(1, max_track_gap)

        frame_idx = -1
        next_track_id = 0
        active_tracks: Dict[str, Dict[str, object]] = {}
        video_has_feature = False

        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_idx += 1

            if frame_idx % step != 0:
                continue

            if extractor.detector is None:
                h, w = frame.shape[:2]
                bboxes = [BBox(0, 0, w, h)]
            else:
                raw_boxes = extractor.detector.detect(frame)
                if not raw_boxes:
                    continue
                bboxes = [BBox(x=int(b.x), y=int(b.y), w=int(b.w), h=int(b.h)) for b in raw_boxes]

            used_track_ids: set[str] = set()
            for bbox in bboxes:
                feat = extractor.extract_from_bgr(frame, bbox=bbox)
                if feat is None:
                    continue

                track_id = _match_track(
                    bbox=bbox,
                    frame_idx=frame_idx,
                    active_tracks=active_tracks,
                    used_track_ids=used_track_ids,
                    iou_threshold=iou_threshold,
                    max_gap_frames=max_gap_frames,
                )
                if track_id is None:
                    track_id = str(next_track_id)
                    next_track_id += 1

                used_track_ids.add(track_id)
                active_tracks[track_id] = {
                    "last_bbox": bbox,
                    "last_frame": frame_idx,
                }

                timestamp = frame_idx / native_fps if native_fps > 0 else 0.0
                features.append(feat)
                meta_rows.append(
                    {
                        "video_id": rec.video_id,
                        "video_path": rec.video_path,
                        "identity": rec.identity or "",
                        "frame_idx": frame_idx,
                        "timestamp": f"{timestamp:.4f}",
                        "track_id": track_id,
                        "x": bbox.x,
                        "y": bbox.y,
                        "w": bbox.w,
                        "h": bbox.h,
                    }
                )
                video_has_feature = True

        cap.release()
        if video_has_feature:
            indexed_videos += 1

    if features:
        feature_mat = np.vstack(features).astype(np.float32)
    else:
        feature_mat = np.empty((0, 512), dtype=np.float32)

    np.save(os.path.join(index_dir, "features.npy"), feature_mat)
    write_csv(
        os.path.join(index_dir, "meta.csv"),
        meta_rows,
        fieldnames=[
            "video_id",
            "video_path",
            "identity",
            "frame_idx",
            "timestamp",
            "track_id",
            "x",
            "y",
            "w",
            "h",
        ],
    )

    info = {
        "type": "video_index",
        "num_frame_features": int(feature_mat.shape[0]),
        "feature_dim": int(feature_mat.shape[1]) if feature_mat.size else 512,
        "sample_fps": float(sample_fps),
        "iou_threshold": float(iou_threshold),
        "max_track_gap": int(max_track_gap),
        "flip_test": bool(extractor.config.flip_test),
        "detect_face": bool(extractor.config.detect_face),
        "blur_threshold": float(extractor.config.blur_threshold),
        "min_face_size": int(extractor.config.min_face_size),
    }
    with open(os.path.join(index_dir, "index_info.json"), "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)

    elapsed = time.time() - t0
    return VideoIndexBuildStats(
        total_videos=len(records),
        indexed_videos=indexed_videos,
        indexed_frame_features=feature_mat.shape[0],
        skipped_videos=len(records) - indexed_videos,
        elapsed_sec=elapsed,
    )
