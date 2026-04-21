"""图像库索引与检索。

作用：
- 将图像库样本提成特征并持久化为 `features.npy + meta.csv`。
- 提供基于余弦相似度的图像检索（全排序 / Top-K）。

典型用法：
```python
stats = build_image_index(records, extractor, "outputs/image_index")
index = ImageIndex("outputs/image_index")
results = index.search_image("query.jpg", extractor, topk=5)
```
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np

from .feature_extractor import ArcFaceFeatureExtractor
from .io_utils import read_csv_rows, write_csv
from .math_utils import cosine_scores, l2_normalize_rows, topk_indices
from .types import BBox, ImageRecord


@dataclass(slots=True)
class ImageIndexBuildStats:
    """图像索引构建统计。"""

    total_records: int
    indexed_records: int
    skipped_records: int
    elapsed_sec: float


class ImageIndex:
    """图像索引读取与检索接口。"""

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

    def rank_all_by_feature(self, query_feat: np.ndarray) -> List[Dict[str, object]]:
        """返回全量排序结果（用于 mAP 等评估）。"""
        scores = cosine_scores(query_feat, self.features)
        order = np.argsort(-scores)
        results: List[Dict[str, object]] = []
        for idx in order:
            row = dict(self.meta[int(idx)])
            row["score"] = float(scores[int(idx)])
            results.append(row)
        return results

    def search_by_feature(self, query_feat: np.ndarray, topk: int = 5, threshold: float = 0.3) -> List[Dict[str, object]]:
        """返回 Top-K 检索结果，并附带阈值匹配标记。"""
        scores = cosine_scores(query_feat, self.features)
        idxs = topk_indices(scores, topk)
        out: List[Dict[str, object]] = []
        for idx in idxs:
            row = dict(self.meta[int(idx)])
            score = float(scores[int(idx)])
            row["score"] = score
            row["matched"] = score >= threshold
            out.append(row)
        return out

    def search_image(
        self,
        query_image_path: str,
        extractor: ArcFaceFeatureExtractor,
        topk: int = 5,
        threshold: float = 0.3,
        bbox: Optional[BBox] = None,
    ) -> List[Dict[str, object]]:
        """从查询图片直接检索图库。"""
        feat = extractor.extract(query_image_path, bbox=bbox)
        if feat is None:
            return []
        return self.search_by_feature(feat, topk=topk, threshold=threshold)


def build_image_index(
    records: Sequence[ImageRecord],
    extractor: ArcFaceFeatureExtractor,
    index_dir: str,
    batch_size: int = 32,
) -> ImageIndexBuildStats:
    """构建图像索引。

    主要流程：
    1) 分批读取路径并提特征
    2) 过滤提特征失败样本
    3) 保存特征矩阵与元信息文件
    """
    os.makedirs(index_dir, exist_ok=True)

    t0 = time.time()
    feats: List[np.ndarray] = []
    meta_rows: List[Dict[str, object]] = []

    for start in range(0, len(records), batch_size):
        batch = records[start : start + batch_size]
        paths = [r.image_path for r in batch]
        bboxes = [r.bbox for r in batch]
        batch_feats, valid_indices = extractor.extract_batch(paths, bboxes=bboxes)
        if batch_feats.size == 0:
            continue

        feats.append(batch_feats)
        for local_idx in valid_indices:
            rec = batch[local_idx]
            row: Dict[str, object] = {
                "image_id": rec.image_id,
                "image_path": rec.image_path,
                "identity": rec.identity or "",
                "x": rec.bbox.x if rec.bbox else "",
                "y": rec.bbox.y if rec.bbox else "",
                "w": rec.bbox.w if rec.bbox else "",
                "h": rec.bbox.h if rec.bbox else "",
            }
            meta_rows.append(row)

    if feats:
        feature_mat = np.vstack(feats).astype(np.float32)
    else:
        feature_mat = np.empty((0, 512), dtype=np.float32)

    np.save(os.path.join(index_dir, "features.npy"), feature_mat)
    write_csv(
        os.path.join(index_dir, "meta.csv"),
        meta_rows,
        fieldnames=["image_id", "image_path", "identity", "x", "y", "w", "h"],
    )

    info = {
        "type": "image_index",
        "num_items": int(feature_mat.shape[0]),
        "feature_dim": int(feature_mat.shape[1]) if feature_mat.size else 512,
        "flip_test": bool(extractor.config.flip_test),
        "detect_face": bool(extractor.config.detect_face),
        "blur_threshold": float(extractor.config.blur_threshold),
        "min_face_size": int(extractor.config.min_face_size),
    }
    with open(os.path.join(index_dir, "index_info.json"), "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)

    elapsed = time.time() - t0
    return ImageIndexBuildStats(
        total_records=len(records),
        indexed_records=feature_mat.shape[0],
        skipped_records=len(records) - feature_mat.shape[0],
        elapsed_sec=elapsed,
    )
