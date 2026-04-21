"""检索评估与消融实验模块。

作用：
- 计算图像/视频检索指标（Recall@1/5、mAP）。
- 按固定配置运行四组消融实验并导出结果表。

典型用法：
```python
rows = run_ablation_experiments(...)
save_experiment_table(rows, "ablation_results.csv")
```
"""

from __future__ import annotations

import csv
import os
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np

from .feature_extractor import ArcFaceFeatureExtractor, ExtractorConfig
from .image_index import ImageIndex, build_image_index
from .metrics import evaluate_queries
from .types import ImageRecord, VideoRecord
from .video_index import VideoIndex, build_video_index


@dataclass(slots=True)
class EvalSummary:
    """一次评估任务的指标汇总。"""

    recall_at_1: float
    recall_at_5: float
    mAP: float
    avg_query_time_ms: float
    valid_queries: int
    skipped_queries: int
    skipped_no_gt: int


def _identity_match(row: Dict[str, object], identity: Optional[str]) -> bool:
    """判断检索结果与查询身份是否一致。"""
    if identity is None:
        return False
    return str(row.get("identity", "")) == str(identity)


def evaluate_image_retrieval(
    index: ImageIndex,
    extractor: ArcFaceFeatureExtractor,
    queries: Sequence[ImageRecord],
    ks: Sequence[int] = (1, 5),
) -> EvalSummary:
    """评估图像检索。

    主要流程：逐查询提特征 -> 全排序 -> 生成 relevance 序列 -> 汇总指标。
    """
    all_rel: List[List[bool]] = []
    query_times: List[float] = []
    skipped_queries = 0
    skipped_no_gt = 0

    for q in queries:
        if not q.identity:
            skipped_no_gt += 1
            continue

        t0 = time.time()
        feat = extractor.extract(q.image_path, bbox=q.bbox)
        query_times.append(time.time() - t0)

        if feat is None:
            skipped_queries += 1
            continue

        ranked = index.rank_all_by_feature(feat)
        relevances = []
        for row in ranked:
            # Exclude exact same image for a fair retrieval score.
            if os.path.abspath(str(row.get("image_path", ""))) == os.path.abspath(q.image_path):
                continue
            relevances.append(_identity_match(row, q.identity))

        if not any(relevances):
            skipped_no_gt += 1
            continue

        all_rel.append(relevances)

    metrics = evaluate_queries(all_rel, ks=ks)
    avg_ms = float(np.mean(query_times) * 1000.0) if query_times else 0.0
    return EvalSummary(
        recall_at_1=float(metrics.get("Recall@1", 0.0)),
        recall_at_5=float(metrics.get("Recall@5", 0.0)),
        mAP=float(metrics.get("mAP", 0.0)),
        avg_query_time_ms=avg_ms,
        valid_queries=len(all_rel),
        skipped_queries=skipped_queries,
        skipped_no_gt=skipped_no_gt,
    )


def evaluate_video_retrieval(
    index: VideoIndex,
    extractor: ArcFaceFeatureExtractor,
    queries: Sequence[ImageRecord],
    ks: Sequence[int] = (1, 5),
) -> EvalSummary:
    """评估视频检索（与图像评估流程一致，目标对象为视频排序结果）。"""
    all_rel: List[List[bool]] = []
    query_times: List[float] = []
    skipped_queries = 0
    skipped_no_gt = 0

    for q in queries:
        if not q.identity:
            skipped_no_gt += 1
            continue

        t0 = time.time()
        feat = extractor.extract(q.image_path, bbox=q.bbox)
        query_times.append(time.time() - t0)

        if feat is None:
            skipped_queries += 1
            continue

        ranked = index.rank_all_by_feature(feat)
        relevances = [_identity_match(row, q.identity) for row in ranked]

        if not any(relevances):
            skipped_no_gt += 1
            continue

        all_rel.append(relevances)

    metrics = evaluate_queries(all_rel, ks=ks)
    avg_ms = float(np.mean(query_times) * 1000.0) if query_times else 0.0
    return EvalSummary(
        recall_at_1=float(metrics.get("Recall@1", 0.0)),
        recall_at_5=float(metrics.get("Recall@5", 0.0)),
        mAP=float(metrics.get("mAP", 0.0)),
        avg_query_time_ms=avg_ms,
        valid_queries=len(all_rel),
        skipped_queries=skipped_queries,
        skipped_no_gt=skipped_no_gt,
    )


def run_ablation_experiments(
    *,
    weight_path: str,
    device: str,
    gallery_records: Sequence[ImageRecord],
    query_records: Sequence[ImageRecord],
    output_dir: str,
    detect_face: bool = False,
    blur_threshold: float = 100.0,
    min_face_size: int = 32,
    video_records: Optional[Sequence[VideoRecord]] = None,
    sample_fps: float = 1.0,
    iou_threshold: float = 0.3,
) -> List[Dict[str, object]]:
    """运行固定消融实验：
    Baseline -> +QualityFilter -> +FlipTest -> +VideoTrackAggregation。
    """
    os.makedirs(output_dir, exist_ok=True)
    rows: List[Dict[str, object]] = []

    image_experiments = [
        {
            "name": "Baseline(image)",
            "flip_test": False,
            "blur_threshold": 0.0,
            "min_face_size": 0,
        },
        {
            "name": "+QualityFilter(image)",
            "flip_test": False,
            "blur_threshold": blur_threshold,
            "min_face_size": min_face_size,
        },
        {
            "name": "+FlipTest(image)",
            "flip_test": True,
            "blur_threshold": blur_threshold,
            "min_face_size": min_face_size,
        },
    ]

    for i, exp in enumerate(image_experiments, start=1):
        index_dir = os.path.join(output_dir, f"exp{i}_image_index")
        cfg = ExtractorConfig(
            weight_path=weight_path,
            device=device,
            flip_test=bool(exp["flip_test"]),
            detect_face=detect_face,
            blur_threshold=float(exp["blur_threshold"]),
            min_face_size=int(exp["min_face_size"]),
        )
        extractor = ArcFaceFeatureExtractor(cfg)
        build_stats = build_image_index(gallery_records, extractor, index_dir=index_dir)
        index = ImageIndex(index_dir)
        summary = evaluate_image_retrieval(index=index, extractor=extractor, queries=query_records)

        rows.append(
            {
                "experiment": exp["name"],
                "index_items": build_stats.indexed_records,
                "Recall@1": summary.recall_at_1,
                "Recall@5": summary.recall_at_5,
                "mAP": summary.mAP,
                "avg_query_time_ms": summary.avg_query_time_ms,
                "valid_queries": summary.valid_queries,
                "skipped_queries": summary.skipped_queries,
                "skipped_no_gt": summary.skipped_no_gt,
            }
        )

    if video_records is not None:
        index_dir = os.path.join(output_dir, "exp4_video_index")
        cfg = ExtractorConfig(
            weight_path=weight_path,
            device=device,
            flip_test=True,
            detect_face=detect_face,
            blur_threshold=blur_threshold,
            min_face_size=min_face_size,
        )
        extractor = ArcFaceFeatureExtractor(cfg)
        build_stats = build_video_index(
            records=video_records,
            extractor=extractor,
            index_dir=index_dir,
            sample_fps=sample_fps,
            iou_threshold=iou_threshold,
        )
        index = VideoIndex(index_dir)
        summary = evaluate_video_retrieval(index=index, extractor=extractor, queries=query_records)
        rows.append(
            {
                "experiment": "+VideoTrackAggregation(video)",
                "index_items": build_stats.indexed_frame_features,
                "Recall@1": summary.recall_at_1,
                "Recall@5": summary.recall_at_5,
                "mAP": summary.mAP,
                "avg_query_time_ms": summary.avg_query_time_ms,
                "valid_queries": summary.valid_queries,
                "skipped_queries": summary.skipped_queries,
                "skipped_no_gt": summary.skipped_no_gt,
            }
        )

    return rows


def save_experiment_table(rows: Sequence[Dict[str, object]], output_csv: str) -> None:
    """将实验结果写入 CSV。"""
    if not rows:
        return
    out_dir = os.path.dirname(output_csv)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with open(output_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
