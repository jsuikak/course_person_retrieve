"""基于已有特征的统一匹配接口（face/person 共用）。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import numpy as np

from .feature_extractor import FeatureMode


@dataclass(slots=True)
class MatchItem:
    """单条匹配结果。"""

    rank: int
    index: int
    score: float
    is_match: bool
    gallery_id: Optional[str] = None


@dataclass(slots=True)
class MatchResult:
    """单条查询特征的匹配结果。"""

    mode: FeatureMode
    topk: int
    total_gallery: int
    items: List[MatchItem] = field(default_factory=list)


def _l2_normalize(vec: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm < eps:
        return vec.astype(np.float32, copy=False)
    return (vec / norm).astype(np.float32, copy=False)


def _l2_normalize_rows(mat: np.ndarray) -> np.ndarray:
    if mat.size == 0:
        return mat.astype(np.float32, copy=False)
    return np.vstack([_l2_normalize(row) for row in mat]).astype(np.float32, copy=False)


def _topk_indices(scores: np.ndarray, topk: int) -> np.ndarray:
    if scores.size == 0:
        return np.empty((0,), dtype=np.int64)
    k = max(1, min(int(topk), scores.shape[0]))
    part = np.argpartition(-scores, k - 1)[:k]
    order = np.argsort(-scores[part])
    return part[order]


def match(
    query_feature: np.ndarray,
    gallery_features: np.ndarray,
    mode: FeatureMode,
    topk: int = 5,
    threshold: Optional[float] = None,
    gallery_ids: Optional[Sequence[str]] = None,
) -> MatchResult:
    """单查询特征匹配。"""
    if query_feature.ndim != 1:
        raise ValueError("query_feature must be 1-D.")
    if gallery_features.ndim != 2:
        raise ValueError("gallery_features must be 2-D.")
    if gallery_features.shape[0] == 0:
        return MatchResult(mode=mode, topk=topk, total_gallery=0, items=[])
    if query_feature.shape[0] != gallery_features.shape[1]:
        raise ValueError(f"Feature dim mismatch: query={query_feature.shape[0]}, gallery={gallery_features.shape[1]}")
    if gallery_ids is not None and len(gallery_ids) != gallery_features.shape[0]:
        raise ValueError("gallery_ids length must equal gallery_features rows.")

    query = _l2_normalize(query_feature.astype(np.float32, copy=False))
    gallery = _l2_normalize_rows(gallery_features.astype(np.float32, copy=False))
    scores = gallery @ query

    top_indices = _topk_indices(scores, topk=topk)
    items: List[MatchItem] = []
    for rank, idx in enumerate(top_indices, start=1):
        score = float(scores[idx])
        is_match = True if threshold is None else score >= float(threshold)
        gallery_id = None if gallery_ids is None else str(gallery_ids[idx])
        items.append(
            MatchItem(
                rank=rank,
                index=int(idx),
                score=score,
                is_match=is_match,
                gallery_id=gallery_id,
            )
        )

    return MatchResult(
        mode=mode,
        topk=max(1, min(int(topk), gallery_features.shape[0])),
        total_gallery=int(gallery_features.shape[0]),
        items=items,
    )


def match_batch(
    query_features: np.ndarray,
    gallery_features: np.ndarray,
    mode: FeatureMode,
    topk: int = 5,
    threshold: Optional[float] = None,
    gallery_ids: Optional[Sequence[str]] = None,
) -> List[MatchResult]:
    """批量查询特征匹配。"""
    if query_features.ndim != 2:
        raise ValueError("query_features must be 2-D.")
    return [
        match(
            query_feature=query_features[i],
            gallery_features=gallery_features,
            mode=mode,
            topk=topk,
            threshold=threshold,
            gallery_ids=gallery_ids,
        )
        for i in range(query_features.shape[0])
    ]
