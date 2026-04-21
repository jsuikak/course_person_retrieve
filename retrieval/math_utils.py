"""检索数学工具函数。

作用：
- 封装向量归一化、余弦分数计算、Top-K 选择等基础数值逻辑。

典型用法：
```python
scores = cosine_scores(query_feat, gallery_feats)
idxs = topk_indices(scores, k=5)
```
"""

from __future__ import annotations

from typing import Iterable

import numpy as np


def l2_normalize(vec: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """对单个向量做 L2 归一化。"""
    norm = float(np.linalg.norm(vec))
    if norm < eps:
        return vec.astype(np.float32, copy=False)
    return (vec / norm).astype(np.float32, copy=False)


def l2_normalize_rows(mat: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """对矩阵逐行做 L2 归一化。"""
    if mat.size == 0:
        return mat.astype(np.float32, copy=False)
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms = np.maximum(norms, eps)
    return (mat / norms).astype(np.float32, copy=False)


def cosine_scores(query_feat: np.ndarray, gallery_feats: np.ndarray) -> np.ndarray:
    """计算查询向量与图库矩阵的余弦分数。"""
    if gallery_feats.size == 0:
        return np.empty((0,), dtype=np.float32)
    return gallery_feats @ query_feat.astype(np.float32, copy=False)


def topk_indices(scores: np.ndarray, k: int) -> np.ndarray:
    """返回分数最高的 Top-K 索引（按分数降序）。"""
    if scores.size == 0:
        return np.empty((0,), dtype=np.int64)
    k = max(1, min(k, scores.shape[0]))
    part = np.argpartition(-scores, k - 1)[:k]
    order = np.argsort(-scores[part])
    return part[order]


def mean_topk(values: Iterable[float], k: int = 3) -> float:
    """计算序列中最高 K 个值的均值。"""
    arr = np.array(list(values), dtype=np.float32)
    if arr.size == 0:
        return 0.0
    k = max(1, min(k, arr.size))
    idx = np.argpartition(-arr, k - 1)[:k]
    return float(np.mean(arr[idx]))
