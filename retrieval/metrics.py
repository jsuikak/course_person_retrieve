"""检索评估指标实现。

作用：
- 提供 AP、Recall@K、mAP 的纯 Python/NumPy 计算逻辑。

典型用法：
```python
metrics = evaluate_queries(all_relevances, ks=(1, 5))
```
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np


def average_precision(relevances: Sequence[bool]) -> float:
    """计算单个查询的 AP。"""
    num_rel = int(np.sum(relevances))
    if num_rel == 0:
        return 0.0

    precisions: List[float] = []
    hit = 0
    for i, is_rel in enumerate(relevances, start=1):
        if is_rel:
            hit += 1
            precisions.append(hit / i)
    return float(np.mean(precisions)) if precisions else 0.0


def recall_at_k(relevances: Sequence[bool], k: int) -> float:
    """计算单个查询的 Recall@K（命中即 1，否则 0）。"""
    if k <= 0:
        return 0.0
    top = list(relevances[:k])
    return 1.0 if any(top) else 0.0


def evaluate_queries(
    all_relevances: Sequence[Sequence[bool]],
    ks: Sequence[int] = (1, 5),
) -> Dict[str, float]:
    """汇总多查询指标，输出 Recall@K 与 mAP。"""
    if not all_relevances:
        metrics = {f"Recall@{k}": 0.0 for k in ks}
        metrics["mAP"] = 0.0
        return metrics

    out: Dict[str, float] = {}
    for k in ks:
        out[f"Recall@{k}"] = float(np.mean([recall_at_k(rel, k) for rel in all_relevances]))
    out["mAP"] = float(np.mean([average_precision(rel) for rel in all_relevances]))
    return out
