"""Retrieval metrics for image-based person ReID benchmarks."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np


@dataclass(slots=True)
class RetrievalMetricResult:
    """Aggregate CMC and mAP metrics.

    Metric values are percentages in the range [0, 100].
    """

    ranks: tuple[int, ...]
    cmc: dict[int, float]
    mAP: float
    total_queries: int
    valid_queries: int
    invalid_queries: int
    queries_without_positive: int
    per_query_ap: list[float] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        out: dict[str, object] = {
            "metric_unit": "percent",
            "mAP": float(self.mAP),
            "total_queries": int(self.total_queries),
            "valid_queries": int(self.valid_queries),
            "invalid_queries": int(self.invalid_queries),
            "queries_without_positive": int(self.queries_without_positive),
        }
        for rank in self.ranks:
            out[f"rank{rank}"] = float(self.cmc.get(int(rank), 0.0))
        return out


def _as_label_array(labels: Sequence[str] | np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(labels, dtype=str)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a 1-D sequence.")
    return arr


def _average_precision(sorted_matches: np.ndarray, positive_count: int) -> float:
    if positive_count <= 0:
        return 0.0
    hit_positions = np.flatnonzero(sorted_matches)
    if hit_positions.size == 0:
        return 0.0
    precisions = []
    for pos in hit_positions:
        rank = int(pos) + 1
        hits_up_to_rank = int(sorted_matches[:rank].sum())
        precisions.append(hits_up_to_rank / rank)
    return float(np.sum(precisions) / positive_count)


def evaluate_reid(
    scores: np.ndarray,
    query_ids: Sequence[str] | np.ndarray,
    gallery_ids: Sequence[str] | np.ndarray,
    ranks: Iterable[int] = (1, 5, 10),
    valid_query_mask: Sequence[bool] | np.ndarray | None = None,
) -> RetrievalMetricResult:
    """Evaluate ReID retrieval scores with CMC Rank@K and mAP.

    Args:
        scores: Similarity matrix shaped ``(num_query, num_gallery)``. Higher is better.
        query_ids: Identity label for each query row.
        gallery_ids: Identity label for each gallery column.
        ranks: Rank cutoffs used for CMC/Recall@K.
        valid_query_mask: Optional mask. Invalid queries are kept in the denominator
            and contribute zero AP and zero Rank@K hits.
    """

    score_mat = np.asarray(scores, dtype=np.float32)
    if score_mat.ndim != 2:
        raise ValueError("scores must be a 2-D matrix.")

    q_ids = _as_label_array(query_ids, "query_ids")
    g_ids = _as_label_array(gallery_ids, "gallery_ids")
    if score_mat.shape != (q_ids.shape[0], g_ids.shape[0]):
        raise ValueError(
            "scores shape must match query/gallery labels: "
            f"scores={score_mat.shape}, query={q_ids.shape[0]}, gallery={g_ids.shape[0]}"
        )

    rank_tuple = tuple(sorted({int(rank) for rank in ranks if int(rank) > 0}))
    if not rank_tuple:
        raise ValueError("ranks must contain at least one positive integer.")

    if valid_query_mask is None:
        valid_mask = np.ones((q_ids.shape[0],), dtype=bool)
    else:
        valid_mask = np.asarray(valid_query_mask, dtype=bool)
        if valid_mask.shape != (q_ids.shape[0],):
            raise ValueError("valid_query_mask length must equal query_ids length.")

    hits_at_rank = {rank: [] for rank in rank_tuple}
    per_query_ap: list[float] = []
    invalid_queries = 0
    queries_without_positive = 0

    for q_idx, query_id in enumerate(q_ids):
        if not bool(valid_mask[q_idx]):
            invalid_queries += 1
            per_query_ap.append(0.0)
            for rank in rank_tuple:
                hits_at_rank[rank].append(0.0)
            continue

        positive_mask = g_ids == query_id
        positive_count = int(positive_mask.sum())
        if positive_count == 0:
            queries_without_positive += 1
            per_query_ap.append(0.0)
            for rank in rank_tuple:
                hits_at_rank[rank].append(0.0)
            continue

        order = np.argsort(-score_mat[q_idx], kind="mergesort")
        sorted_matches = positive_mask[order]
        per_query_ap.append(_average_precision(sorted_matches=sorted_matches, positive_count=positive_count))
        for rank in rank_tuple:
            cutoff = min(rank, sorted_matches.shape[0])
            hits_at_rank[rank].append(float(bool(sorted_matches[:cutoff].any())))

    total_queries = int(q_ids.shape[0])
    if total_queries == 0:
        cmc = {rank: 0.0 for rank in rank_tuple}
        mean_ap = 0.0
    else:
        cmc = {rank: float(np.mean(hits_at_rank[rank]) * 100.0) for rank in rank_tuple}
        mean_ap = float(np.mean(per_query_ap) * 100.0)

    return RetrievalMetricResult(
        ranks=rank_tuple,
        cmc=cmc,
        mAP=mean_ap,
        total_queries=total_queries,
        valid_queries=int(total_queries - invalid_queries),
        invalid_queries=int(invalid_queries),
        queries_without_positive=int(queries_without_positive),
        per_query_ap=per_query_ap,
    )


__all__ = ["RetrievalMetricResult", "evaluate_reid"]
