from __future__ import annotations
from collections import defaultdict
from typing import Iterable


def reciprocal_rank_fusion(rankings: Iterable[list[tuple[int, float]]], k: int = 60, top_k: int = 10):
    scores: dict[int, float] = defaultdict(float)
    detail: dict[int, list[int]] = defaultdict(list)
    for ranking in rankings:
        for rank, (doc_id, _score) in enumerate(ranking, start=1):
            scores[doc_id] += 1.0 / (k + rank)
            detail[doc_id].append(rank)
    ordered = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))[:top_k]
    return [(doc_id, score, detail[doc_id]) for doc_id, score in ordered]
