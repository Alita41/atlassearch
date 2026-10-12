from __future__ import annotations
import math


def precision_at_k(ranked_ids, relevant_ids, k):
    if k <= 0:
        return 0.0
    rel = set(relevant_ids)
    return sum(doc_id in rel for doc_id in ranked_ids[:k]) / k


def recall_at_k(ranked_ids, relevant_ids, k):
    rel = set(relevant_ids)
    if not rel:
        return 0.0
    return sum(doc_id in rel for doc_id in ranked_ids[:k]) / len(rel)


def mrr_at_k(ranked_ids, relevant_ids, k):
    rel = set(relevant_ids)
    for i, doc_id in enumerate(ranked_ids[:k], start=1):
        if doc_id in rel:
            return 1.0 / i
    return 0.0


def ndcg_at_k(ranked_ids, graded_relevance, k):
    def dcg(gains):
        return sum((2 ** gain - 1) / math.log2(i + 2) for i, gain in enumerate(gains))
    gains = [graded_relevance.get(doc_id, 0) for doc_id in ranked_ids[:k]]
    ideal = sorted(graded_relevance.values(), reverse=True)[:k]
    denom = dcg(ideal)
    return dcg(gains) / denom if denom else 0.0
