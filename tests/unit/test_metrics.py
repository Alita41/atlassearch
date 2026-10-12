from eval.metrics import precision_at_k, recall_at_k, mrr_at_k, ndcg_at_k


def test_metrics():
    ranked = [3, 1, 2]
    rel = {1, 3}
    assert precision_at_k(ranked, rel, 2) == 1.0
    assert recall_at_k(ranked, rel, 2) == 1.0
    assert mrr_at_k(ranked, rel, 3) == 1.0
    assert ndcg_at_k(ranked, {3: 2, 1: 1}, 2) > 0.99
