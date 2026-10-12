from atlassearch.fusion.rrf import reciprocal_rank_fusion


def test_rrf_rewards_overlap():
    a = [(1, 10.0), (2, 9.0)]
    b = [(2, 0.9), (3, 0.8)]
    out = reciprocal_rank_fusion([a, b], top_k=3)
    assert out[0][0] == 2
