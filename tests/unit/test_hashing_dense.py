from atlassearch.dense.hashing import HashingDenseRetriever


def test_hashing_dense_build_load_search(tmp_path):
    docs = [
        (1, "machine learning retrieval search"),
        (2, "database postgres transaction index"),
        (3, "roman urdu search spelling variation"),
    ]
    index = tmp_path / "hash.npy"
    ids = tmp_path / "ids.json"
    r = HashingDenseRetriever(dim=256)
    r.build(docs, str(index), str(ids))

    loaded = HashingDenseRetriever()
    loaded.load(str(index), str(ids))
    result = loaded.search("machine learning search", 2)
    assert result
    assert result[0][0] == 1
    assert len(result) == 2
