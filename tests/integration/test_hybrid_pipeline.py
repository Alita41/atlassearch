from pathlib import Path

from atlassearch.dense.hashing import HashingDenseRetriever
from atlassearch.fusion.rrf import reciprocal_rank_fusion
from atlassearch.native_bm25 import NativeBM25


def test_hybrid_pipeline_end_to_end(tmp_path):
    root = Path(__file__).resolve().parents[2]
    bm25 = NativeBM25(root / "build/cpp/atlassearch_bm25", root / "data/sample/documents.normalized.tsv")
    try:
        dense = HashingDenseRetriever(dim=256)
        dense.build([
            (1, "machine learning search systems hybrid lexical bm25 dense semantic retrieval"),
            (2, "sindh higher education policy undergraduate admission credit hours"),
            (3, "roman urdu search spelling variation code switching"),
            (4, "اردو سرچ معلومات بازیافت"),
            (5, "distributed retrieval architecture redis postgres docker api"),
        ], str(tmp_path / "d.npy"), str(tmp_path / "ids.json"))
        sparse = bm25.search("machine learning search", 5)
        dense_r = dense.search("machine learning search", 5)
        fused = reciprocal_rank_fusion([sparse, dense_r], top_k=5)
        assert fused
        assert fused[0][0] == 1
    finally:
        bm25.close()
