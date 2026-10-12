import sys
import types
import numpy as np

from atlassearch.dense.e5_faiss import DenseRetriever


class FakeModel:
    def __init__(self, name):
        self.name = name
        self.seen = []

    def encode(self, texts, **kwargs):
        self.seen.extend(texts)
        out = []
        for text in texts:
            if "machine" in text:
                out.append([1.0, 0.0, 0.0])
            elif "database" in text:
                out.append([0.0, 1.0, 0.0])
            else:
                out.append([0.0, 0.0, 1.0])
        return np.asarray(out, dtype=np.float32)


class FakeIndexFlatIP:
    def __init__(self, dim):
        self.dim = dim
        self.matrix = np.empty((0, dim), dtype=np.float32)

    @property
    def ntotal(self):
        return self.matrix.shape[0]

    def add(self, matrix):
        self.matrix = np.asarray(matrix, dtype=np.float32)

    def search(self, q, k):
        scores = self.matrix @ q[0]
        idx = np.argsort(-scores)[:k]
        return scores[idx][None, :], idx.astype(np.int64)[None, :]


_FAKE_WRITTEN = {}

def test_e5_prefixes_and_faiss_mapping(monkeypatch, tmp_path):
    st = types.ModuleType("sentence_transformers")
    st.SentenceTransformer = FakeModel
    monkeypatch.setitem(sys.modules, "sentence_transformers", st)

    faiss = types.ModuleType("faiss")
    faiss.IndexFlatIP = FakeIndexFlatIP
    faiss.write_index = lambda index, path: _FAKE_WRITTEN.__setitem__(str(path), index)
    faiss.read_index = lambda path: _FAKE_WRITTEN[str(path)]
    monkeypatch.setitem(sys.modules, "faiss", faiss)

    r = DenseRetriever("fake/e5")
    index = tmp_path / "x.faiss"
    ids = tmp_path / "ids.json"
    r.build([(7, "machine search"), (9, "database storage")], str(index), str(ids))
    assert r.model.seen[0].startswith("passage: ")

    loaded = DenseRetriever("fake/e5")
    loaded.load(str(index), str(ids))
    result = loaded.search("machine retrieval", 1)
    assert loaded.model.seen[0].startswith("query: ")
    assert result[0][0] == 7
