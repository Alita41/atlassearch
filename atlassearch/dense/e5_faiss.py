from __future__ import annotations

import json
from pathlib import Path

import numpy as np


class DenseRetriever:
    """multilingual-E5 encoder backed by FAISS.

    Supports an exact cosine/IP baseline (``flat``) and HNSW for a recall/latency
    sweep. Heavy dependencies are imported lazily so the sparse core remains
    runnable without ML packages.
    """

    def __init__(self, model_name: str = "intfloat/multilingual-e5-base"):
        self.model_name = model_name
        self.model = None
        self.index = None
        self.doc_ids: list[int] = []
        self.index_kind = "flat"

    def _ensure_model(self):
        if self.model is None:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(self.model_name)

    def encode(self, texts: list[str], is_query: bool = False) -> np.ndarray:
        self._ensure_model()
        prefix = "query: " if is_query else "passage: "
        vecs = self.model.encode(
            [prefix + t for t in texts],
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return np.asarray(vecs, dtype="float32")

    def build(
        self,
        docs: list[tuple[int, str]],
        index_path: str,
        ids_path: str,
        *,
        index_kind: str = "flat",
        hnsw_m: int = 32,
        ef_construction: int = 200,
    ):
        import faiss

        self.doc_ids = [int(doc_id) for doc_id, _ in docs]
        matrix = self.encode([text for _, text in docs], is_query=False)
        if matrix.ndim != 2 or matrix.shape[0] != len(self.doc_ids):
            raise ValueError("embedding matrix shape mismatch")
        dim = int(matrix.shape[1])
        if index_kind == "flat":
            index = faiss.IndexFlatIP(dim)
        elif index_kind == "hnsw":
            index = faiss.IndexHNSWFlat(dim, hnsw_m, faiss.METRIC_INNER_PRODUCT)
            index.hnsw.efConstruction = ef_construction
        else:
            raise ValueError("index_kind must be 'flat' or 'hnsw'")
        index.add(matrix)
        Path(index_path).parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(index, str(index_path))
        Path(ids_path).write_text(json.dumps(self.doc_ids), encoding="utf-8")
        meta = {
            "backend": "e5-faiss",
            "model": self.model_name,
            "index_kind": index_kind,
            "documents": len(self.doc_ids),
            "dimension": dim,
        }
        Path(str(index_path) + ".meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        self.index = index
        self.index_kind = index_kind

    def load(self, index_path: str, ids_path: str, *, ef_search: int = 64):
        import faiss

        self.index = faiss.read_index(str(index_path))
        self.doc_ids = [int(x) for x in json.loads(Path(ids_path).read_text(encoding="utf-8"))]
        if int(self.index.ntotal) != len(self.doc_ids):
            raise ValueError("FAISS index/id mapping mismatch")
        if hasattr(self.index, "hnsw"):
            self.index.hnsw.efSearch = ef_search
            self.index_kind = "hnsw"
        else:
            self.index_kind = "flat"

    def search(self, query: str, top_k: int = 10):
        if self.index is None:
            raise RuntimeError("dense index not loaded")
        if top_k <= 0:
            return []
        q = self.encode([query], is_query=True)
        scores, idxs = self.index.search(q, min(top_k, len(self.doc_ids)))
        out = []
        for score, idx in zip(scores[0], idxs[0]):
            if idx < 0:
                continue
            out.append((int(self.doc_ids[int(idx)]), float(score)))
        return out
