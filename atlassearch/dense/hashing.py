from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np

_TOKEN = re.compile(r"\w+", re.UNICODE)


class HashingDenseRetriever:
    """Dependency-light dense-like backend for offline integration tests.

    This is intentionally *not* presented as a semantic-ML replacement for E5.
    It uses signed feature hashing over word and character n-grams so the full
    dense/hybrid API can be validated without downloading model weights.
    """

    def __init__(self, dim: int = 512):
        if dim <= 0:
            raise ValueError("dim must be positive")
        self.dim = dim
        self.matrix: np.ndarray | None = None
        self.doc_ids: list[int] = []

    @staticmethod
    def _stable_hash(feature: str) -> tuple[int, float]:
        digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
        value = int.from_bytes(digest, "little", signed=False)
        return value, -1.0 if (value >> 63) else 1.0

    def _vectorize_one(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        text = (text or "").lower()
        tokens = _TOKEN.findall(text)
        features: list[str] = []
        for tok in tokens:
            features.append(f"w:{tok}")
            padded = f"^{tok}$"
            for n in (3, 4, 5):
                if len(padded) >= n:
                    features.extend(f"c{n}:{padded[i:i+n]}" for i in range(len(padded) - n + 1))
        for feature in features:
            h, sign = self._stable_hash(feature)
            v[h % self.dim] += sign
        norm = float(np.linalg.norm(v))
        if norm:
            v /= norm
        return v

    def encode(self, texts: list[str], is_query: bool = False) -> np.ndarray:
        del is_query
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return np.vstack([self._vectorize_one(t) for t in texts]).astype(np.float32, copy=False)

    def build(self, docs: list[tuple[int, str]], index_path: str, ids_path: str):
        self.doc_ids = [int(doc_id) for doc_id, _ in docs]
        self.matrix = self.encode([text for _, text in docs], is_query=False)
        Path(index_path).parent.mkdir(parents=True, exist_ok=True)
        with Path(index_path).open("wb") as fh:
            np.save(fh, self.matrix, allow_pickle=False)
        Path(ids_path).write_text(json.dumps(self.doc_ids), encoding="utf-8")

    def load(self, index_path: str, ids_path: str):
        with Path(index_path).open("rb") as fh:
            self.matrix = np.load(fh, allow_pickle=False).astype(np.float32, copy=False)
        self.doc_ids = [int(x) for x in json.loads(Path(ids_path).read_text(encoding="utf-8"))]
        if self.matrix.ndim != 2 or self.matrix.shape[0] != len(self.doc_ids):
            raise ValueError("hashing index/id mapping mismatch")
        self.dim = int(self.matrix.shape[1])

    def search(self, query: str, top_k: int = 10):
        if self.matrix is None:
            raise RuntimeError("hashing index not loaded")
        if top_k <= 0 or not self.doc_ids:
            return []
        q = self.encode([query], is_query=True)[0]
        scores = self.matrix @ q
        k = min(top_k, len(self.doc_ids))
        # argpartition keeps this O(N) before ordering only the candidate set.
        if k == len(self.doc_ids):
            idxs = np.argsort(-scores)
        else:
            candidate = np.argpartition(-scores, k - 1)[:k]
            idxs = candidate[np.argsort(-scores[candidate])]
        return [(self.doc_ids[int(i)], float(scores[int(i)])) for i in idxs[:k]]
