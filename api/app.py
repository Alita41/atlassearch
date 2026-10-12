from __future__ import annotations
import atexit
import json
import os
from pathlib import Path
from time import perf_counter

from flask import Flask, jsonify, request

from atlassearch.native_bm25 import NativeBM25
from atlassearch.fusion.rrf import reciprocal_rank_fusion
from atlassearch.cache import QueryCache
from atlassearch.metadata_store import MetadataStore

ROOT = Path(__file__).resolve().parents[1]
DOCS_JSONL = Path(os.getenv("ATLAS_DOCS_JSONL", ROOT / "data/sample/documents.jsonl"))
DOCS_TSV = Path(os.getenv("ATLAS_DOCS_TSV", ROOT / "data/sample/documents.normalized.tsv"))
BM25_BIN = Path(os.getenv("ATLAS_BM25_BIN", ROOT / "build/cpp/atlassearch_bm25"))

app = Flask(__name__)


STORE = MetadataStore(DOCS_JSONL)
CACHE = QueryCache(index_version=os.getenv("ATLAS_INDEX_VERSION", "dev-v1"))
BM25 = NativeBM25(BM25_BIN, DOCS_TSV)
atexit.register(BM25.close)

DENSE = None
DENSE_BACKEND = None
if os.getenv("ATLAS_ENABLE_DENSE", "0") == "1":
    from atlassearch.dense.factory import create_dense_retriever
    DENSE_BACKEND = os.getenv("ATLAS_DENSE_BACKEND", "e5").strip().lower()
    DENSE = create_dense_retriever(
        DENSE_BACKEND,
        os.getenv("ATLAS_E5_MODEL", "intfloat/multilingual-e5-base"),
    )
    default_ext = "faiss" if DENSE_BACKEND == "e5" else "npy"
    DENSE.load(
        os.getenv("ATLAS_DENSE_INDEX", str(ROOT / f"data/index/{DENSE_BACKEND}.{default_ext}")),
        os.getenv("ATLAS_DENSE_IDS", str(ROOT / f"data/index/{DENSE_BACKEND}.ids.json")),
    )


def hydrate(results):
    hydrated = []
    for rank, item in enumerate(results, start=1):
        if len(item) == 2:
            doc_id, score = item
            meta = {"score": score}
        else:
            doc_id, score, ranks = item
            meta = {"score": score, "component_ranks": ranks}
        doc = STORE.get(int(doc_id)) or {"id": int(doc_id)}
        hydrated.append({"rank": rank, **doc, **meta})
    return hydrated


@app.get("/v1/health")
def health():
    return jsonify({"status": "ok", "documents": STORE.count(), "dense_enabled": DENSE is not None, "dense_backend": DENSE_BACKEND, "metadata_backend": STORE.backend, "cache_enabled": CACHE.enabled})


@app.get("/v1/autocomplete")
def autocomplete():
    q = request.args.get("q", "")
    limit = min(max(int(request.args.get("limit", 8)), 1), 20)
    return jsonify({"query": q, "suggestions": BM25.autocomplete(q, limit)})


@app.post("/v1/search")
def search():
    body = request.get_json(force=True, silent=False)
    q = str(body.get("query", "")).strip()
    if not q:
        return jsonify({"error": "query is required"}), 400
    top_k = min(max(int(body.get("top_k", 10)), 1), 50)
    mode = body.get("mode", "bm25")
    started = perf_counter()
    cached = CACHE.get(mode, q, top_k)
    if cached is not None:
        cached["cache"] = "hit"
        return jsonify(cached)

    if mode == "bm25":
        results = BM25.search(q, top_k)
    elif mode == "dense":
        if DENSE is None:
            return jsonify({"error": "dense retrieval is not enabled"}), 503
        results = DENSE.search(q, top_k)
    elif mode == "hybrid":
        if DENSE is None:
            return jsonify({"error": "dense retrieval is not enabled"}), 503
        sparse = BM25.search(q, max(50, top_k))
        dense = DENSE.search(q, max(50, top_k))
        results = reciprocal_rank_fusion([sparse, dense], top_k=top_k)
    else:
        return jsonify({"error": "mode must be bm25, dense, or hybrid"}), 400

    elapsed = (perf_counter() - started) * 1000
    response = {
        "query": q,
        "mode": mode,
        "latency_ms": round(elapsed, 3),
        "cache": "miss",
        "results": hydrate(results),
    }
    CACHE.set(mode, q, top_k, response)
    return jsonify(response)


@app.get("/v1/documents/<int:doc_id>")
def document(doc_id: int):
    doc = STORE.get(doc_id)
    if not doc:
        return jsonify({"error": "not found"}), 404
    return jsonify(doc)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "8080")), debug=False)
