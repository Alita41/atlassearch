# Architecture

```text
                         +------------------+
client ---------------->| Python REST API  |
                         +---+----------+---+
                             |          |
                  sparse     |          | dense
                             |          |
                    +--------v--+    +--v------------------+
                    | C++17 BM25|    | multilingual E5     |
                    | inverted  |    | + FAISS Flat/HNSW   |
                    | index     |    +----------+-----------+
                    | + Trie    |               |
                    +-----+-----+               |
                          |                     |
                          +----------+----------+
                                     |
                                  RRF fusion
                                     |
                    +----------------v----------------+
                    | hydrate metadata / cache result |
                    +-----------+---------------------+
                                |
                      +---------+---------+
                      |                   |
                  PostgreSQL            Redis
```

## Sparse core

The C++17 engine owns the inverted index, document-frequency statistics, Okapi BM25 scoring, top-K heap retrieval, and trie autocomplete. Python performs Unicode/text normalization before the corpus reaches the native index. The REST service keeps one native child process alive and communicates over a small line protocol, avoiding process startup per request.

## Dense retrieval

The production ML path uses `intfloat/multilingual-e5-base` through Sentence Transformers. Documents are encoded with the `passage:` prefix and queries with the `query:` prefix, embeddings are L2-normalized, and FAISS uses inner product so the exact FlatIP baseline is equivalent to cosine ranking. The code also supports HNSW for a later recall/latency sweep.

An explicit `hashing` backend is included only for offline integration tests. It lets the repository exercise dense/hybrid control flow when model downloads are unavailable; it must not be described as E5 or semantic ML.

## Hybrid retrieval

BM25 and dense rankings are fused with Reciprocal Rank Fusion. RRF avoids trying to calibrate fundamentally different BM25 and vector similarity score scales.

## Metadata and cache

PostgreSQL stores document metadata when `DATABASE_URL` is configured and automatically falls back to JSONL for a zero-friction demo. Redis caches versioned search responses when `REDIS_URL` is configured. Cache keys include the index version, retrieval mode, normalized query payload, and `top_k`.

## Ingestion and reproducibility

`build_corpus.py` parallelizes normalization using worker processes and emits a SHA-256 manifest. The native benchmark records document count, vocabulary size, index load time, query latency percentiles, and environment metadata. Synthetic corpora are clearly marked and should be used only for scale testing.
