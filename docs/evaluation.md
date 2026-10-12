# Evaluation

The tiny bundled qrels file is a **smoke test**, not evidence that one retrieval method is superior. It exists to validate metric calculation and the BM25/dense/hybrid evaluation pipeline.

## Retrieval metrics

- Precision@5
- Recall@10 / Recall@20
- MRR@10
- nDCG@10
- language/script slices for English, Urdu script, Roman Urdu, and code-switched queries

## Systems metrics

- index preprocessing/build time
- native index load time
- p50/p95/p99 latency
- throughput under API concurrency
- memory usage where available
- cold vs warm cache latency

## Real evaluation protocol

1. Build a non-trivial corpus with source provenance.
2. Create at least 100–300 human-judged queries split into development and held-out test sets.
3. Tune BM25/RRF/HNSW only on development queries.
4. Compare BM25, E5+FlatIP, hybrid RRF, and optionally E5+HNSW.
5. Report per-language metrics and bootstrap confidence intervals for the final test comparison when possible.
6. Record hardware, package/model versions, corpus checksum, and Git commit with every latency result.

The bundled 5-document smoke set is intended only for correctness checks, not meaningful retrieval-quality comparisons.
