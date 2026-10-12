# AtlasSearch

[![CI](https://github.com/Alita41/atlassearch/actions/workflows/ci.yml/badge.svg)](https://github.com/Alita41/atlassearch/actions/workflows/ci.yml)

A hybrid **Urdu/English search and retrieval engine** built around a custom C++17 sparse-search core, with optional multilingual dense retrieval.

AtlasSearch combines an inverted index, Okapi BM25 ranking, trie-based autocomplete, and top-K retrieval with a Python REST API, PostgreSQL metadata, Redis caching, and an optional multilingual-E5 + FAISS retrieval path fused with BM25 using Reciprocal Rank Fusion (RRF).

## Why this project

Modern search systems often hide the core retrieval logic behind large frameworks. AtlasSearch keeps the important pieces visible and testable: indexing, ranking, caching, metadata storage, hybrid retrieval, evaluation, and latency benchmarking.

## Features

- Custom **C++17 inverted index** and **BM25 ranker**
- Top-K heap retrieval
- Trie-based autocomplete
- Persistent C++ ↔ Python process integration
- Urdu-script and Roman-Urdu-safe normalization
- REST endpoints for sparse, dense, and hybrid search
- PostgreSQL document metadata store with JSONL fallback
- Redis result caching with index-versioned keys
- Optional **multilingual-E5 + FAISS** dense retrieval
- Reciprocal Rank Fusion for hybrid ranking
- Docker and Docker Compose
- Retrieval evaluation with Precision@K, Recall@K, MRR, and nDCG
- Native and API latency benchmarks
- C++ unit tests plus Python unit/integration tests
- GitHub Actions CI

## Architecture

```text
                         +------------------+
client ---------------->| Python REST API  |
                         +---+----------+---+
                             |          |
                    +--------v--+    +--v------------------+
                    | C++17 BM25|    | multilingual E5     |
                    | + Trie    |    | + FAISS             |
                    +-----+-----+    +----------+-----------+
                          |                     |
                          +----------+----------+
                                     |
                                  RRF fusion
                                     |
                             PostgreSQL / Redis
```

See [`docs/architecture.md`](docs/architecture.md) for a deeper walkthrough of the indexing, ranking, storage, and hybrid-retrieval design.

## Repository layout

```text
cpp/                  C++ search engine and native tests
atlassearch/          Python package: bridges, dense retrieval, fusion, cache/storage
api/                  REST API and lightweight BM25 smoke server
scripts/              corpus/index builders and evaluation tools
benchmarks/           native and API latency benchmarks
eval/                 retrieval metrics and qrels
infra/                Dockerfiles and Compose configuration
tests/                Python unit/integration tests
docs/                 architecture, API, and evaluation documentation
artifacts/             reproducible benchmark/evaluation outputs
```

## Quick start

### Sparse search

Requirements: Python 3.10+, CMake 3.16+, and a C++17 compiler.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
make test
make run
```

Search:

```bash
curl -X POST http://127.0.0.1:8080/v1/search \
  -H 'Content-Type: application/json' \
  -d '{"query":"machine learning search","mode":"bm25","top_k":5}'
```

Autocomplete:

```bash
curl 'http://127.0.0.1:8080/v1/autocomplete?q=mach'
```

## Dense + hybrid search

Install the ML extras and a compatible PyTorch build:

```bash
pip install -r requirements-ml.txt
PYTHONPATH=. python scripts/check_ml_runtime.py
PYTHONPATH=. python scripts/build_dense_index.py --backend e5 --index-kind flat
ATLAS_ENABLE_DENSE=1 \
ATLAS_DENSE_BACKEND=e5 \
PYTHONPATH=. python api/app.py
```

Use `"mode":"dense"` or `"mode":"hybrid"` in `/v1/search`.

For an HNSW index:

```bash
PYTHONPATH=. python scripts/build_dense_index.py --backend e5 --index-kind hnsw
```

The repository also includes a deterministic feature-hashing dense backend for offline integration testing. It exercises the complete dense → fusion → API control flow without pretending to be a learned semantic model.

## Evaluation

```bash
make evaluate
```

The evaluation runner reports:

- Precision@K
- Recall@K
- Mean Reciprocal Rank (MRR)
- nDCG

See [`docs/evaluation.md`](docs/evaluation.md) for methodology and guidance on building a meaningful judged query set.

## 100K-document benchmark

Generate and benchmark a synthetic 100K-document corpus:

```bash
make benchmark-100k
```

A reference run produced:

| Metric | Result |
|---|---:|
| Synthetic documents | 100,000 |
| Vocabulary terms | 100,029 |
| Parallel normalization | 2.158 s |
| Normalization throughput | 46,340 docs/s |
| Native index load | 606.29 ms |
| Query p50 | 5.0836 ms |
| Query p95 | 11.2583 ms |
| Query p99 | 13.6682 ms |

Raw outputs are kept under `artifacts/` so the results are inspectable. These figures are machine-specific reference measurements; rerun the benchmark on your own hardware for comparison.

## Tests

```bash
make test
```

The test suite covers:

- BM25 search behavior
- normalization
- ranking metrics
- Reciprocal Rank Fusion
- offline dense retrieval
- E5 query/passage prefix handling
- FAISS ID mapping
- end-to-end hybrid retrieval
- REST API behavior

## Docker

Run the API with PostgreSQL and Redis:

```bash
docker compose -f infra/compose.yaml up --build
```

An ML-capable image definition is available at `infra/Dockerfile.ml`.

## Design notes

- **Sparse retrieval:** custom C++17 index and BM25 implementation rather than Elasticsearch/OpenSearch.
- **Dense retrieval:** multilingual-E5 embeddings with FAISS FlatIP or HNSW.
- **Hybrid ranking:** Reciprocal Rank Fusion keeps sparse and dense score scales independent.
- **Caching:** Redis keys include the index version so stale result sets are naturally invalidated after rebuilds.
- **Metadata:** PostgreSQL stores document metadata separately from the native retrieval index.

## License

MIT
