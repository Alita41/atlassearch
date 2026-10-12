# API

## `GET /v1/health`

Returns document count, metadata/cache status, and whether dense retrieval is enabled.

## `POST /v1/search`

```json
{
  "query": "machine learning search",
  "mode": "hybrid",
  "top_k": 10
}
```

`mode` is one of `bm25`, `dense`, or `hybrid`. Dense/hybrid require `ATLAS_ENABLE_DENSE=1` and a built dense index.

The response includes rank, document metadata, score, request latency, and cache status. Hybrid results also contain component-rank information from Reciprocal Rank Fusion.

## `GET /v1/autocomplete?q=mach&limit=8`

Returns trie-based token completions from the native sparse index.

## `GET /v1/documents/{id}`

Returns document metadata from PostgreSQL when configured, otherwise from the local JSONL fallback.
