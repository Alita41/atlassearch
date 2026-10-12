from __future__ import annotations

import argparse
import json
from pathlib import Path

from atlassearch.dense.factory import create_dense_retriever
from atlassearch.fusion.rrf import reciprocal_rank_fusion
from atlassearch.native_bm25 import NativeBM25
from eval.metrics import precision_at_k, recall_at_k, mrr_at_k, ndcg_at_k

ROOT = Path(__file__).resolve().parents[1]


def mean(values):
    return sum(values) / len(values) if values else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qrels", type=Path, default=ROOT / "eval/qrels/sample_qrels.json")
    ap.add_argument("--docs-tsv", type=Path, default=ROOT / "data/sample/documents.normalized.tsv")
    ap.add_argument("--bm25-bin", type=Path, default=ROOT / "build/cpp/atlassearch_bm25")
    ap.add_argument("--dense-backend", choices=["none", "hashing", "e5"], default="hashing")
    ap.add_argument("--dense-index", type=Path, default=ROOT / "data/index/hashing.npy")
    ap.add_argument("--dense-ids", type=Path, default=ROOT / "data/index/hashing.ids.json")
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    qrels_raw = json.loads(args.qrels.read_text(encoding="utf-8"))
    qrels = {q: {int(doc): int(rel) for doc, rel in rels.items()} for q, rels in qrels_raw.items()}
    bm25 = NativeBM25(args.bm25_bin, args.docs_tsv)
    dense = None
    if args.dense_backend != "none":
        dense = create_dense_retriever(args.dense_backend)
        dense.load(str(args.dense_index), str(args.dense_ids))

    modes = ["bm25"] + (["dense", "hybrid"] if dense else [])
    agg = {m: {"p5": [], "r10": [], "mrr10": [], "ndcg10": []} for m in modes}
    per_query = []
    try:
        for query, graded in qrels.items():
            relevant = [doc_id for doc_id, rel in graded.items() if rel > 0]
            sparse = bm25.search(query, max(50, args.top_k))
            dense_r = dense.search(query, max(50, args.top_k)) if dense else []
            rankings = {"bm25": sparse[:args.top_k]}
            if dense:
                rankings["dense"] = dense_r[:args.top_k]
                rankings["hybrid"] = reciprocal_rank_fusion([sparse, dense_r], top_k=args.top_k)
            row = {"query": query}
            for mode, ranking in rankings.items():
                ranked_ids = [int(x[0]) for x in ranking]
                metrics = {
                    "p5": precision_at_k(ranked_ids, relevant, 5),
                    "r10": recall_at_k(ranked_ids, relevant, 10),
                    "mrr10": mrr_at_k(ranked_ids, relevant, 10),
                    "ndcg10": ndcg_at_k(ranked_ids, graded, 10),
                }
                row[mode] = {"ids": ranked_ids, **{k: round(v, 6) for k, v in metrics.items()}}
                for k, v in metrics.items():
                    agg[mode][k].append(v)
            per_query.append(row)
    finally:
        bm25.close()

    summary = {mode: {k: round(mean(vals), 6) for k, vals in metrics.items()} for mode, metrics in agg.items()}
    result = {"queries": len(qrels), "summary": summary, "per_query": per_query}
    text = json.dumps(result, indent=2, ensure_ascii=False)
    print(text)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
