from __future__ import annotations

import argparse
import json
from pathlib import Path

from atlassearch.dense.factory import create_dense_retriever
from atlassearch.preprocessing.normalize import normalize_text

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", type=Path, default=ROOT / "data/sample/documents.jsonl")
    ap.add_argument("--backend", choices=["e5", "hashing"], default="e5")
    ap.add_argument("--model", default="intfloat/multilingual-e5-base")
    ap.add_argument("--index-kind", choices=["flat", "hnsw"], default="flat")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "data/index")
    args = ap.parse_args()

    docs: list[tuple[int, str]] = []
    with args.docs.open(encoding="utf-8") as fh:
        for line in fh:
            obj = json.loads(line)
            text = normalize_text(f"{obj.get('title','')} {obj.get('text','')}")
            docs.append((int(obj["id"]), text))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "faiss" if args.backend == "e5" else "npy"
    index_path = args.out_dir / f"{args.backend}.{suffix}"
    ids_path = args.out_dir / f"{args.backend}.ids.json"
    retriever = create_dense_retriever(args.backend, args.model)
    if args.backend == "e5":
        retriever.build(docs, str(index_path), str(ids_path), index_kind=args.index_kind)
    else:
        retriever.build(docs, str(index_path), str(ids_path))
    print(json.dumps({
        "backend": args.backend,
        "documents": len(docs),
        "index": str(index_path),
        "ids": str(ids_path),
        "index_kind": args.index_kind if args.backend == "e5" else "exact-numpy",
    }, indent=2))


if __name__ == "__main__":
    main()
