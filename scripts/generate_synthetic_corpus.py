"""Generate synthetic documents ONLY for load/scale testing.

Do not use these generated rows to claim a real-world corpus size on a résumé.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

TEMPLATES = [
    "machine learning retrieval ranking bm25 dense search document {i}",
    "sindh education policy university credit hours admissions document {i}",
    "roman urdu search engine query spelling variation document {i}",
    "distributed systems caching postgres redis docker api document {i}",
]

ap = argparse.ArgumentParser()
ap.add_argument("--count", type=int, default=100000)
ap.add_argument("--out", type=Path, default=Path("data/synthetic.jsonl"))
args = ap.parse_args()
args.out.parent.mkdir(parents=True, exist_ok=True)
with args.out.open("w", encoding="utf-8") as fh:
    for i in range(1, args.count + 1):
        text = TEMPLATES[i % len(TEMPLATES)].format(i=i)
        fh.write(json.dumps({"id": i, "title": f"Synthetic {i}", "text": text, "source": "synthetic-load-test"}) + "\n")
print(f"wrote {args.count} synthetic docs to {args.out}")
