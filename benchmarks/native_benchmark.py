from __future__ import annotations

import argparse
import json
import os
import resource
import statistics
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
QUERIES = [
    "machine learning retrieval",
    "distributed systems redis docker",
    "roman urdu search",
    "education policy admissions",
]


def pct(values, p):
    values = sorted(values)
    if not values:
        return 0.0
    i = int(round((len(values) - 1) * p / 100.0))
    return values[max(0, min(i, len(values) - 1))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--binary", type=Path, default=ROOT / "build/cpp/atlassearch_bm25")
    ap.add_argument("--tsv", type=Path, required=True)
    ap.add_argument("--queries", type=int, default=1000)
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    t0 = time.perf_counter()
    proc = subprocess.Popen(
        [str(args.binary), str(args.tsv)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )
    ready = proc.stderr.readline().strip()
    load_ms = (time.perf_counter() - t0) * 1000
    if not ready.startswith("READY\t"):
        raise RuntimeError(ready)
    _, documents, vocabulary = ready.split("\t")
    lat = []
    for i in range(args.queries):
        q = QUERIES[i % len(QUERIES)]
        t1 = time.perf_counter()
        proc.stdin.write(f"S\t{args.top_k}\t{q}\n")
        proc.stdin.flush()
        line = proc.stdout.readline()
        if not line.startswith("R\t"):
            raise RuntimeError(line)
        lat.append((time.perf_counter() - t1) * 1000)
    proc.stdin.write("QUIT\n")
    proc.stdin.flush()
    proc.terminate()

    result = {
        "documents": int(documents),
        "vocabulary": int(vocabulary),
        "queries": args.queries,
        "index_load_ms": round(load_ms, 3),
        "latency_ms": {
            "mean": round(statistics.mean(lat), 4),
            "p50": round(pct(lat, 50), 4),
            "p95": round(pct(lat, 95), 4),
            "p99": round(pct(lat, 99), 4),
        },
        "max_rss_kb": int(resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss),
        "environment": {
            "cpu_count": os.cpu_count(),
            "note": "Reference measurement from the machine that ran this script; rerun on your own hardware before using numbers externally.",
        },
    }
    text = json.dumps(result, indent=2)
    print(text)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
