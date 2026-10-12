from __future__ import annotations
import argparse
import json
import statistics
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

QUERIES = [
    "machine learning search",
    "Sindh education policy",
    "university credit hour requirements",
    "roman urdu retrieval",
    "سرچ انجن اور تعلیم",
]


def one(url, mode, query):
    payload = json.dumps({"query": query, "mode": mode, "top_k": 10}).encode()
    req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=30) as resp:
        resp.read()
    return (time.perf_counter() - t0) * 1000


def percentile(values, p):
    if not values:
        return 0.0
    values = sorted(values)
    idx = min(len(values)-1, max(0, round((p/100)*(len(values)-1))))
    return values[idx]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8080/v1/search")
    ap.add_argument("--requests", type=int, default=100)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--mode", default="bm25")
    args = ap.parse_args()

    jobs = [(args.url, args.mode, QUERIES[i % len(QUERIES)]) for i in range(args.requests)]
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        lat = list(ex.map(lambda x: one(*x), jobs))
    elapsed = time.perf_counter() - t0
    print(json.dumps({
        "requests": args.requests,
        "concurrency": args.concurrency,
        "mode": args.mode,
        "p50_ms": round(percentile(lat, 50), 2),
        "p95_ms": round(percentile(lat, 95), 2),
        "p99_ms": round(percentile(lat, 99), 2),
        "qps": round(args.requests / elapsed, 2),
    }, indent=2))

if __name__ == "__main__":
    main()
