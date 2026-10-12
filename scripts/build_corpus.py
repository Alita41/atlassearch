from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from atlassearch.preprocessing.normalize import normalize_text


def normalize_record(line: str) -> tuple[int, str]:
    obj = json.loads(line)
    return int(obj["id"]), normalize_text(f"{obj.get('title','')} {obj.get('text','')}")


def main():
    ap = argparse.ArgumentParser(description="Normalize a JSONL corpus into the TSV consumed by the C++ index.")
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=max(1, min(4, os.cpu_count() or 1)))
    ap.add_argument("--manifest", type=Path)
    args = ap.parse_args()

    started = time.perf_counter()
    raw_lines = [line for line in args.input.read_text(encoding="utf-8").splitlines() if line.strip()]
    if args.workers == 1:
        rows = map(normalize_record, raw_lines)
    else:
        ex = ProcessPoolExecutor(max_workers=args.workers)
        rows = ex.map(normalize_record, raw_lines, chunksize=256)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    h = hashlib.sha256()
    with args.output.open("w", encoding="utf-8") as out:
        for doc_id, normalized in rows:
            encoded = f"{doc_id}\t{normalized}\n".encode("utf-8")
            out.write(encoded.decode("utf-8"))
            h.update(encoded)
            count += 1
    if args.workers != 1:
        ex.shutdown()
    elapsed = time.perf_counter() - started
    manifest_path = args.manifest or args.output.with_suffix(args.output.suffix + ".manifest.json")
    manifest = {
        "source": str(args.input),
        "output": str(args.output),
        "documents": count,
        "workers": args.workers,
        "sha256": h.hexdigest(),
        "build_seconds": round(elapsed, 6),
        "documents_per_second": round(count / elapsed, 2) if elapsed else None,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
