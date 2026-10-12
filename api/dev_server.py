"""Dependency-free smoke-test REST server for the BM25 core.

The production-facing app remains Flask/Gunicorn. This file exists so the native
engine can be demonstrated even on machines where Python packages have not yet
been installed.
"""
from __future__ import annotations
import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from atlassearch.native_bm25 import NativeBM25

ROOT = Path(__file__).resolve().parents[1]
ENGINE = NativeBM25(ROOT / "build/cpp/atlassearch_bm25", ROOT / "data/sample/documents.normalized.tsv")
DOCS = {}
for line in (ROOT / "data/sample/documents.jsonl").read_text(encoding="utf-8").splitlines():
    obj = json.loads(line)
    DOCS[int(obj["id"])] = obj


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, obj):
        data = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/v1/health":
            return self._send(200, {"status": "ok", "documents": len(DOCS), "server": "stdlib-smoke"})
        if parsed.path == "/v1/autocomplete":
            qs = parse_qs(parsed.query)
            q = qs.get("q", [""])[0]
            return self._send(200, {"query": q, "suggestions": ENGINE.autocomplete(q, 8)})
        return self._send(404, {"error": "not found"})

    def do_POST(self):
        if self.path != "/v1/search":
            return self._send(404, {"error": "not found"})
        length = int(self.headers.get("Content-Length", "0"))
        body = json.loads(self.rfile.read(length) or b"{}")
        query = str(body.get("query", "")).strip()
        if not query:
            return self._send(400, {"error": "query is required"})
        top_k = min(max(int(body.get("top_k", 10)), 1), 50)
        results = []
        for rank, (doc_id, score) in enumerate(ENGINE.search(query, top_k), start=1):
            results.append({"rank": rank, **DOCS.get(doc_id, {"id": doc_id}), "score": score})
        return self._send(200, {"query": query, "mode": "bm25", "results": results})

    def log_message(self, fmt, *args):
        pass


if __name__ == "__main__":
    try:
        port = int(os.getenv("PORT", "8080"))
        ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()
    finally:
        ENGINE.close()
