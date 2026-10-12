from __future__ import annotations
import os
from pathlib import Path
import subprocess
import threading
from .preprocessing.normalize import normalize_text


class NativeBM25:
    def __init__(self, binary: str | Path, docs_tsv: str | Path):
        self.binary = str(binary)
        self.docs_tsv = str(docs_tsv)
        self._lock = threading.Lock()
        self.proc = subprocess.Popen(
            [self.binary, self.docs_tsv],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        ready = self.proc.stderr.readline().strip()
        if not ready.startswith("READY\t"):
            raise RuntimeError(f"native engine failed to start: {ready}")

    def _request(self, line: str) -> str:
        if not self.proc.stdin or not self.proc.stdout:
            raise RuntimeError("native engine unavailable")
        with self._lock:
            self.proc.stdin.write(line + "\n")
            self.proc.stdin.flush()
            return self.proc.stdout.readline().rstrip("\n")

    def search(self, query: str, top_k: int = 10):
        query = normalize_text(query).replace("\t", " ").replace("\n", " ")
        raw = self._request(f"S\t{top_k}\t{query}")
        if not raw.startswith("R\t"):
            raise RuntimeError(raw)
        payload = raw[2:]
        out = []
        if payload:
            for item in payload.split(","):
                doc_id, score = item.split(":", 1)
                out.append((int(doc_id), float(score)))
        return out

    def autocomplete(self, prefix: str, limit: int = 8):
        prefix = normalize_text(prefix).replace("\t", " ").replace("\n", " ")
        raw = self._request(f"A\t{limit}\t{prefix}")
        if not raw.startswith("C\t"):
            raise RuntimeError(raw)
        payload = raw[2:]
        return payload.split("|") if payload else []

    def close(self):
        if self.proc.poll() is None and self.proc.stdin:
            try:
                self.proc.stdin.write("QUIT\n")
                self.proc.stdin.flush()
            except Exception:
                pass
            self.proc.terminate()
