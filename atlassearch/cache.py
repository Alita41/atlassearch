from __future__ import annotations
import hashlib
import json
import os


class QueryCache:
    def __init__(self, url: str | None = None, index_version: str = "dev-v1"):
        self.url = url or os.getenv("REDIS_URL")
        self.index_version = index_version
        self.client = None
        if self.url:
            try:
                import redis
                self.client = redis.from_url(self.url, decode_responses=True, socket_connect_timeout=1)
                self.client.ping()
            except Exception:
                self.client = None

    @property
    def enabled(self) -> bool:
        return self.client is not None

    def _key(self, mode: str, query: str, top_k: int) -> str:
        raw = json.dumps([self.index_version, mode, query, top_k], ensure_ascii=False, separators=(",", ":"))
        digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
        return f"atlassearch:{self.index_version}:search:{digest}"

    def get(self, mode: str, query: str, top_k: int):
        if not self.client:
            return None
        raw = self.client.get(self._key(mode, query, top_k))
        return json.loads(raw) if raw else None

    def set(self, mode: str, query: str, top_k: int, value, ttl: int = 300):
        if not self.client:
            return
        self.client.setex(self._key(mode, query, top_k), ttl, json.dumps(value, ensure_ascii=False))
