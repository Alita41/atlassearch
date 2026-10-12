from __future__ import annotations
import json
import os
from pathlib import Path


class MetadataStore:
    def __init__(self, fallback_jsonl: Path, database_url: str | None = None):
        self.fallback_jsonl = fallback_jsonl
        self.database_url = database_url or os.getenv("DATABASE_URL")
        self.conn = None
        self.docs = self._load_jsonl()
        if self.database_url:
            self._try_connect()

    def _load_jsonl(self):
        docs = {}
        with self.fallback_jsonl.open(encoding="utf-8") as fh:
            for line in fh:
                obj = json.loads(line)
                docs[int(obj["id"])] = obj
        return docs

    def _try_connect(self):
        try:
            import psycopg
            self.conn = psycopg.connect(self.database_url, connect_timeout=2, autocommit=True)
            self._ensure_schema()
            self._seed_fallback_docs()
        except Exception:
            self.conn = None

    @property
    def backend(self):
        return "postgres" if self.conn is not None else "jsonl"

    def _ensure_schema(self):
        if not self.conn:
            return
        with self.conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    id BIGINT PRIMARY KEY,
                    title TEXT NOT NULL,
                    text TEXT NOT NULL,
                    source TEXT,
                    language TEXT,
                    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
                )
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS ingestion_runs (
                    id BIGSERIAL PRIMARY KEY,
                    corpus_name TEXT NOT NULL,
                    document_count BIGINT NOT NULL,
                    index_version TEXT,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
                )
            """)

    def _seed_fallback_docs(self):
        if not self.conn:
            return
        with self.conn.cursor() as cur:
            for doc in self.docs.values():
                extra = {k: v for k, v in doc.items() if k not in {"id", "title", "text", "source", "language"}}
                cur.execute("""
                    INSERT INTO documents (id, title, text, source, language, metadata)
                    VALUES (%s, %s, %s, %s, %s, %s::jsonb)
                    ON CONFLICT (id) DO UPDATE SET
                        title = EXCLUDED.title,
                        text = EXCLUDED.text,
                        source = EXCLUDED.source,
                        language = EXCLUDED.language,
                        metadata = EXCLUDED.metadata,
                        updated_at = now()
                """, (
                    int(doc["id"]), doc.get("title", ""), doc.get("text", ""),
                    doc.get("source"), doc.get("language"), json.dumps(extra)
                ))

    def get(self, doc_id: int):
        if self.conn:
            with self.conn.cursor() as cur:
                cur.execute("SELECT id, title, text, source, language, metadata FROM documents WHERE id=%s", (doc_id,))
                row = cur.fetchone()
                if row:
                    return {"id": int(row[0]), "title": row[1], "text": row[2], "source": row[3], "language": row[4], **(row[5] or {})}
        return self.docs.get(doc_id)

    def count(self):
        if self.conn:
            with self.conn.cursor() as cur:
                cur.execute("SELECT COUNT(*) FROM documents")
                return int(cur.fetchone()[0])
        return len(self.docs)
