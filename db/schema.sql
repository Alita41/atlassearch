CREATE TABLE IF NOT EXISTS documents (
    id BIGINT PRIMARY KEY,
    title TEXT NOT NULL,
    text TEXT NOT NULL,
    source TEXT,
    language TEXT,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS passages (
    id BIGSERIAL PRIMARY KEY,
    document_id BIGINT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    passage_key TEXT UNIQUE NOT NULL,
    ordinal INTEGER NOT NULL,
    raw_text TEXT NOT NULL,
    normalized_text TEXT NOT NULL,
    language TEXT,
    checksum TEXT NOT NULL,
    token_count INTEGER,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb
);

CREATE TABLE IF NOT EXISTS ingestion_runs (
    id BIGSERIAL PRIMARY KEY,
    corpus_name TEXT NOT NULL,
    document_count BIGINT NOT NULL,
    passage_count BIGINT,
    index_version TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS evaluation_runs (
    id BIGSERIAL PRIMARY KEY,
    run_name TEXT NOT NULL,
    config JSONB NOT NULL,
    metrics JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
