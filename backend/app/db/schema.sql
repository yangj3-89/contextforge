-- ContextForge PostgreSQL schema. {dim} is substituted with the embedding dimension.
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS documents (
    id            TEXT PRIMARY KEY,
    filename      TEXT NOT NULL,
    source_type   TEXT NOT NULL,
    content_hash  TEXT NOT NULL,
    title         TEXT NOT NULL,
    text          TEXT NOT NULL,
    metadata      JSONB NOT NULL DEFAULT '{{}}',
    chunk_count   INT NOT NULL DEFAULT 0,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (filename, content_hash)
);

CREATE TABLE IF NOT EXISTS chunks (
    id           TEXT PRIMARY KEY,
    document_id  TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    chunk_index  INT NOT NULL,
    text         TEXT NOT NULL,
    search_text  TEXT NOT NULL,
    token_count  INT NOT NULL,
    char_start   INT NOT NULL,
    char_end     INT NOT NULL,
    metadata     JSONB NOT NULL DEFAULT '{{}}',
    embedding    vector({dim}),
    tsv          tsvector GENERATED ALWAYS AS (to_tsvector('english', search_text)) STORED,
    UNIQUE (document_id, chunk_index)
);
CREATE INDEX IF NOT EXISTS chunks_tsv_gin ON chunks USING GIN (tsv);
CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw ON chunks USING hnsw (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS chunks_document_idx ON chunks (document_id);

CREATE TABLE IF NOT EXISTS entities (
    id              TEXT PRIMARY KEY,
    canonical_name  TEXT NOT NULL,
    entity_type     TEXT NOT NULL,
    aliases         TEXT[] NOT NULL DEFAULT '{{}}',
    mention_count   INT NOT NULL DEFAULT 0,
    document_count  INT NOT NULL DEFAULT 0,
    metadata        JSONB NOT NULL DEFAULT '{{}}'
);
CREATE INDEX IF NOT EXISTS entities_type_idx ON entities (entity_type);

CREATE TABLE IF NOT EXISTS entity_mentions (
    id           TEXT PRIMARY KEY,
    chunk_id     TEXT NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
    document_id  TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    surface      TEXT NOT NULL,
    entity_type  TEXT NOT NULL,
    char_start   INT NOT NULL,
    char_end     INT NOT NULL,
    extractor    TEXT NOT NULL,
    confidence   REAL NOT NULL,
    entity_id    TEXT REFERENCES entities(id) ON DELETE SET NULL
);
CREATE INDEX IF NOT EXISTS mentions_entity_idx ON entity_mentions (entity_id);
CREATE INDEX IF NOT EXISTS mentions_chunk_idx ON entity_mentions (chunk_id);

CREATE TABLE IF NOT EXISTS entity_relationships (
    id                   TEXT PRIMARY KEY,
    source_entity_id     TEXT NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
    target_entity_id     TEXT NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
    relationship_type    TEXT NOT NULL,
    confidence           REAL NOT NULL,
    supporting_chunk_id  TEXT NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
    evidence             TEXT NOT NULL,
    method               TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS relationships_source_idx ON entity_relationships (source_entity_id);
CREATE INDEX IF NOT EXISTS relationships_target_idx ON entity_relationships (target_entity_id);

CREATE TABLE IF NOT EXISTS resolution_decisions (
    id             SERIAL PRIMARY KEY,
    left_surface   TEXT NOT NULL,
    right_surface  TEXT NOT NULL,
    entity_type    TEXT NOT NULL,
    score          REAL NOT NULL,
    decision       TEXT NOT NULL,
    features       JSONB NOT NULL,
    reason         TEXT NOT NULL
);

-- Corpus statistics for BM25 (document frequency per lexeme, corpus size, avg length).
-- Refreshed after each ingestion batch (REFRESH MATERIALIZED VIEW).
CREATE MATERIALIZED VIEW IF NOT EXISTS lexeme_stats AS
    SELECT word AS lexeme, ndoc AS df FROM ts_stat('SELECT tsv FROM chunks');
CREATE UNIQUE INDEX IF NOT EXISTS lexeme_stats_lexeme_idx ON lexeme_stats (lexeme);

CREATE MATERIALIZED VIEW IF NOT EXISTS corpus_stats AS
    SELECT count(*)::float8 AS n_chunks, COALESCE(avg(length(tsv)), 1)::float8 AS avgdl FROM chunks;
