-- Dev/test DDL — phải khớp Backend migration (EMBEDDING_DIMENSION=768)
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS knowledge_chunks (
    id UUID PRIMARY KEY,
    document_id UUID NOT NULL,
    owner_id UUID NULL,
    scope VARCHAR(20) NOT NULL,
    chunk_index INT NOT NULL,
    content TEXT NOT NULL,
    metadata JSONB,
    embedding vector(768) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS ix_knowledge_chunks_document_id ON knowledge_chunks (document_id);
CREATE INDEX IF NOT EXISTS ix_knowledge_chunks_scope_owner ON knowledge_chunks (scope, owner_id);

DO $$
BEGIN
    CREATE INDEX ix_knowledge_chunks_embedding_hnsw
    ON knowledge_chunks USING hnsw (embedding vector_cosine_ops);
EXCEPTION
    WHEN duplicate_table THEN NULL;
END $$;
