"""Direct pgvector similarity_search SQL test (no Ollama)."""
import os
from pathlib import Path

import psycopg
from pgvector.psycopg import register_vector
from psycopg.rows import dict_row

env_path = Path(__file__).resolve().parents[1] / ".env"
for line in env_path.read_text(encoding="utf-8").splitlines():
    line = line.strip()
    if line and not line.startswith("#") and "=" in line:
        k, _, v = line.partition("=")
        os.environ.setdefault(k.strip(), v.strip())

DATABASE_URL = os.environ["DATABASE_URL"]

with psycopg.connect(DATABASE_URL, connect_timeout=15) as conn:
    register_vector(conn)
    with conn.cursor(row_factory=dict_row) as cur:
        # Lấy 1 embedding có sẵn làm query vector
        cur.execute(
            "SELECT embedding FROM tbl_knowledge_chunks WHERE scope='SYSTEM' LIMIT 1"
        )
        row = cur.fetchone()
        if not row:
            print("No SYSTEM chunk to test")
            raise SystemExit(1)
        query_emb = row["embedding"]

        cur.execute(
            """
            SELECT document_id, chunk_index, scope,
                   1 - (embedding <=> %s::vector) AS score
            FROM tbl_knowledge_chunks
            WHERE scope = 'SYSTEM' AND owner_id IS NULL
            ORDER BY embedding <=> %s::vector
            LIMIT 5
            """,
            (query_emb, query_emb),
        )
        results = cur.fetchall()
        print(f"Direct SQL similarity returned {len(results)} rows")
        for r in results:
            print(f"  score={r['score']:.4f} doc={r['document_id']} idx={r['chunk_index']}")

        cur.execute("SELECT current_database(), inet_server_addr(), current_schema()")
        print("DB context:", cur.fetchone())
