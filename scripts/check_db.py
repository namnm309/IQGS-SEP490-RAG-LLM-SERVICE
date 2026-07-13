"""One-off diagnostic: Azure Postgres + pgvector + RAG tables."""
from __future__ import annotations

import sys

import psycopg

CONN = {
    "host": "iqgs-postgres.postgres.database.azure.com",
    "port": 5432,
    "dbname": "postgres",
    "user": "iqgsadmin",
    "password": "Postgre@123",
    "sslmode": "require",
}


def main() -> int:
    try:
        with psycopg.connect(**CONN, connect_timeout=15) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT version()")
                print("CONNECTION: OK")
                print(f"  version: {cur.fetchone()[0][:100]}...")

                cur.execute(
                    "SELECT extname, extversion FROM pg_extension WHERE extname = 'vector'"
                )
                row = cur.fetchone()
                if row:
                    print(f"PGVECTOR: ENABLED ({row[0]} {row[1]})")
                else:
                    print("PGVECTOR: MISSING — can CREATE EXTENSION vector")

                cur.execute(
                    """
                    SELECT table_name FROM information_schema.tables
                    WHERE table_schema = 'public' AND table_type = 'BASE TABLE'
                    ORDER BY table_name
                    """
                )
                tables = [r[0] for r in cur.fetchall()]
                print(f"TABLES ({len(tables)}): {', '.join(tables)}")

                for tbl in ("knowledge_chunks", "knowledge_documents"):
                    cur.execute(
                        """
                        SELECT EXISTS (
                            SELECT 1 FROM information_schema.tables
                            WHERE table_schema='public' AND table_name=%s
                        )
                        """,
                        (tbl,),
                    )
                    exists = cur.fetchone()[0]
                    print(f"{tbl.upper()}: {'EXISTS' if exists else 'MISSING'}")

                cur.execute(
                    "SELECT setting FROM pg_settings WHERE name = 'azure.extensions'"
                )
                az = cur.fetchone()
                if az:
                    print(f"AZURE.EXTENSIONS: {az[0] or '(empty)'}")

        return 0
    except Exception as exc:
        print(f"CONNECTION: FAILED")
        print(f"  error: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
