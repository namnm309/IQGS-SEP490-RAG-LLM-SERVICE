"""Diagnostic: tbl_knowledge_chunks + retrieval conditions vs RAG similarity SQL."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import httpx
import psycopg
from psycopg.rows import dict_row

# Load RAG .env
env_path = Path(__file__).resolve().parents[1] / ".env"
if env_path.exists():
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, _, v = line.partition("=")
            os.environ.setdefault(k.strip(), v.strip())

DATABASE_URL = os.environ.get("DATABASE_URL", "")
INTERNAL_API_KEY = os.environ.get("INTERNAL_API_KEY", "internal-secret")
RAG_BASE = os.environ.get("RAG_BASE_URL", "https://iqgsrag.cloud")
BACKEND_BASE = os.environ.get("BACKEND_INTERNAL_BASE_URL", "https://api.hiregen.io.vn")

JOB_ID = "5c8b04eb-09d9-4567-a199-a10bc472ad65"
OWNER_ID = "964239c5-3838-411d-9724-d95883145332"
JD_SNIPPET = "Backend Developer ASP.NET Core PostgreSQL JWT Entity Framework Core"


def section(title: str) -> None:
    print(f"\n{'=' * 60}\n{title}\n{'=' * 60}")


def check_db() -> int:
    section("1. DATABASE — tbl_knowledge_chunks")
    issues = 0
    with psycopg.connect(DATABASE_URL, connect_timeout=15) as conn:
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute("SELECT COUNT(*) AS c FROM tbl_knowledge_chunks")
            total = cur.fetchone()["c"]
            print(f"Total chunks: {total}")

            cur.execute(
                """
                SELECT scope, owner_id IS NULL AS owner_null, COUNT(*) AS c
                FROM tbl_knowledge_chunks GROUP BY scope, owner_id IS NULL ORDER BY scope
                """
            )
            print("By scope / owner_null:", [dict(r) for r in cur.fetchall()])

            # Exact SQL RAG uses for SYSTEM retrieve
            cur.execute(
                """
                SELECT COUNT(*) AS c FROM tbl_knowledge_chunks
                WHERE scope = 'SYSTEM' AND owner_id IS NULL
                """
            )
            system_retrievable = cur.fetchone()["c"]
            print(f"SYSTEM retrievable (scope=SYSTEM AND owner_id IS NULL): {system_retrievable}")
            if system_retrievable == 0:
                issues += 1
                print("  *** ISSUE: RAG SYSTEM query would return 0 rows!")

            cur.execute(
                """
                SELECT COUNT(*) AS c FROM tbl_knowledge_chunks
                WHERE scope = 'SYSTEM' AND owner_id IS NOT NULL
                """
            )
            bad_system = cur.fetchone()["c"]
            if bad_system:
                issues += 1
                print(f"  *** ISSUE: {bad_system} SYSTEM chunks have owner_id set (invisible to RAG)")

            cur.execute(
                """
                SELECT COUNT(*) AS c FROM tbl_knowledge_chunks
                WHERE embedding IS NULL
                """
            )
            null_emb = cur.fetchone()["c"]
            print(f"Chunks with NULL embedding: {null_emb}")
            if null_emb:
                issues += 1
                print("  *** ISSUE: NULL embeddings may break vector search")

            cur.execute(
                """
                SELECT vector_dims(embedding) AS dims, COUNT(*) AS c
                FROM tbl_knowledge_chunks
                WHERE embedding IS NOT NULL
                GROUP BY vector_dims(embedding)
                """
            )
            print("Embedding dimensions:", [dict(r) for r in cur.fetchall()])

            cur.execute(
                """
                SELECT id, scope, status, owner_id, file_name
                FROM tbl_knowledge_documents ORDER BY created_at
                """
            )
            print("Documents:")
            for row in cur.fetchall():
                print(f"  {row}")

            cur.execute(
                """
                SELECT "Id", "OwnerId", "Status", "CreatedAt",
                       LEFT("ErrorMessage", 120) AS err
                FROM question_generation_jobs
                WHERE "Status" = 'FAILED' AND "ErrorMessage" LIKE '%RETRIEVAL_EMPTY%'
                ORDER BY "CreatedAt" DESC LIMIT 5
                """
            )
            print("Recent RETRIEVAL_EMPTY jobs:")
            for row in cur.fetchall():
                print(f"  {row}")

    return issues


def check_rag_health() -> None:
    section("2. RAG /health")
    try:
        r = httpx.get(f"{RAG_BASE}/health", timeout=10)
        print(f"HTTP {r.status_code}: {r.text}")
    except Exception as e:
        print(f"FAILED: {e}")


def check_rag_generate_sync() -> None:
    section("3. RAG POST /internal/rag/generate-plan (sync)")
    body = {
        "ownerId": OWNER_ID,
        "jobDescription": JD_SNIPPET,
        "numberOfQuestions": 5,
        "difficulty": "medium",
        "questionTypes": ["technical", "behavioral"],
        "skills": ["C#", "ASP.NET Core"],
    }
    headers = {"X-Internal-Api-Key": INTERNAL_API_KEY}
    try:
        r = httpx.post(
            f"{RAG_BASE}/internal/rag/generate-plan",
            json=body,
            headers=headers,
            timeout=180,
        )
        print(f"HTTP {r.status_code}")
        data = r.json()
        print(f"success={data.get('success')} stage={data.get('stage')} error={data.get('error')}")
        if data.get("success"):
            plan = data.get("plan") or {}
            print(f"roleTitle={plan.get('roleTitle', plan.get('role_title', 'N/A'))[:80]}")
    except Exception as e:
        print(f"FAILED: {e}")


def check_rag_generate_async() -> None:
    section("4. RAG POST /internal/rag/generate-plan/async")
    body = {
        "jobId": JOB_ID,
        "ownerId": OWNER_ID,
        "jobDescription": JD_SNIPPET,
        "numberOfQuestions": 5,
        "difficulty": "medium",
        "questionTypes": ["technical"],
        "skills": ["C#"],
    }
    headers = {"X-Internal-Api-Key": INTERNAL_API_KEY}
    try:
        r = httpx.post(
            f"{RAG_BASE}/internal/rag/generate-plan/async",
            json=body,
            headers=headers,
            timeout=30,
        )
        print(f"HTTP {r.status_code}: {r.text[:500]}")
    except Exception as e:
        print(f"FAILED: {e}")


def check_backend_internal_key() -> None:
    section("5. Backend internal callback reachability")
    headers = {"X-Internal-Api-Key": INTERNAL_API_KEY}
    url = f"{BACKEND_BASE}/internal/question-generation-jobs/{JOB_ID}/generation-result"
    # Dry-run OPTIONS or invalid PATCH to see if route exists
    try:
        r = httpx.patch(url, json={"phase": "PLAN", "success": False, "error": "diag"}, headers=headers, timeout=15)
        print(f"PATCH callback probe HTTP {r.status_code}: {r.text[:300]}")
    except Exception as e:
        print(f"FAILED: {e}")


def main() -> int:
    if not DATABASE_URL:
        print("DATABASE_URL not set")
        return 1
    issues = check_db()
    check_rag_health()
    check_rag_generate_sync()
    check_rag_generate_async()
    check_backend_internal_key()
    section("SUMMARY")
    print(f"DB issues found: {issues}")
    return issues


if __name__ == "__main__":
    sys.exit(main())
