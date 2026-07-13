"""Stress sync generate-plan — log non-JSON / error responses."""
from __future__ import annotations

import os
import time
from pathlib import Path

import httpx

env_path = Path(__file__).resolve().parents[1] / ".env"
for line in env_path.read_text(encoding="utf-8").splitlines():
    line = line.strip()
    if line and not line.startswith("#") and "=" in line:
        k, _, v = line.partition("=")
        os.environ.setdefault(k.strip(), v.strip())

RAG_BASE = os.environ.get("RAG_BASE_URL", "https://iqgsrag.cloud")
KEY = os.environ.get("INTERNAL_API_KEY", "internal-secret")
headers = {"X-Internal-Api-Key": KEY}
body = {
    "ownerId": "964239c5-3838-411d-9724-d95883145332",
    "jobDescription": "Backend Developer C# ASP.NET Core",
    "numberOfQuestions": 3,
    "difficulty": "medium",
    "questionTypes": ["technical"],
    "skills": ["C#"],
}

for i in range(10):
    t0 = time.time()
    try:
        r = httpx.post(
            f"{RAG_BASE}/internal/rag/generate-plan",
            json=body,
            headers=headers,
            timeout=180,
        )
        elapsed = time.time() - t0
        text = r.text[:200]
        if r.headers.get("content-type", "").startswith("application/json"):
            data = r.json()
            status = "OK" if data.get("success") else f"FAIL {data.get('stage')} {data.get('error')}"
        else:
            status = f"NON-JSON HTTP {r.status_code}: {text}"
        print(f"#{i+1} {status} ({elapsed:.1f}s)")
    except Exception as e:
        print(f"#{i+1} EXC {e} ({time.time()-t0:.1f}s)")
