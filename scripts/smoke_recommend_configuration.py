#!/usr/bin/env python3
"""Real RAG/LLM smoke for recommend-interview-configuration (3 JD fixtures).

Usage:
  cd RAG
  python scripts/smoke_recommend_configuration.py

Requires running RAG service + valid LLM credentials in env/.env.
Outputs: docs/features/studio-phase1-smoke-outputs.json
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.settings import get_settings  # noqa: E402

JD_FIXTURES = {
    "dotnet_backend": """
Senior .NET Backend Developer

We are hiring a Senior Backend Engineer to build scalable APIs with ASP.NET Core 8,
Entity Framework Core, and PostgreSQL. Responsibilities include designing REST APIs,
optimizing database queries, implementing caching with Redis, and mentoring junior developers.
Requirements: 5+ years C#, microservices, Azure, CI/CD, unit testing with xUnit.
""".strip(),
    "react_frontend": """
React Frontend Engineer

Looking for a mid-level frontend developer proficient in React 18, TypeScript, Redux Toolkit,
and modern CSS. You will build responsive HR dashboards, integrate REST APIs, write component tests
with React Testing Library, and collaborate with UX on accessibility (WCAG).
""".strip(),
    "data_engineer": """
Data Engineer

Join our data platform team to design ETL pipelines with Apache Spark, Airflow, and Snowflake.
You will model data warehouses, ensure data quality, optimize batch/stream jobs, and support
analytics teams with reliable datasets. Python, SQL, and cloud (AWS) experience required.
""".strip(),
}


def main() -> int:
    settings = get_settings()
    base_url = os.environ.get("RAG_SMOKE_BASE_URL", "http://127.0.0.1:8000")
    api_key = settings.internal_api_key or os.environ.get("INTERNAL_API_KEY", "")
    if not api_key:
        print("Missing INTERNAL_API_KEY — set in .env or environment.")
        return 1

    headers = {"X-Internal-Api-Key": api_key, "Content-Type": "application/json"}
    owner_id = os.environ.get("RAG_SMOKE_OWNER_ID", "11111111-1111-1111-1111-111111111111")
    out_path = ROOT.parent / "docs" / "features" / "studio-phase1-smoke-outputs.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    results: dict[str, object] = {
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "ragBaseUrl": base_url,
        "fixtures": {},
    }

    with httpx.Client(base_url=base_url, headers=headers, timeout=180.0) as client:
        for key, jd in JD_FIXTURES.items():
            print(f"--- Smoke: {key} ---")
            analyze_resp = client.post(
                "/internal/rag/analyze-jd",
                json={"jobDescription": jd},
            )
            analyze_resp.raise_for_status()
            profile = analyze_resp.json()

            recommend_resp = client.post(
                "/internal/rag/recommend-interview-configuration",
                json={
                    "ownerId": owner_id,
                    "jobDescription": jd,
                    "jobProfile": {
                        "jobTitle": profile.get("jobTitle") or profile.get("position"),
                        "experienceLevel": profile.get("experienceLevel"),
                        "detectedRole": profile.get("detectedRole"),
                        "detectedLanguage": profile.get("detectedLanguage"),
                        "skills": profile.get("skills") or [],
                        "responsibilities": profile.get("responsibilities") or [],
                        "summary": profile.get("summary"),
                    },
                    "documentIds": [],
                    "numberOfQuestions": 10,
                },
            )
            recommend_resp.raise_for_status()
            rec = recommend_resp.json()
            cfg = rec.get("recommendedConfiguration") or {}
            focus_names = [f.get("name") for f in (cfg.get("focusAreas") or [])]

            results["fixtures"][key] = {
                "analyze": profile,
                "recommend": rec,
                "focusAreaNames": focus_names,
            }
            print(f"  focus areas: {focus_names}")

    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
