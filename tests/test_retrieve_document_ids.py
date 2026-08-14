"""SCRUM-388: document_ids filter + query_extra trên RagRetrievalService."""
from __future__ import annotations

from unittest.mock import MagicMock

from services.rag_retrieval_service import RagRetrievalService
from vectorstores.base import RetrievedChunk


def test_retrieve_for_job_passes_document_ids_and_query_extra():
    store = MagicMock()
    embedding = MagicMock()
    embedding.embed_query.return_value = [0.1, 0.2, 0.3]
    settings = MagicMock()
    settings.top_k_system = 3
    settings.top_k_hr = 5

    sys_chunk = RetrievedChunk(
        document_id="sys-1",
        chunk_index=0,
        content="sys",
        scope="SYSTEM",
        owner_id=None,
        score=0.9,
        metadata={},
    )
    hr_chunk = RetrievedChunk(
        document_id="hr-doc-1",
        chunk_index=0,
        content="git rebase",
        scope="HR",
        owner_id="owner-1",
        score=0.8,
        metadata={"fileName": "git.pdf"},
    )
    store.similarity_search.side_effect = [[sys_chunk], [hr_chunk]]

    svc = RagRetrievalService(store, embedding, settings)
    system, hr = svc.retrieve_for_job(
        "Backend engineer JD",
        "owner-1",
        document_ids=["hr-doc-1", "hr-doc-2"],
        query_extra="FOCUS_HINTS: git\nINSTRUCTION: 30 câu kỹ thuật Git",
    )

    assert system == [sys_chunk]
    assert hr == [hr_chunk]
    # Embedding dùng JD + instruction
    called_query = embedding.embed_query.call_args[0][0]
    assert "Backend engineer JD" in called_query
    assert "git" in called_query.lower()

    # SYSTEM không filter doc; HR có document_ids
    sys_call = store.similarity_search.call_args_list[0]
    hr_call = store.similarity_search.call_args_list[1]
    assert sys_call.kwargs.get("document_ids") is None or sys_call[1].get("document_ids") is None
    # positional/kwargs flexible
    assert hr_call.kwargs.get("document_ids") == ["hr-doc-1", "hr-doc-2"] or (
        len(hr_call.args) >= 5 and hr_call.args[4] == ["hr-doc-1", "hr-doc-2"]
    ) or hr_call.kwargs.get("document_ids") == ["hr-doc-1", "hr-doc-2"]


def test_generate_plan_request_accepts_document_ids():
    from models.internal_schemas import GeneratePlanRequest

    req = GeneratePlanRequest.model_validate(
        {
            "ownerId": "o1",
            "jobDescription": "JD text here enough",
            "numberOfQuestions": 10,
            "difficulty": "medium",
            "questionTypes": ["technical"],
            "documentIds": ["aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"],
            "language": "English",
        }
    )
    assert req.document_ids == ["aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"]
    assert req.language == "English"
