"""SCRUM-447: retrieve_system_only accepts document_ids filter."""
from unittest.mock import MagicMock


def test_retrieve_system_only_passes_document_ids():
    store = MagicMock()
    store.similarity_search.return_value = []
    embedding = MagicMock()
    embedding.embed_query.return_value = [0.1, 0.2]

    settings = MagicMock()
    settings.top_k_system = 5

    from services.rag_retrieval_service import RagRetrievalService

    svc = RagRetrievalService(vector_store=store, embedding=embedding, settings=settings)
    svc.retrieve_system_only(
        "aspnet core middleware",
        document_ids=["doc-tech-1", "doc-tech-2"],
        query_extra="junior",
    )

    kwargs = store.similarity_search.call_args.kwargs
    assert kwargs.get("scope") == "SYSTEM"
    assert kwargs.get("document_ids") == ["doc-tech-1", "doc-tech-2"]
