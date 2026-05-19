"""RAG retrieval sanity test.

Requires the ChromaDB collection to exist (run scripts/ingest_sops.py). When
it does not, the test skips instead of failing so CI without a populated store
stays green; Phase 9 governs full CI mocking strategy.
"""

import pytest

from src.rag import RAGError, retrieve_sops


def test_refund_query_retrieves_refund_sop_in_top_3():
    try:
        results = retrieve_sops("I want my money back for my order", k=3)
    except RAGError as exc:
        pytest.skip(f"SOP store unavailable: {exc}")

    filenames = [chunk.filename for chunk in results]
    assert "refund_policy.md" in filenames
