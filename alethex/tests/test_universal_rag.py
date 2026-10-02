"""
Tests for Universal RAG Reconciliation Layer.
Verifies compatibility with raw strings, dicts, and multi-chunk temporal updates.
"""

from datetime import datetime
from alethex.integrations.universal_rag import UniversalRAG


def test_universal_rag_filter_drop():
    rag = UniversalRAG(mode="drop", device="cpu")
    chunks = [
        {"id": "c1", "text": "Alice is in London in Jan 2024.", "timestamp": "2024-01-01T00:00:00"},
        {"id": "c2", "text": "Alice likes reading.", "timestamp": "2024-02-01T00:00:00"},
        {"id": "c3", "text": "Alice moved to Tokyo in June 2024.", "timestamp": "2024-06-01T00:00:00"}
    ]
    surviving = rag.filter(chunks, mode="drop")
    assert isinstance(surviving, list)
    assert len(surviving) >= 2


def test_universal_rag_reconciled_context():
    rag = UniversalRAG(mode="drop", device="cpu")
    chunks = [
        "Alice is employed as a software engineer at Meta in London.",
        "Alice enjoys hiking on weekends."
    ]
    ctx = rag.get_reconciled_context(chunks)
    assert isinstance(ctx, str)
    assert len(ctx) > 0


def test_universal_rag_wrap_retriever():
    rag = UniversalRAG(mode="drop", device="cpu")

    def mock_vector_db_search(query: str):
        return [
            {"id": "v1", "text": f"Found fact 1 for {query}", "timestamp": datetime.now().isoformat()},
            {"id": "v2", "text": f"Found fact 2 for {query}", "timestamp": datetime.now().isoformat()},
        ]

    reconciled_search = rag.wrap(mock_vector_db_search)
    results = reconciled_search("Alice")
    assert len(results) == 2
    assert "Found fact 1 for Alice" in results[0]["text"]
