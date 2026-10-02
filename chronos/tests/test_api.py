"""
Tests for public ALETHEX API (check, filter_context, ConsistencyEngine).
"""

from datetime import datetime

from alethex import ConsistencyEngine, check, filter_context


def test_consistency_engine_empty():
    engine = ConsistencyEngine(device="cpu")
    res = engine.check_documents([])
    assert res["corpus_ci"] == 1.0
    assert res["total_claims"] == 0
    assert len(res["contradictions"]) == 0


def test_api_check_documents():
    t0 = datetime(2024, 1, 1, 10, 0, 0)
    t1 = datetime(2024, 1, 1, 11, 0, 0)
    
    docs = [
        {"source_id": "doc1", "timestamp": t0.isoformat(), "text": "Alice works at Google in London."},
        {"source_id": "doc2", "timestamp": t1.isoformat(), "text": "Alice likes Python programming."}
    ]
    
    res = check(docs, device="cpu")
    assert "corpus_ci" in res
    assert 0.0 <= res["corpus_ci"] <= 1.0
    assert res["total_claims"] >= 1


def test_api_filter_context():
    t0 = datetime(2024, 1, 1, 10, 0, 0)
    t1 = datetime(2024, 6, 1, 10, 0, 0)
    
    entries = [
        {"id": "mem_1", "timestamp": t0.isoformat(), "text": "Alice lives in Paris."},
        {"id": "mem_2", "timestamp": t1.isoformat(), "text": "Alice moved to Tokyo."},
    ]
    
    filtered = filter_context(entries, device="cpu")
    assert len(filtered) == 2
    for item in filtered:
        assert "source_id" in item
        assert "status" in item
        assert "is_valid" in item

