from datetime import datetime
from uuid import uuid4

from alethex.config import Config
from alethex.context_memory.compressor import (
    _dedupe_keep_latest,
    _salience_score,
    compress_session,
)
from alethex.context_memory.injector import inject
from alethex.context_memory.retriever import RetrievalResult, retrieve
from alethex.graph.belief_graph import BeliefGraph
from alethex.ingestion.schema import Claim, ClaimPair, EntityNode


def test_salience_score_boosts_high_salience_predicate_and_user_subject():
    dt = datetime(2024, 1, 1)
    c1 = Claim(claim_id="1", source_id="s1", timestamp=dt, subject="Alice", predicate="lives in", object_="NY", confidence=0.8, raw_sentence="")
    c2 = Claim(claim_id="2", source_id="s1", timestamp=dt, subject="user", predicate="employer", object_="Google", confidence=0.8, raw_sentence="")
    s1 = _salience_score(c1, ["employer"])
    s2 = _salience_score(c2, ["employer"])
    assert s2 > s1


def test_dedupe_keep_latest_keeps_most_recent_duplicate():
    t0 = datetime(2024, 1, 1)
    t1 = datetime(2024, 6, 1)
    c1 = Claim(claim_id="1", source_id="s1", timestamp=t0, subject="User", predicate="lives in", object_="Paris", confidence=0.8, raw_sentence="")
    c2 = Claim(claim_id="2", source_id="s2", timestamp=t1, subject="User", predicate="lives in", object_="Paris", confidence=0.9, raw_sentence="")
    deduped = _dedupe_keep_latest([c1, c2])
    assert len(deduped) == 1
    assert deduped[0].claim_id == "2"


def test_compress_session_returns_bounded_claims_and_summary(tmp_path):
    t0 = datetime(2024, 1, 1)
    text = "Alice lives in London. She works at DeepMind. She loves coffee."
    res = compress_session(text, doc_timestamp=t0, max_claims=2)
    assert len(res.claims) <= 2
    assert "Session on 2024-01-01" in res.narrative_summary


def test_compress_session_empty_text_returns_empty_result():
    t0 = datetime(2024, 1, 1)
    res = compress_session("", doc_timestamp=t0)
    assert len(res.claims) == 0
    assert "no structured claims" in res.narrative_summary


def test_retrieve_filters_superseded_and_conflicting_claims():
    graph = BeliefGraph()
    dt0 = datetime(2024, 1, 1)
    dt1 = datetime(2024, 6, 1)
    node = EntityNode(entity_id="ent_alice", canonical_name="Alice", aliases=["Alice"])
    
    c_old = Claim(claim_id="1", source_id="s1", timestamp=dt0, subject="Alice", predicate="lives in", object_="NY", confidence=1.0, raw_sentence="", validity_start=dt0, validity_end=dt1)
    c_new = Claim(claim_id="2", source_id="s2", timestamp=dt1, subject="Alice", predicate="lives in", object_="London", confidence=1.0, raw_sentence="", validity_start=dt1)
    c_conf = Claim(claim_id="3", source_id="s3", timestamp=dt1, subject="Alice", predicate="lives in", object_="Paris", confidence=1.0, raw_sentence="", validity_start=dt1)

    pair_super = ClaimPair(claim_a=c_old, claim_b=c_new, relation="superseded", confidence=0.95, justification="")
    pair_conf = ClaimPair(claim_a=c_new, claim_b=c_conf, relation="conflicting", confidence=0.9, justification="")

    graph.build([node], [c_old, c_new, c_conf], [pair_super, pair_conf])
    res = retrieve("Where does Alice live?", entity_id="ent_alice", belief_graph=graph, top_k=5)
    
    assert len(res.filtered_superseded) >= 1
    assert any(c.claim_id == "1" for c in res.filtered_superseded)
    assert len(res.filtered_contradictions) >= 1


def test_retrieve_unknown_entity_returns_empty_result():
    graph = BeliefGraph()
    res = retrieve("query", entity_id="unknown_ent", belief_graph=graph)
    assert len(res.claims) == 0
    assert len(res.filtered_contradictions) == 0


def test_retrieve_respects_top_k():
    graph = BeliefGraph()
    dt = datetime(2024, 1, 1)
    node = EntityNode(entity_id="ent_user", canonical_name="user", aliases=["user"])
    claims = [
        Claim(claim_id=str(i), source_id="s", timestamp=dt, subject="user", predicate=f"skill_{i}", object_="val", confidence=1.0, raw_sentence="", validity_start=dt)
        for i in range(10)
    ]
    graph.build([node], claims, [])
    res = retrieve("skills", entity_id="ent_user", belief_graph=graph, top_k=3)
    assert len(res.claims) == 3


def test_inject_formats_claims_with_timestamps():
    dt = datetime(2024, 1, 15)
    c = Claim(claim_id="1", source_id="s", timestamp=dt, subject="User", predicate="lives in", object_="Tokyo", confidence=1.0, raw_sentence="")
    res = RetrievalResult(claims=[c])
    formatted = inject(res, narrative_summary="User profile update.")
    assert "Summary: User profile update." in formatted
    assert "[2024-01-15] User lives in Tokyo" in formatted


def test_inject_includes_conflict_warning_when_contradictions_present():
    dt = datetime(2024, 1, 1)
    c1 = Claim(claim_id="1", source_id="s1", timestamp=dt, subject="User", predicate="is", object_="veg", confidence=1.0, raw_sentence="")
    c2 = Claim(claim_id="2", source_id="s2", timestamp=dt, subject="User", predicate="is", object_="nonveg", confidence=1.0, raw_sentence="")
    res = RetrievalResult(claims=[c2], filtered_contradictions=[(c1, c2)])
    formatted = inject(res)
    assert "Conflicting claims were detected" in formatted


def test_inject_truncates_to_token_budget():
    dt = datetime(2024, 1, 1)
    claims = [
        Claim(claim_id=str(i), source_id="s", timestamp=dt, subject="User", predicate=f"attribute_{i}", object_="value " * 20, confidence=1.0, raw_sentence="")
        for i in range(50)
    ]
    res = RetrievalResult(claims=claims)
    formatted = inject(res, max_tokens=10)
    assert len(formatted) <= 10 * 4 + 100


def test_inject_empty_claims_reports_no_relevant_claims():
    res = RetrievalResult(claims=[])
    formatted = inject(res)
    assert "No relevant claims found." in formatted
