from datetime import datetime, timedelta
from alethex.ingestion.schema import Claim, EntityNode, ClaimPair
from alethex.graph.belief_graph import BeliefGraph
from alethex.scoring import consistency_index, contradiction_report, staleness_report

def test_scoring_clean():
    graph = BeliefGraph()
    dt = datetime.now()
    node = EntityNode(entity_id="e1", canonical_name="user", aliases=[])
    c1 = Claim(claim_id="1", source_id="s1", timestamp=dt, subject="e1", predicate="p", object_="o", confidence=1.0, raw_sentence="", validity_start=dt)
    graph.build([node], [c1], [])
    
    corpus_ci, entity_cis = consistency_index.calculate_ci(graph)
    assert corpus_ci == 1.0
    assert entity_cis["e1"] == 1.0

def test_scoring_conflict():
    graph = BeliefGraph()
    dt = datetime.now()
    node = EntityNode(entity_id="e1", canonical_name="user", aliases=[])
    c1 = Claim(claim_id="1", source_id="s1", timestamp=dt, subject="e1", predicate="p", object_="o1", confidence=1.0, raw_sentence="", validity_start=dt)
    c2 = Claim(claim_id="2", source_id="s2", timestamp=dt, subject="e1", predicate="p", object_="o2", confidence=1.0, raw_sentence="", validity_start=dt)
    pair = ClaimPair(claim_a=c1, claim_b=c2, relation="conflicting", confidence=0.9, justification="")
    
    graph.build([node], [c1, c2], [pair])
    
    corpus_ci, entity_cis = consistency_index.calculate_ci(graph)
    assert corpus_ci < 1.0
    assert entity_cis["e1"] == 0.0 # 1 pair, 1 conflict -> 0.0
    
    report = contradiction_report.generate_report(graph)
    assert len(report) == 1
    assert report[0]["entity"] == "user"

def test_staleness_report():
    graph = BeliefGraph()
    dt = datetime.now()
    node = EntityNode(entity_id="e1", canonical_name="user", aliases=[])
    c1 = Claim(claim_id="1", source_id="s1", timestamp=dt, subject="e1", predicate="p", object_="o", confidence=1.0, raw_sentence="", validity_start=dt, validity_end=dt - timedelta(days=40))
    graph.build([node], [c1], [])
    
    report = staleness_report.generate_report(graph)
    assert len(report) == 1
    assert report[0]["days_stale"] >= 40