from datetime import datetime
from alethex.ingestion.schema import Claim, EntityNode, ClaimPair
from alethex.graph.belief_graph import BeliefGraph
from alethex.graph import query

def test_belief_graph():
    graph = BeliefGraph()
    dt1 = datetime(2026, 1, 1)
    
    node = EntityNode(entity_id="ent1", canonical_name="user", aliases=["user"])
    
    c1 = Claim(claim_id="1", source_id="s1", timestamp=dt1, subject="user", predicate="live", object_="NY", confidence=1.0, raw_sentence="", validity_start=dt1)
    c2 = Claim(claim_id="2", source_id="s2", timestamp=dt1, subject="user", predicate="live", object_="CA", confidence=1.0, raw_sentence="", validity_start=dt1)
    c3 = Claim(claim_id="3", source_id="s3", timestamp=dt1, subject="user", predicate="work", object_="Google", confidence=1.0, raw_sentence="", validity_start=dt1, validity_end=datetime(2026, 6, 1))
    
    pair = ClaimPair(claim_a=c1, claim_b=c2, relation="conflicting", confidence=0.9, justification="")
    
    graph.build([node], [c1, c2, c3], [pair])
    
    # Test valid claims (c1 and c2 are valid, c3 is valid since validity_end > now, assuming now > 2026-06-01?)
    # Actually now is likely > 2026-06-01 based on current date, so c3 might be expired.
    # Let's check explicitly
    valid = graph.get_current_valid_claims("ent1")
    # c1 and c2 have no validity_end, so they are valid.
    assert c1 in valid
    assert c2 in valid
    
    # Test queries
    conflicts = query.get_conflicts(graph, "ent1")
    assert len(conflicts) == 1
    assert (c1, c2) in conflicts or (c2, c1) in conflicts
    
    superseded = query.get_superseded(graph, "ent1")
    assert len(superseded) == 1
    assert superseded[0] == c3
    
    timeline = query.get_timeline(graph, "ent1", "live")
    assert len(timeline) == 2