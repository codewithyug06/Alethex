from alethex.coreference.entity_linker import EntityLinker
from alethex.ingestion.schema import Claim
from datetime import datetime

def test_entity_linker():
    # Use eps=0.25, min_samples=2 as specified
    linker = EntityLinker(eps=0.25, min_samples=2)
    
    dt = datetime.now()
    claims = [
        Claim(claim_id="1", source_id="s1", timestamp=dt, subject="The User", predicate="is", object_="John", confidence=0.9, raw_sentence=""),
        Claim(claim_id="2", source_id="s2", timestamp=dt, subject="the user", predicate="is", object_="Jane", confidence=0.9, raw_sentence=""),
        Claim(claim_id="3", source_id="s3", timestamp=dt, subject="User", predicate="is", object_="Bob", confidence=0.9, raw_sentence="")
    ]
    
    updated_claims, nodes = linker.link(claims)
    
    # "The User" and "the user" will hard-merge to "the user".
    # "User" will normalize to "user".
    # "the user" and "user" will cluster via DBSCAN because sentence-transformers embeddings are very close.
    # Therefore, all three claims should have the same canonical subject now.
    
    assert updated_claims[0].subject == updated_claims[1].subject == updated_claims[2].subject
