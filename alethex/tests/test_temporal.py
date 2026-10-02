from datetime import datetime
from alethex.ingestion.schema import Claim
from alethex.temporal.timex_tagger import TimexTagger
from alethex.temporal.interval_resolver import IntervalResolver

def test_timex_tagger():
    tagger = TimexTagger()
    ref_date = datetime(2026, 6, 1) 
    
    expressions = [
        ("I moved here last March.", 3), 
        ("I have been working since 2023.", 2023),
        ("Let's meet tomorrow.", 2),
        ("It happened yesterday.", 31),
        ("He died on 2010-05-15.", 2010),
        ("two weeks ago we did it", 5), 
    ]
    
    for text, expected_part in expressions:
        res = tagger.extract_time(text, ref_date)
        assert res is not None, f"Failed to extract date from: {text}"

def test_interval_resolver():
    resolver = IntervalResolver()
    dt1 = datetime(2026, 1, 1)
    dt2 = datetime(2026, 6, 1)
    
    claims = [
        Claim(claim_id="1", source_id="s1", timestamp=dt1, subject="user", predicate="live in", object_="NY", confidence=0.9, raw_sentence="I live in NY."),
        Claim(claim_id="2", source_id="s2", timestamp=dt2, subject="user", predicate="live in", object_="CA", confidence=0.9, raw_sentence="I live in CA."),
        Claim(claim_id="3", source_id="s3", timestamp=dt1, subject="user", predicate="work at", object_="Google", confidence=0.9, raw_sentence="I work at Google until 2026-05-01.")
    ]
    
    resolved = resolver.resolve_intervals(claims)
    
    c1 = next(c for c in resolved if c.claim_id == "1")
    c2 = next(c for c in resolved if c.claim_id == "2")
    c3 = next(c for c in resolved if c.claim_id == "3")
    
    assert c1.validity_end == c2.validity_start
    assert c2.validity_end is None
    
    assert c3.validity_end is not None
    assert c3.validity_end.year == 2026
    assert c3.validity_end.month == 5
    assert c3.validity_end.day == 1
