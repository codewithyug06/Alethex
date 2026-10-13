from datetime import datetime
import pytest
from pydantic import ValidationError
from alethex.ingestion.schema import Claim, RawDocument
from alethex.extraction.openie_extractor import OpenIEExtractor
from alethex.extraction.llm_extractor import LLMExtractor
from alethex.extraction.claim_merger import ClaimMerger
import json
from unittest.mock import patch, MagicMock

def test_schema():
    """Test that Pydantic models validate correctly."""
    claim = Claim(
        claim_id="c1",
        source_id="s1",
        timestamp=datetime.now(),
        subject="user",
        predicate="works at",
        object="Google",
        confidence=0.9,
        raw_sentence="user works at Google"
    )
    assert claim.object_ == "Google"
    
    with pytest.raises(ValidationError):
        Claim(
            claim_id="c2",
            source_id="s1",
            timestamp=datetime.now(),
            subject="user",
            predicate="works at",
            object="Google",
            confidence=1.5,
            raw_sentence="user works at Google"
        )

def test_raw_document():
    doc = RawDocument(
        source_id="doc1",
        timestamp=datetime.now(),
        text="Hello world"
    )
    assert doc.text == "Hello world"


def test_openie_extractor():
    extractor = OpenIEExtractor()
    doc = RawDocument(
        source_id="doc1",
        timestamp=datetime.now(),
        text="Apple acquired Beats. John loves pizza. The user prefers dark mode."
    )
    claims = extractor.extract(doc)
    
    # Check "Apple acquired Beats"
    c1 = next((c for c in claims if c.subject == "apple" and c.predicate == "acquire" and c.object_ in ("beat", "beats")), None)
    assert c1 is not None

def test_llm_extractor(tmp_path):
    log_file = tmp_path / "logs" / "llm_calls.jsonl"
    extractor = LLMExtractor(api_key="fake", log_file=str(log_file))
    
    doc = RawDocument(
        source_id="doc1",
        timestamp=datetime.now(),
        text=""
    )
    
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = json.dumps({
        "claims": [
            {"subject": "user", "predicate": "use", "object": "postgres", "confidence": 0.9}
        ]
    })
    
    with patch.object(extractor.client.chat.completions, 'create', return_value=mock_response):
        claims = extractor.extract(doc, ["the user uses postgres tbh"])
        assert len(claims) == 1
        assert claims[0].subject == "user"
        assert claims[0].predicate == "use"
        assert claims[0].object_ == "postgres"
        assert claims[0].confidence == 0.9
        
    assert log_file.exists()
    with open(log_file, "r") as f:
        log_entry = json.loads(f.readline())
        assert "latency" in log_entry
        assert log_entry["model"] == "gpt-4o"

def test_claim_merger():
    merger = ClaimMerger(min_confidence=0.4)
    dt = datetime.now()
    
    claims = [
        Claim(claim_id="c1", source_id="s1", timestamp=dt, subject="a", predicate="b", object_="c", confidence=0.3, raw_sentence=""),
        Claim(claim_id="c2", source_id="s1", timestamp=dt, subject="a", predicate="b", object_="c", confidence=0.8, raw_sentence=""),
        Claim(claim_id="c3", source_id="s1", timestamp=dt, subject="a", predicate="b", object_="c", confidence=0.9, raw_sentence=""),
        Claim(claim_id="c4", source_id="s2", timestamp=dt, subject="a", predicate="b", object_="c", confidence=0.9, raw_sentence="")
    ]
    
    merged = merger.merge(claims)
    assert len(merged) == 2
    
    s1_claim = next(c for c in merged if c.source_id == "s1")
    assert s1_claim.confidence == 0.9