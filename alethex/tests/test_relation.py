import json
from datetime import datetime
from unittest.mock import patch, MagicMock
from alethex.ingestion.schema import Claim
from alethex.relation.pair_generator import PairGenerator
from alethex.relation.nli_classifier import NLIClassifier
from alethex.relation.llm_judge import LLMJudge

def test_pair_generator():
    generator = PairGenerator(time_window_seconds=60)
    dt1 = datetime(2026, 1, 1, 12, 0, 0)
    dt2 = datetime(2026, 1, 1, 12, 0, 30) # Within 60 seconds
    dt3 = datetime(2026, 1, 1, 12, 5, 0) # Outside 60 seconds
    
    claims = [
        Claim(claim_id="1", source_id="s1", timestamp=dt1, subject="user", predicate="is", object_="a", confidence=1.0, raw_sentence=""),
        Claim(claim_id="2", source_id="s1", timestamp=dt2, subject="user", predicate="is", object_="b", confidence=1.0, raw_sentence=""),
        Claim(claim_id="3", source_id="s2", timestamp=dt3, subject="user", predicate="is", object_="c", confidence=1.0, raw_sentence="")
    ]
    
    pairs = list(generator.generate(claims))
    assert len(pairs) == 2

def test_nli_classifier_logic():
    # Use dummy to avoid downloading the real model
    classifier = NLIClassifier(dummy=True)
    
    def fake_classifier(*args, **kwargs):
        return [{"label": "ENTAILMENT", "score": 0.9}]
    
    classifier.classifier = fake_classifier
    
    dt1 = datetime(2026, 1, 1)
    dt2 = datetime(2026, 1, 2)
    c1 = Claim(claim_id="1", source_id="s1", timestamp=dt1, subject="user", predicate="like", object_="apple", confidence=1.0, raw_sentence="", validity_start=dt1)
    c2 = Claim(claim_id="2", source_id="s2", timestamp=dt2, subject="user", predicate="like", object_="apple", confidence=1.0, raw_sentence="", validity_start=dt2)
    
    res = classifier.classify_pair(c1, c2)
    assert res.relation == "consistent"
    
    def fake_contradiction(*args, **kwargs):
        return [{"label": "CONTRADICTION", "score": 0.9}]
    
    classifier.classifier = fake_contradiction
    
    # dt2 is after dt1, and we explicitly set validity_end on c1 to not overlap to trigger superseded
    c1.validity_end = dt1
    res2 = classifier.classify_pair(c1, c2)
    assert res2.relation == "superseded"
    
    # same time, overlapping
    c3 = Claim(claim_id="3", source_id="s3", timestamp=dt1, subject="user", predicate="like", object_="banana", confidence=1.0, raw_sentence="", validity_start=dt1)
    c1.validity_end = None
    res3 = classifier.classify_pair(c1, c3)
    assert res3.relation == "conflicting"

def test_llm_judge(tmp_path):
    log_file = tmp_path / "judge.jsonl"
    judge = LLMJudge(api_key="fake", log_file=str(log_file))
    
    dt = datetime(2026, 1, 1)
    c1 = Claim(claim_id="1", source_id="s1", timestamp=dt, subject="u", predicate="p", object_="o1", confidence=1.0, raw_sentence="")
    c2 = Claim(claim_id="2", source_id="s2", timestamp=dt, subject="u", predicate="p", object_="o2", confidence=1.0, raw_sentence="")
    
    classifier = NLIClassifier(dummy=True)
    classifier.classifier = None # trigger dummy return
    pair = classifier.classify_pair(c1, c2)
    
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = json.dumps({
        "relation": "conflicting",
        "confidence": 0.9,
        "justification": "They disagree"
    })
    
    with patch.object(judge.client.chat.completions, 'create', return_value=mock_response):
        res = judge.adjudicate(pair)
        assert res.relation == "conflicting"
        assert res.confidence == 0.9
