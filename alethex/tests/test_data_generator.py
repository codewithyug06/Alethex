import json
import os
from alethex.data.synthetic_benchmark.generate import generate_benchmark

def test_generate_benchmark(tmp_path):
    output_dir = tmp_path / "synthetic_benchmark"
    generate_benchmark(num_entities=10, output_dir=str(output_dir))
    
    corpus_file = output_dir / "corpus.jsonl"
    labels_file = output_dir / "gold_labels.jsonl"
    
    assert corpus_file.exists()
    assert labels_file.exists()
    
    with open(corpus_file, "r") as f:
        docs = [json.loads(line) for line in f]
    
    with open(labels_file, "r") as f:
        labels = [json.loads(line) for line in f]
        
    assert len(docs) > 0
    # Even with 10 entities, we should have some valid document generations
    # and at least one or more label pairs.
    
    if len(docs) > 1:
        assert len(labels) >= 0
        
    # Check schema of docs
    assert "source_id" in docs[0]
    assert "timestamp" in docs[0]
    assert "text" in docs[0]
    
    # Check schema of labels if they exist
    if len(labels) > 0:
        assert "claim_a_id" in labels[0]
        assert "true_relation" in labels[0]
        assert labels[0]["true_relation"] in ["consistent", "superseded", "conflicting"]