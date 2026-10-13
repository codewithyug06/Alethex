import os
from alethex.pipeline import run_pipeline
from alethex.data.synthetic_benchmark.generate import generate_benchmark

def test_end_to_end_pipeline(tmp_path):
    output_dir = tmp_path / "synthetic_benchmark"
    generate_benchmark(num_entities=3, output_dir=str(output_dir))
    
    corpus_file = output_dir / "corpus.jsonl"
    assert corpus_file.exists()
    
    graph, ci = run_pipeline(str(corpus_file), dummy_ml=True)
    
    assert graph is not None
    assert len(graph.entities) > 0
    assert 0.0 <= ci <= 1.0