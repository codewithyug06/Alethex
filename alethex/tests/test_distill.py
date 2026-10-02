"""
Tests for Knowledge Distillation components (NLIPairDataset, trainer step, synthetic generation).
"""

import torch
from transformers import AutoTokenizer

from alethex.training.distill import NLIPairDataset, generate_synthetic_nli_pairs


def test_synthetic_nli_generation():
    pairs = generate_synthetic_nli_pairs(10)
    assert len(pairs) == 10
    for p in pairs:
        assert "premise" in p
        assert "hypothesis" in p
        assert "label" in p
        assert p["label"] in ("entailment", "contradiction", "neutral")


def test_nli_pair_dataset():
    pairs = [
        {"premise": "Alice lives in NY.", "hypothesis": "Alice lives in US.", "label": "entailment"},
        {"premise": "Bob likes cats.", "hypothesis": "Bob hates cats.", "label": "contradiction"},
    ]
    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    dataset = NLIPairDataset(pairs, tokenizer, max_length=64)

    assert len(dataset) == 2
    item0 = dataset[0]
    assert "input_ids" in item0
    assert "attention_mask" in item0
    assert "label" in item0
    assert isinstance(item0["input_ids"], torch.Tensor)
    assert item0["input_ids"].shape[0] == 64
    assert item0["label"].item() == 1  # entailment -> 1
