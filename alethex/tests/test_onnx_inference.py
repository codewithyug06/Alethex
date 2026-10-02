"""
Tests for ONNX Export, Dynamic INT8 Quantization, and ONNX Runtime Acceleration.
"""

from datetime import datetime
from pathlib import Path

import pytest

from alethex.ingestion.schema import Claim
from alethex.optim.onnx_pipeline import ONNXTextClassificationPipeline
from alethex.relation.nli_classifier import NLIClassifier


def test_onnx_pipeline_inference():
    onnx_dir = Path("models/alethex-nli-int8")
    if not (onnx_dir / "model_int8.onnx").exists():
        pytest.skip("models/alethex-nli-int8 not found")

    pipe = ONNXTextClassificationPipeline(onnx_dir)
    res = pipe({"text": "Alice lives in London.", "text_pair": "Alice moved to Tokyo."})

    assert isinstance(res, list)
    assert len(res) == 3
    # Top score should have valid probability
    top_item = res[0]
    assert "label" in top_item
    assert "score" in top_item
    assert 0.0 <= top_item["score"] <= 1.0


def test_nli_classifier_with_onnx_backend():
    onnx_dir = Path("models/alethex-nli-int8")
    if not (onnx_dir / "model_int8.onnx").exists():
        pytest.skip("models/alethex-nli-int8 not found")

    classifier = NLIClassifier(model_name=str(onnx_dir), use_onnx=True)
    assert classifier.is_onnx is True

    dt = datetime(2026, 1, 1)
    c1 = Claim(claim_id="1", source_id="s1", timestamp=dt, subject="Alice", predicate="works at", object_="Google", confidence=1.0, raw_sentence="")
    c2 = Claim(claim_id="2", source_id="s2", timestamp=dt, subject="Alice", predicate="works at", object_="Meta", confidence=1.0, raw_sentence="")

    pair = classifier.classify_pair(c1, c2)
    assert pair.relation in ("conflicting", "superseded")
    assert pair.confidence > 0.7
