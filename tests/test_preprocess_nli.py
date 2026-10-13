"""Tests for NLI dataset preprocessing and preprocessed dataset integrity."""
import json
import os
from pathlib import Path
import pytest
from alethex.data.preprocess_nli import (
    CANONICAL_LABELS,
    _MNLI_SNLI_RAW_TO_CANONICAL,
    _to_canonical,
)
from datasets import Dataset


def test_canonical_label_mapping():
    """Verify DeBERTa-v3 canonical label IDs and raw MNLI/SNLI conversion mapping."""
    assert CANONICAL_LABELS["contradiction"] == 0
    assert CANONICAL_LABELS["entailment"] == 1
    assert CANONICAL_LABELS["neutral"] == 2

    # In raw MultiNLI/SNLI: 0=entailment, 1=neutral, 2=contradiction
    assert _MNLI_SNLI_RAW_TO_CANONICAL[0] == CANONICAL_LABELS["entailment"]
    assert _MNLI_SNLI_RAW_TO_CANONICAL[1] == CANONICAL_LABELS["neutral"]
    assert _MNLI_SNLI_RAW_TO_CANONICAL[2] == CANONICAL_LABELS["contradiction"]


def test_to_canonical_transformation():
    """Verify _to_canonical maps raw dataset schema to canonical (premise, hypothesis, label)."""
    raw_data = {
        "premise": ["A dog runs.", "A cat sleeps.", "Unknown label item."],
        "hypothesis": ["An animal is running.", "A dog barks.", "Something."],
        "label": [0, 2, -1],  # -1 represents unlabelled / no gold consensus in SNLI
    }
    raw_ds = Dataset.from_dict(raw_data)
    canonical_ds = _to_canonical(
        raw_ds,
        premise_col="premise",
        hypothesis_col="hypothesis",
        label_col="label",
        label_map=_MNLI_SNLI_RAW_TO_CANONICAL,
        name="test_transform",
    )

    assert len(canonical_ds) == 2
    assert set(canonical_ds.column_names) == {"premise", "hypothesis", "label"}
    
    # 0 (raw entailment) -> 1 (canonical entailment)
    assert canonical_ds[0]["label"] == 1
    assert canonical_ds[0]["premise"] == "A dog runs."
    assert canonical_ds[0]["hypothesis"] == "An animal is running."

    # 2 (raw contradiction) -> 0 (canonical contradiction)
    assert canonical_ds[1]["label"] == 0
    assert canonical_ds[1]["premise"] == "A cat sleeps."


def test_preprocessed_dataset_integrity():
    """Verify on-disk unified preprocessed dataset files if present."""
    base_dirs = [
        Path("dataset/preprocessed/nli/unified"),
        Path("alethex/dataset/preprocessed/nli/unified"),
    ]
    target_dir = None
    for d in base_dirs:
        if (d / "validation.jsonl").exists():
            target_dir = d
            break

    if target_dir is None:
        pytest.skip("Preprocessed dataset not found locally (skipped for fresh git checkout).")

    val_path = target_dir / "validation.jsonl"
    assert val_path.exists()
    assert val_path.stat().st_size > 1_000_000  # validation set is ~7MB

    valid_labels = {0, 1, 2}
    checked = 0
    with open(val_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= 1000:
                break
            row = json.loads(line)
            assert "premise" in row and isinstance(row["premise"], str) and len(row["premise"]) > 0
            assert "hypothesis" in row and isinstance(row["hypothesis"], str) and len(row["hypothesis"]) > 0
            assert "label" in row and row["label"] in valid_labels
            checked += 1

    assert checked == 1000
