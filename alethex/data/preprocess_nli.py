"""
Unifies the raw NLI datasets already downloaded into the HF cache (via
download_datasets.py) into a single canonical 3-way schema:

    {"premise": str, "hypothesis": str, "label": int}
    label: 0 = entailment, 1 = neutral, 2 = contradiction

Each source dataset is remapped to this scheme explicitly rather than assumed,
because the raw schemas differ (see per-source loaders below) and a silent
mismatch here previously produced worse-than-chance training runs.

Writes:
  dataset/preprocessed/nli/unified/train.jsonl
  dataset/preprocessed/nli/unified/validation.jsonl
  dataset/preprocessed/stats/preprocessing_summary.md
"""
import os
from collections import Counter

from datasets import load_dataset, concatenate_datasets, Value
from rich.console import Console

console = Console()

ALETHEX_HOME = os.environ.get("ALETHEX_HOME", os.getcwd())
OUT_DIR = os.path.join(ALETHEX_HOME, "dataset", "preprocessed", "nli", "unified")
STATS_DIR = os.path.join(ALETHEX_HOME, "dataset", "preprocessed", "stats")

# Canonical scheme used everywhere downstream. MUST match the pretrained
# cross-encoder/nli-deberta-v3-large checkpoint's own id2label ordering
# (verified via AutoConfig.from_pretrained(...).id2label -- it is
# {0: contradiction, 1: entailment, 2: neutral}, NOT the MultiNLI/SNLI-native
# {0: entailment, 1: neutral, 2: contradiction} scheme). Getting this wrong
# fights the pretrained head instead of fine-tuning it and produces
# worse-than-chance accuracy -- confirmed empirically (smoke test scored
# 12% accuracy / -0.33 MCC before this fix).
CANONICAL_LABELS = {"contradiction": 0, "entailment": 1, "neutral": 2}


def _to_canonical(ds, premise_col, hypothesis_col, label_col, label_map, name):
    keep = {premise_col, hypothesis_col, label_col}
    ds = ds.remove_columns([c for c in ds.column_names if c not in keep])
    # Drop rows whose raw label isn't in our map (e.g. MNLI/SNLI use -1 for "no gold label").
    ds = ds.filter(lambda ex: ex[label_col] in label_map)
    ds = ds.map(
        lambda ex: {
            "premise": ex[premise_col],
            "hypothesis": ex[hypothesis_col],
            "label": label_map[ex[label_col]],
        },
        remove_columns=ds.column_names,
    )
    console.print(f"  [green]{name}[/green]: {len(ds)} pairs")
    return ds


def load_temporal_nli(split):
    # Raw schema: {"Premise": str, "Hypothesis": str, "Label": "entailment"|"neutral"|"contradiction"}
    ds = load_dataset("tasksource/temporal-nli", split=split)
    return _to_canonical(ds, "Premise", "Hypothesis", "Label", CANONICAL_LABELS, f"temporal_nli_{split}")


# Raw MultiNLI/SNLI ClassLabel scheme is entailment=0, neutral=1, contradiction=2 --
# different from CANONICAL_LABELS (which matches the pretrained head instead). Remap explicitly.
_MNLI_SNLI_RAW_TO_CANONICAL = {
    0: CANONICAL_LABELS["entailment"],
    1: CANONICAL_LABELS["neutral"],
    2: CANONICAL_LABELS["contradiction"],
}


def load_multi_nli(split):
    ds = load_dataset("nyu-mll/multi_nli", split=split)
    return _to_canonical(ds, "premise", "hypothesis", "label", _MNLI_SNLI_RAW_TO_CANONICAL, f"multi_nli_{split}")


def load_snli(split):
    # label=-1 means "no gold consensus" in the raw data; not in the map, so filtered out by _to_canonical.
    ds = load_dataset("stanfordnlp/snli", split=split)
    return _to_canonical(ds, "premise", "hypothesis", "label", _MNLI_SNLI_RAW_TO_CANONICAL, f"snli_{split}")


def load_all_nli_entailment(split):
    # "pair" config only contains positive (entailment) pairs: {"anchor": premise, "positive": hypothesis}.
    ds = load_dataset("sentence-transformers/all-nli", "pair", split=split)
    entailment_label = CANONICAL_LABELS["entailment"]
    ds = ds.map(
        lambda ex: {"premise": ex["anchor"], "hypothesis": ex["positive"], "label": entailment_label},
        remove_columns=ds.column_names,
    )
    console.print(f"  [green]all_nli_{split}[/green]: {len(ds)} pairs")
    return ds


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(STATS_DIR, exist_ok=True)

    console.print("[bold blue]Loading & remapping train splits...[/bold blue]")
    train_parts = [
        load_temporal_nli("train"),
        load_multi_nli("train"),
        load_snli("train"),
        load_all_nli_entailment("train"),
    ]

    console.print("[bold blue]Loading & remapping validation splits...[/bold blue]")
    val_parts = [
        load_temporal_nli("test"),
        load_multi_nli("validation_matched"),
        load_snli("validation"),
    ]

    # Normalize feature types before concatenation (label must be same dtype across parts).
    def _cast(ds):
        return ds.cast_column("label", Value("int64"))

    train_parts = [_cast(d) for d in train_parts]
    val_parts = [_cast(d) for d in val_parts]

    train_ds = concatenate_datasets(train_parts).shuffle(seed=42)
    val_ds = concatenate_datasets(val_parts).shuffle(seed=42)

    train_path = os.path.join(OUT_DIR, "train.jsonl")
    val_path = os.path.join(OUT_DIR, "validation.jsonl")
    train_ds.to_json(train_path)
    val_ds.to_json(val_path)

    id2label = {v: k for k, v in CANONICAL_LABELS.items()}
    train_counts = Counter(train_ds["label"])
    val_counts = Counter(val_ds["label"])

    with open(os.path.join(STATS_DIR, "preprocessing_summary.md"), "w", encoding="utf-8") as f:
        f.write("# ALETHEX NLI Preprocessing Summary (real data only)\n\n")
        f.write(f"- Unified train pairs: {len(train_ds)} -> `{train_path}`\n")
        f.write(f"- Unified validation pairs: {len(val_ds)} -> `{val_path}`\n\n")
        f.write("## Train class distribution\n\n")
        for label_id, count in sorted(train_counts.items()):
            f.write(f"- **{id2label[label_id]}**: {count}\n")
        f.write("\n## Validation class distribution\n\n")
        for label_id, count in sorted(val_counts.items()):
            f.write(f"- **{id2label[label_id]}**: {count}\n")
        f.write(
            "\nNote: `dataset/corpus.jsonl` (synthetic placeholder corpus) is intentionally excluded "
            "from this training data; it is only a tiny end-to-end pipeline smoke-test fixture.\n"
        )

    console.print(f"[bold green]Done.[/bold green] Train: {len(train_ds)}, Validation: {len(val_ds)}")


if __name__ == "__main__":
    main()
