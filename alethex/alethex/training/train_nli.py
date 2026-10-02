import argparse
import os

import numpy as np
import yaml
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from rich.console import Console

console = Console()

ALETHEX_HOME = os.environ.get("ALETHEX_HOME", os.getcwd())
DEFAULT_CONFIG_PATH = os.path.join(ALETHEX_HOME, "configs", "training.yaml")

# Chance-level floor a 3-way classifier must clear (with positive MCC) before a
# full run is allowed to proceed. Catches a repeat of the label-permutation bug
# that previously produced 5.4% accuracy / MCC -0.42 without wasting ~13.5h.
SMOKE_TEST_MIN_ACCURACY = 0.45
SMOKE_TEST_MIN_MCC = 0.0


def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "macro_f1": f1_score(labels, preds, average="macro"),
        "mcc": matthews_corrcoef(labels, preds),
    }


def tokenize_function(tokenizer, examples):
    return tokenizer(examples["premise"], examples["hypothesis"], padding="max_length", truncation=True, max_length=128)


def train_nli(config_path=DEFAULT_CONFIG_PATH, test_mode=False):
    cfg = load_config(config_path)
    model_name = cfg["model_name"]
    output_dir = os.path.join(ALETHEX_HOME, cfg["output_dir"])
    train_path = os.path.join(ALETHEX_HOME, cfg["preprocessed_train"])
    val_path = os.path.join(ALETHEX_HOME, cfg["preprocessed_validation"])

    for p in (train_path, val_path):
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"Required preprocessed dataset not found: {p}\n"
                "Run `python -m alethex.data.preprocess_nli` first (never falls back to synthetic corpus.jsonl)."
            )

    console.print(f"[bold blue]Loading base model {model_name}[/bold blue]")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3, ignore_mismatched_sizes=True)

    console.print(f"[bold blue]Loading real preprocessed datasets from {train_path} / {val_path}[/bold blue]")
    train_split = "train[:200]" if test_mode else "train"
    val_split = "train[:50]" if test_mode else "train"
    train_ds = load_dataset("json", data_files=train_path, split=train_split)
    val_ds = load_dataset("json", data_files=val_path, split=val_split)
    console.print(f"Loaded {len(train_ds)} train pairs, {len(val_ds)} validation pairs")

    train_ds = train_ds.map(lambda ex: tokenize_function(tokenizer, ex), batched=True)
    val_ds = val_ds.map(lambda ex: tokenize_function(tokenizer, ex), batched=True)

    training_args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",
        save_strategy="no" if test_mode else "epoch",
        learning_rate=cfg["learning_rate"],
        per_device_train_batch_size=cfg["test_batch_size"] if test_mode else cfg["batch_size"],
        per_device_eval_batch_size=cfg["per_device_eval_batch_size"],
        gradient_accumulation_steps=cfg["test_grad_accum"] if test_mode else cfg["grad_accum"],
        num_train_epochs=1 if test_mode else cfg["epochs"],
        weight_decay=cfg["weight_decay"],
        warmup_steps=cfg["warmup_ratio"],  # this transformers version accepts a float ratio (<1) here
        save_total_limit=cfg["save_total_limit"],
        logging_steps=cfg["logging_steps_test_mode"] if test_mode else cfg["logging_steps"],
        # fp16/gradient_checkpointing were hardcoded True because the 6.4GB laptop GPU needs them to
        # avoid VRAM overflow (Windows silently falls back to "shared GPU memory" otherwise -- the
        # ~575s/step, ~4.5-year-ETA run that had to be killed). A GPU with ample VRAM (e.g. an H100)
        # doesn't need either and is faster without them, so these are now config-driven per device
        # profile instead of hardcoded, defaulting to the laptop-safe values if unset.
        fp16=cfg.get("fp16", True),
        bf16=cfg.get("bf16", False),
        gradient_checkpointing=cfg.get("gradient_checkpointing", True),
        optim=cfg["optimizer"],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
    )

    console.print("[bold yellow]Starting Training Loop...[/bold yellow]")
    trainer.train()

    metrics = trainer.evaluate()
    console.print(f"[bold cyan]Final eval metrics: {metrics}[/bold cyan]")

    if test_mode:
        acc = metrics.get("eval_accuracy", 0.0)
        mcc = metrics.get("eval_mcc", -1.0)
        if acc < SMOKE_TEST_MIN_ACCURACY or mcc < SMOKE_TEST_MIN_MCC:
            console.print(
                f"[bold red]Smoke test FAILED gate (accuracy={acc:.3f} < {SMOKE_TEST_MIN_ACCURACY} or "
                f"mcc={mcc:.3f} < {SMOKE_TEST_MIN_MCC}). Do NOT start the full run — "
                "check label mapping / preprocessing before retrying.[/bold red]"
            )
            raise SystemExit(1)
        console.print("[bold green]Smoke test PASSED gate. Safe to start the full training run.[/bold green]")
    else:
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        console.print(f"[bold green]Training complete! Model saved to {output_dir}[/bold green]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true", help="Run a fast smoke test (subset + 1 epoch) with a hard accuracy/MCC gate before the full run")
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH, help="Path to training.yaml")
    args = parser.parse_args()

    train_nli(config_path=args.config, test_mode=args.test)
