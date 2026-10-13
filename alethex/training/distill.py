"""
Knowledge Distillation Engine for ALETHEX.
Distills large DeBERTa-v3 Cross-Encoder teacher into a fast, lightweight 22M-parameter student (alethex-mini).
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import click
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from rich.console import Console
from rich.table import Table
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_cosine_schedule_with_warmup

console = Console()


class NLIPairDataset(Dataset):
    """PyTorch Dataset for text premise-hypothesis pairs with optional labels."""

    def __init__(
        self,
        pairs: List[Dict[str, Any]],
        tokenizer: Any,
        max_length: int = 128,
    ):
        self.pairs = pairs
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.pairs[idx]
        premise = item.get("premise", item.get("sentence1", ""))
        hypothesis = item.get("hypothesis", item.get("sentence2", ""))
        label = item.get("label", 0)

        # Map string labels if necessary
        label_map = {"contradiction": 0, "entailment": 1, "neutral": 2}
        if isinstance(label, str):
            label = label_map.get(label.lower(), 0)

        encoded = self.tokenizer(
            premise,
            hypothesis,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "label": torch.tensor(int(label), dtype=torch.long),
        }


class KnowledgeDistillationTrainer:
    """
    Transfers reasoning capabilities from a heavy Cross-Encoder teacher model
    to a lightweight student model via soft-target KL-Divergence + cross-entropy loss.
    """

    def __init__(
        self,
        teacher_model_name: str = "cross-encoder/nli-deberta-v3-small",
        student_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        output_dir: str = "models/alethex-mini",
        temperature: float = 2.0,
        alpha: float = 0.5,
        device: str = "cpu",
    ):
        self.teacher_name = teacher_model_name
        self.student_name = student_model_name
        self.output_dir = Path(output_dir)
        self.temperature = temperature
        self.alpha = alpha
        self.device = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")

        logger.info(f"Loading teacher: {teacher_model_name} on {self.device}")
        self.teacher_tokenizer = AutoTokenizer.from_pretrained(teacher_model_name)
        self.teacher = AutoModelForSequenceClassification.from_pretrained(teacher_model_name)
        self.teacher.to(self.device)
        self.teacher.eval()
        for param in self.teacher.parameters():
            param.requires_grad = False

        logger.info(f"Loading student: {student_model_name} on {self.device}")
        self.student_tokenizer = AutoTokenizer.from_pretrained(student_model_name)
        self.student = AutoModelForSequenceClassification.from_pretrained(
            student_model_name,
            num_labels=3,
            id2label={0: "contradiction", 1: "entailment", 2: "neutral"},
            label2id={"contradiction": 0, "entailment": 1, "neutral": 2},
        )
        self.student.to(self.device)

    def train(
        self,
        train_pairs: List[Dict[str, Any]],
        eval_pairs: Optional[List[Dict[str, Any]]] = None,
        epochs: int = 3,
        batch_size: int = 8,
        lr: float = 3e-5,
        max_steps: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Runs the knowledge distillation training loop."""
        self.output_dir.mkdir(parents=True, exist_ok=True)

        train_dataset = NLIPairDataset(train_pairs, self.student_tokenizer)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        optimizer = torch.optim.AdamW(self.student.parameters(), lr=lr, weight_decay=0.01)
        total_steps = len(train_loader) * epochs if max_steps is None else max_steps
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(total_steps * 0.1),
            num_training_steps=total_steps,
        )

        kl_loss_fn = nn.KLDivLoss(reduction="batchmean")
        ce_loss_fn = nn.CrossEntropyLoss()

        logger.info(f"Starting distillation: {len(train_pairs)} pairs, {epochs} epochs, {total_steps} steps")
        global_step = 0
        loss_history = []

        self.student.train()
        for epoch in range(epochs):
            epoch_loss = 0.0
            for batch in train_loader:
                if max_steps is not None and global_step >= max_steps:
                    break

                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["label"].to(self.device)

                # Teacher inference (frozen)
                with torch.no_grad():
                    teacher_outputs = self.teacher(input_ids=input_ids, attention_mask=attention_mask)
                    teacher_logits = teacher_outputs.logits

                # Student forward
                student_outputs = self.student(input_ids=input_ids, attention_mask=attention_mask)
                student_logits = student_outputs.logits

                # KD loss
                soft_student = F.log_softmax(student_logits / self.temperature, dim=-1)
                soft_teacher = F.softmax(teacher_logits / self.temperature, dim=-1)
                loss_kd = kl_loss_fn(soft_student, soft_teacher) * (self.temperature ** 2)

                # Hard label cross-entropy loss
                loss_ce = ce_loss_fn(student_logits, labels)

                # Total loss
                total_loss = (self.alpha * loss_kd) + ((1.0 - self.alpha) * loss_ce)

                optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.student.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()

                loss_val = total_loss.item()
                epoch_loss += loss_val
                loss_history.append({"step": global_step, "loss": round(loss_val, 4)})
                global_step += 1

                if global_step % 20 == 0 or global_step == 1:
                    logger.info(f"Step {global_step}/{total_steps} | Loss: {loss_val:.4f} (KD: {loss_kd.item():.4f}, CE: {loss_ce.item():.4f})")

            avg_epoch_loss = epoch_loss / max(1, len(train_loader))
            logger.info(f"Epoch {epoch + 1}/{epochs} Complete | Avg Loss: {avg_epoch_loss:.4f}")

        # Evaluation on held-out set if available
        eval_metrics = {}
        if eval_pairs:
            eval_metrics = self.evaluate(eval_pairs, batch_size=batch_size)

        # Save distilled student checkpoint
        logger.info(f"Saving distilled student checkpoint to: {self.output_dir}")
        self.student.save_pretrained(str(self.output_dir))
        self.student_tokenizer.save_pretrained(str(self.output_dir))

        report = {
            "teacher_model": self.teacher_name,
            "student_model": self.student_name,
            "epochs": epochs,
            "total_steps": global_step,
            "final_loss": loss_history[-1]["loss"] if loss_history else 0.0,
            "eval_metrics": eval_metrics,
            "student_parameters": sum(p.numel() for p in self.student.parameters()),
        }
        with open(self.output_dir / "distillation_report.json", "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

        return report

    def evaluate(self, eval_pairs: List[Dict[str, Any]], batch_size: int = 16) -> Dict[str, float]:
        """Evaluates student accuracy on validation pairs."""
        dataset = NLIPairDataset(eval_pairs, self.student_tokenizer)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        self.student.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["label"].to(self.device)

                logits = self.student(input_ids=input_ids, attention_mask=attention_mask).logits
                preds = torch.argmax(logits, dim=-1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        acc = correct / max(1, total)
        logger.info(f"Distilled Student Eval Accuracy: {acc * 100:.2f}% ({correct}/{total})")
        return {"eval_accuracy": round(acc, 4), "total_samples": total}


def generate_synthetic_nli_pairs(num_samples: int = 50) -> List[Dict[str, Any]]:
    """Generates synthetic high-quality premise-hypothesis pairs with temporal variations."""
    templates = [
        ("Alice lives in London in 2024.", "Alice lives in London.", "entailment"),
        ("Alice works at Google.", "Alice works at Meta.", "contradiction"),
        ("Bob likes Python.", "Bob prefers Java.", "neutral"),
        ("The user moved to Tokyo in June.", "The user moved to Tokyo.", "entailment"),
        ("System memory is 16GB.", "System memory is 64GB.", "contradiction"),
        ("Charlie completed the project yesterday.", "Charlie is working on the project.", "neutral"),
    ]
    pairs = []
    for i in range(num_samples):
        tmpl = templates[i % len(templates)]
        pairs.append({
            "premise": tmpl[0],
            "hypothesis": tmpl[1],
            "label": tmpl[2],
        })
    return pairs


@click.command()
@click.option("--teacher", default="cross-encoder/nli-deberta-v3-small", help="Teacher model")
@click.option("--student", default="sentence-transformers/all-MiniLM-L6-v2", help="Student backbone")
@click.option("--output-dir", default="models/alethex-mini", help="Output directory")
@click.option("--epochs", default=2, type=int, help="Number of training epochs")
@click.option("--batch-size", default=8, type=int, help="Batch size")
@click.option("--max-steps", default=None, type=int, help="Max steps (smoke test)")
@click.option("--device", default="cpu", help="Device (cpu or cuda)")
def main(teacher: str, student: str, output_dir: str, epochs: int, batch_size: int, max_steps: Optional[int], device: str):
    """CLI to train distilled alethex-mini student model."""
    trainer = KnowledgeDistillationTrainer(
        teacher_model_name=teacher,
        student_model_name=student,
        output_dir=output_dir,
        device=device,
    )
    pairs = generate_synthetic_nli_pairs(60)
    trainer.train(
        train_pairs=pairs[:45],
        eval_pairs=pairs[45:],
        epochs=epochs,
        batch_size=batch_size,
        max_steps=max_steps,
    )


if __name__ == "__main__":
    main()
