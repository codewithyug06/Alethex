"""
ONNX Export and Dynamic INT8 Quantization Utility for ALETHEX Cross-Encoder NLI Models.
Reduces model footprint ~4x and delivers sub-5ms CPU latency.
"""

import json
import os
from pathlib import Path
from typing import Tuple, Union, Optional

import click
import torch
from loguru import logger
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from onnxruntime.quantization import QuantType, quantize_dynamic


def export_and_quantize_nli(
    model_name_or_path: Union[str, Path],
    output_dir: Union[str, Path],
    opset: int = 14,
) -> Tuple[Path, Path]:
    """
    Exports a PyTorch NLI Cross-Encoder model to ONNX and quantizes to dynamic INT8.

    Args:
        model_name_or_path: HuggingFace model ID or path to local checkpoint.
        output_dir: Target directory for saving ONNX and quantized artifacts.
        opset: ONNX operator set version (default 14).

    Returns:
        Tuple of (path_to_fp32_onnx, path_to_int8_onnx).
    """
    model_name_or_path = str(model_name_or_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fp32_onnx_path = output_dir / "model.onnx"
    int8_onnx_path = output_dir / "model_int8.onnx"

    logger.info(f"Loading model & tokenizer from: {model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path)
    model.eval()

    # Save tokenizer and config in the target directory
    tokenizer.save_pretrained(str(output_dir))
    model.config.save_pretrained(str(output_dir))

    # Sample input for tracing
    sample_text_a = "Alice lives in London."
    sample_text_b = "Alice resides in England."
    inputs = tokenizer(sample_text_a, sample_text_b, return_tensors="pt")

    logger.info(f"Exporting FP32 ONNX graph to: {fp32_onnx_path}")
    torch.onnx.export(
        model,
        (inputs["input_ids"], inputs["attention_mask"]),
        str(fp32_onnx_path),
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            "attention_mask": {0: "batch_size", 1: "sequence_length"},
            "logits": {0: "batch_size"},
        },
        opset_version=opset,
        do_constant_folding=True,
    )

    fp32_size_mb = os.path.getsize(fp32_onnx_path) / (1024 * 1024)
    logger.info(f"FP32 ONNX exported successfully: {fp32_size_mb:.2f} MB")

    logger.info(f"Applying dynamic INT8 quantization to: {int8_onnx_path}")
    quantize_dynamic(
        model_input=str(fp32_onnx_path),
        model_output=str(int8_onnx_path),
        weight_type=QuantType.QInt8,
    )

    int8_size_mb = os.path.getsize(int8_onnx_path) / (1024 * 1024)
    compression_ratio = fp32_size_mb / (int8_size_mb + 1e-6)
    logger.info(f"INT8 ONNX quantized successfully: {int8_size_mb:.2f} MB ({compression_ratio:.2f}x compression)")

    # Save metadata
    meta = {
        "source_model": model_name_or_path,
        "fp32_size_mb": round(fp32_size_mb, 2),
        "int8_size_mb": round(int8_size_mb, 2),
        "compression_ratio": round(compression_ratio, 2),
        "opset_version": opset,
        "id2label": getattr(model.config, "id2label", {}),
    }
    with open(output_dir / "quant_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    return fp32_onnx_path, int8_onnx_path


@click.command()
@click.option("--model", default="cross-encoder/nli-deberta-v3-small", help="Source model name or path")
@click.option("--output-dir", default="models/alethex-nli-int8", help="Output directory")
@click.option("--opset", default=14, type=int, help="ONNX opset version")
def main(model: str, output_dir: str, opset: int):
    """CLI to export and quantize NLI models for ultra-fast local inference."""
    export_and_quantize_nli(model, output_dir, opset=opset)


if __name__ == "__main__":
    main()
