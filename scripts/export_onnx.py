"""Export fine-tuned DeBERTa NLI checkpoint to quantized ONNX for Transformers.js."""
import argparse
import os
import sys
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


def export_and_quantize(checkpoint_path: str, out_dir_path: str):
    checkpoint = Path(checkpoint_path)
    out_dir = Path(out_dir_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading tokenizer and model from {checkpoint}...")
    tokenizer = AutoTokenizer.from_pretrained(str(checkpoint))
    model = AutoModelForSequenceClassification.from_pretrained(str(checkpoint))
    model.eval()

    # Save tokenizer + config alongside ONNX so Transformers.js can use them
    tokenizer.save_pretrained(str(out_dir))
    model.config.save_pretrained(str(out_dir))

    # Build a dummy input
    dummy = tokenizer(
        "I live in New York.",
        "I live in London.",
        return_tensors="pt",
        padding="max_length",
        max_length=128,
        truncation=True,
    )

    input_ids = dummy["input_ids"]
    attention_mask = dummy["attention_mask"]
    token_type_ids = dummy.get("token_type_ids")

    # DeBERTa-v3 uses token_type_ids optionally
    if token_type_ids is not None:
        inputs = (input_ids, attention_mask, token_type_ids)
        input_names = ["input_ids", "attention_mask", "token_type_ids"]
    else:
        inputs = (input_ids, attention_mask)
        input_names = ["input_ids", "attention_mask"]

    onnx_path = out_dir / "model.onnx"

    print(f"Exporting to {onnx_path} ...")
    with torch.no_grad():
        torch.onnx.export(
            model,
            inputs,
            str(onnx_path),
            input_names=input_names,
            output_names=["logits"],
            dynamic_axes={
                name: {0: "batch", 1: "sequence"} for name in input_names + ["logits"]
            },
            opset_version=14,
            do_constant_folding=True,
        )

    print(f"ONNX export done: {onnx_path.stat().st_size / 1e6:.1f} MB")

    # Quantize to INT8 to shrink ~4x for browser use
    print("Quantizing to INT8...")
    try:
        from onnxruntime.quantization import quantize_dynamic, QuantType
        quant_path = out_dir / "model_quantized.onnx"
        quantize_dynamic(str(onnx_path), str(quant_path), weight_type=QuantType.QInt8)
        print(f"Quantized model: {quant_path.stat().st_size / 1e6:.1f} MB  ->  {quant_path}")
    except Exception as e:
        print(f"Quantization skipped ({e}) -- using full FP32 model")
        quant_path = onnx_path

    print("\nDone. Files written to:", out_dir)
    for f in sorted(out_dir.iterdir()):
        print(f"  {f.name}  ({f.stat().st_size / 1e6:.2f} MB)")


def main():
    parser = argparse.ArgumentParser(
        description="Export fine-tuned DeBERTa NLI checkpoint to quantized ONNX for Transformers.js."
    )
    parser.add_argument(
        "--checkpoint",
        default="models/alethex-nli/checkpoint-124605",
        help="Path or HuggingFace ID of the PyTorch checkpoint (default: models/alethex-nli/checkpoint-124605)",
    )
    parser.add_argument(
        "--out-dir",
        default="models/alethex-nli-onnx",
        help="Target output directory for ONNX and config files (default: models/alethex-nli-onnx)",
    )
    args = parser.parse_args()
    export_and_quantize(args.checkpoint, args.out_dir)


if __name__ == "__main__":
    main()
