"""
Ultra-fast drop-in ONNX Runtime text-classification pipeline for ALETHEX.
Matches HuggingFace pipeline interface for zero code changes in inference callers.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer


class ONNXTextClassificationPipeline:
    """
    Drop-in replacement for `transformers.pipeline("text-classification", top_k=None)`
    powered by ONNX Runtime with dynamic INT8 acceleration.
    """

    def __init__(
        self,
        model_path_or_dir: Union[str, Path],
        device: str = "cpu",
        prefer_quantized: bool = True,
    ):
        model_path = Path(model_path_or_dir)

        if model_path.is_dir():
            tokenizer_dir = model_path
            if prefer_quantized and (model_path / "model_int8.onnx").exists():
                onnx_file = model_path / "model_int8.onnx"
            elif (model_path / "model.onnx").exists():
                onnx_file = model_path / "model.onnx"
            elif (model_path / "model_int8.onnx").exists():
                onnx_file = model_path / "model_int8.onnx"
            else:
                raise FileNotFoundError(f"No model.onnx or model_int8.onnx found in {model_path}")
        else:
            onnx_file = model_path
            tokenizer_dir = model_path.parent

        self.tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_dir))

        # Determine execution providers
        providers = ["CPUExecutionProvider"]
        if device == "cuda" and "CUDAExecutionProvider" in ort.get_available_providers():
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(str(onnx_file), sess_options=opts, providers=providers)

        # Load label mapping
        config_file = tokenizer_dir / "config.json"
        if config_file.exists():
            with open(config_file, "r", encoding="utf-8") as f:
                cfg = json.load(f)
                raw_id2label = cfg.get("id2label", {})
                self.id2label = {int(k): v for k, v in raw_id2label.items()}
        else:
            self.id2label = {0: "contradiction", 1: "entailment", 2: "neutral"}

        # Mock model property for resolve_nli_label_map inspection
        self.model = type("MockModel", (), {"config": type("MockConfig", (), {"id2label": self.id2label})()})()

    def __call__(
        self,
        inputs: Union[Dict[str, str], List[Dict[str, str]]],
        **kwargs: Any,
    ) -> List[List[Dict[str, Any]]]:
        """
        Runs batch classification over one or multiple text pairs.

        Returns:
            List of lists of dicts with 'label' and 'score', matching HuggingFace top_k=None.
        """
        is_single = isinstance(inputs, dict)
        batch = [inputs] if is_single else inputs

        texts_a = [item.get("text", "") for item in batch]
        texts_b = [item.get("text_pair", "") for item in batch]

        tokenized = self.tokenizer(
            texts_a,
            texts_b,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="np",
        )

        ort_inputs = {
            "input_ids": tokenized["input_ids"].astype(np.int64),
            "attention_mask": tokenized["attention_mask"].astype(np.int64),
        }

        logits = self.session.run(["logits"], ort_inputs)[0]

        # Softmax over logits
        exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
        probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

        batch_results = []
        for row in probs:
            row_results = []
            for idx, prob in enumerate(row):
                label_name = self.id2label.get(idx, f"LABEL_{idx}")
                row_results.append({
                    "label": label_name,
                    "score": float(prob),
                })
            row_results.sort(key=lambda x: x["score"], reverse=True)
            batch_results.append(row_results)

        return batch_results if not is_single else batch_results[0]
