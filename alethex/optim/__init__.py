"""
ALETHEX Optimization & Inference Acceleration Module.
Provides ONNX export, dynamic INT8 quantization, and ultra-low-latency runtime pipelines.
"""

from alethex.optim.onnx_exporter import export_and_quantize_nli
from alethex.optim.onnx_pipeline import ONNXTextClassificationPipeline

__all__ = [
    "export_and_quantize_nli",
    "ONNXTextClassificationPipeline",
]
