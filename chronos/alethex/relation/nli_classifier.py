from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple, List, Union

from transformers import pipeline

from alethex.config import Config, default_config_dir, PROJECT_ROOT
from alethex.ingestion.schema import Claim, ClaimPair
from alethex.optim.onnx_pipeline import ONNXTextClassificationPipeline

# Explicit NLI label-index mappings per known model family. A model's config
# (id2label) can differ in label order, so guessing generically is unsafe -
# unknown models must be registered here or classification fails loudly
# rather than silently mislabeling entailment/contradiction.
KNOWN_LABEL_MAPS: Dict[str, Dict[str, str]] = {
    "cross-encoder/nli-deberta-v3-large": {
        "LABEL_0": "CONTRADICTION", "LABEL_1": "ENTAILMENT", "LABEL_2": "NEUTRAL",
        "CONTRADICTION": "CONTRADICTION", "ENTAILMENT": "ENTAILMENT", "NEUTRAL": "NEUTRAL",
        "contradiction": "CONTRADICTION", "entailment": "ENTAILMENT", "neutral": "NEUTRAL",
    },
    "cross-encoder/nli-deberta-v3-small": {
        "LABEL_0": "CONTRADICTION", "LABEL_1": "ENTAILMENT", "LABEL_2": "NEUTRAL",
        "CONTRADICTION": "CONTRADICTION", "ENTAILMENT": "ENTAILMENT", "NEUTRAL": "NEUTRAL",
        "contradiction": "CONTRADICTION", "entailment": "ENTAILMENT", "neutral": "NEUTRAL",
    },
    "cross-encoder/nli-deberta-base": {
        "LABEL_0": "CONTRADICTION", "LABEL_1": "ENTAILMENT", "LABEL_2": "NEUTRAL",
        "CONTRADICTION": "CONTRADICTION", "ENTAILMENT": "ENTAILMENT", "NEUTRAL": "NEUTRAL",
        "contradiction": "CONTRADICTION", "entailment": "ENTAILMENT", "neutral": "NEUTRAL",
    },
    "facebook/bart-large-mnli": {
        "LABEL_0": "CONTRADICTION", "LABEL_1": "NEUTRAL", "LABEL_2": "ENTAILMENT",
        "CONTRADICTION": "CONTRADICTION", "ENTAILMENT": "ENTAILMENT", "NEUTRAL": "NEUTRAL",
        "contradiction": "CONTRADICTION", "entailment": "ENTAILMENT", "neutral": "NEUTRAL",
    },
    "models/alethex-nli": {
        "LABEL_0": "CONTRADICTION", "LABEL_1": "ENTAILMENT", "LABEL_2": "NEUTRAL",
        "0": "CONTRADICTION", "1": "ENTAILMENT", "2": "NEUTRAL",
        "CONTRADICTION": "CONTRADICTION", "ENTAILMENT": "ENTAILMENT", "NEUTRAL": "NEUTRAL",
        "contradiction": "CONTRADICTION", "entailment": "ENTAILMENT", "neutral": "NEUTRAL",
    },
    "alethex-nli": {
        "LABEL_0": "CONTRADICTION", "LABEL_1": "ENTAILMENT", "LABEL_2": "NEUTRAL",
        "0": "CONTRADICTION", "1": "ENTAILMENT", "2": "NEUTRAL",
        "CONTRADICTION": "CONTRADICTION", "ENTAILMENT": "ENTAILMENT", "NEUTRAL": "NEUTRAL",
        "contradiction": "CONTRADICTION", "entailment": "ENTAILMENT", "neutral": "NEUTRAL",
    },
    "models/alethex-nli-int8": {
        "LABEL_0": "CONTRADICTION", "LABEL_1": "ENTAILMENT", "LABEL_2": "NEUTRAL",
        "0": "CONTRADICTION", "1": "ENTAILMENT", "2": "NEUTRAL",
        "CONTRADICTION": "CONTRADICTION", "ENTAILMENT": "ENTAILMENT", "NEUTRAL": "NEUTRAL",
        "contradiction": "CONTRADICTION", "entailment": "ENTAILMENT", "neutral": "NEUTRAL",
    },
    "alethex-nli-int8": {
        "LABEL_0": "CONTRADICTION", "LABEL_1": "ENTAILMENT", "LABEL_2": "NEUTRAL",
        "0": "CONTRADICTION", "1": "ENTAILMENT", "2": "NEUTRAL",
        "CONTRADICTION": "CONTRADICTION", "ENTAILMENT": "ENTAILMENT", "NEUTRAL": "NEUTRAL",
        "contradiction": "CONTRADICTION", "entailment": "ENTAILMENT", "neutral": "NEUTRAL",
    },
    "alethex-mini": {
        "LABEL_0": "CONTRADICTION", "LABEL_1": "ENTAILMENT", "LABEL_2": "NEUTRAL",
        "0": "CONTRADICTION", "1": "ENTAILMENT", "2": "NEUTRAL",
        "CONTRADICTION": "CONTRADICTION", "ENTAILMENT": "ENTAILMENT", "NEUTRAL": "NEUTRAL",
        "contradiction": "CONTRADICTION", "entailment": "ENTAILMENT", "neutral": "NEUTRAL",
    },
    "models/alethex-mini": {
        "LABEL_0": "CONTRADICTION", "LABEL_1": "ENTAILMENT", "LABEL_2": "NEUTRAL",
        "0": "CONTRADICTION", "1": "ENTAILMENT", "2": "NEUTRAL",
        "CONTRADICTION": "CONTRADICTION", "ENTAILMENT": "ENTAILMENT", "NEUTRAL": "NEUTRAL",
        "contradiction": "CONTRADICTION", "entailment": "ENTAILMENT", "neutral": "NEUTRAL",
    },
}


class UnknownNLILabelMappingError(ValueError):
    """Raised when a model's output labels cannot be safely mapped to entailment/contradiction/neutral."""


def resolve_nli_label_map(model_name: str, pipeline_model: Optional[Any] = None) -> Dict[str, str]:
    """Resolves label mapping for known models, local checkpoints, or custom HF pipelines."""
    # 1. Direct match in KNOWN_LABEL_MAPS
    if model_name in KNOWN_LABEL_MAPS:
        return KNOWN_LABEL_MAPS[model_name]

    # 2. Check if path/name matches alethex-nli, alethex-mini, or common checkpoint names
    name_or_stem = Path(model_name).name.lower()
    if "alethex-nli" in str(model_name).lower() or name_or_stem in ("alethex-nli", "alethex-nli-int8", "checkpoint-500", "checkpoint-13500", "checkpoint-13635"):
        return KNOWN_LABEL_MAPS["alethex-nli"]
    if "alethex-mini" in str(model_name).lower() or name_or_stem == "alethex-mini":
        return KNOWN_LABEL_MAPS["alethex-mini"]

    # 3. Dynamic inspection of pipeline model config id2label
    if pipeline_model is not None and hasattr(pipeline_model, "config") and hasattr(pipeline_model.config, "id2label"):
        id2label = {str(k): str(v).lower() for k, v in pipeline_model.config.id2label.items()}
        known_classes = {"contradiction", "entailment", "neutral"}
        if set(id2label.values()).issubset(known_classes):
            mapping: Dict[str, str] = {}
            for idx_str, name in id2label.items():
                upper_name = name.upper()
                mapping[f"LABEL_{idx_str}"] = upper_name
                mapping[idx_str] = upper_name
                mapping[name] = upper_name
                mapping[upper_name] = upper_name
            return mapping

    raise UnknownNLILabelMappingError(
        f"No NLI label mapping registered for model '{model_name}'. "
        f"Add an entry to KNOWN_LABEL_MAPS in nli_classifier.py mapping this "
        f"model's output labels to CONTRADICTION/ENTAILMENT/NEUTRAL before using it."
    )


def _to_naive(dt: Optional[datetime]) -> Optional[datetime]:
    if dt is None:
        return None
    if dt.tzinfo is not None:
        return dt.astimezone(timezone.utc).replace(tzinfo=None)
    return dt


class NLIClassifier:
    _pipeline_cache: Dict[Tuple[str, Optional[str], bool], Any] = {}

    def __init__(
        self,
        model_name: Optional[str] = None,
        dummy: bool = False,
        device: Optional[str] = None,
        config: Optional[Config] = None,
        use_onnx: bool = False,
        **kwargs: Any,
    ):
        try:
            cfg = config or Config(default_config_dir())
            default_model = cfg.nli_model
            default_device = cfg.nli_device
            default_use_onnx = getattr(cfg, "nli_use_onnx", False)
            home_dir = cfg.home_dir
        except Exception:
            default_model = "models/alethex-nli"
            default_device = "cpu"
            default_use_onnx = False
            home_dir = PROJECT_ROOT

        model_name = model_name or default_model
        device = device or default_device
        use_onnx = use_onnx or kwargs.get("use_onnx", False) or default_use_onnx

        # If model_name is a relative local path, resolve against home_dir or PROJECT_ROOT if it exists
        if model_name:
            local_candidate = home_dir / model_name
            if local_candidate.exists():
                model_name = str(local_candidate.resolve())
            elif (PROJECT_ROOT / model_name).exists():
                model_name = str((PROJECT_ROOT / model_name).resolve())
            elif Path(model_name).exists():
                model_name = str(Path(model_name).resolve())

        is_onnx_dir = Path(model_name).is_dir() and ((Path(model_name) / "model_int8.onnx").exists() or (Path(model_name) / "model.onnx").exists())
        is_onnx = use_onnx or is_onnx_dir or str(model_name).endswith(".onnx")

        self.model_name = model_name
        self.device = device
        self.dummy = dummy
        self.is_onnx = is_onnx

        if dummy:
            self.classifier = None
            try:
                self.label_map = resolve_nli_label_map(model_name)
            except Exception:
                self.label_map = {
                    "LABEL_0": "CONTRADICTION", "LABEL_1": "ENTAILMENT", "LABEL_2": "NEUTRAL",
                    "0": "CONTRADICTION", "1": "ENTAILMENT", "2": "NEUTRAL",
                    "CONTRADICTION": "CONTRADICTION", "ENTAILMENT": "ENTAILMENT", "NEUTRAL": "NEUTRAL",
                    "contradiction": "CONTRADICTION", "entailment": "ENTAILMENT", "neutral": "NEUTRAL",
                }
        else:
            cache_key = (model_name, device, is_onnx)
            if cache_key not in NLIClassifier._pipeline_cache:
                if is_onnx:
                    NLIClassifier._pipeline_cache[cache_key] = ONNXTextClassificationPipeline(
                        model_name,
                        device=device,
                    )
                else:
                    dev_idx = 0 if device == "cuda" else -1
                    NLIClassifier._pipeline_cache[cache_key] = pipeline(
                        "text-classification",
                        model=model_name,
                        top_k=None,
                        device=dev_idx,
                    )
            self.classifier = NLIClassifier._pipeline_cache[cache_key]
            model_obj = getattr(self.classifier, "model", None)
            self.label_map = resolve_nli_label_map(model_name, pipeline_model=model_obj)

    def _intervals_overlap(self, c1: Claim, c2: Claim) -> bool:
        start1 = _to_naive(c1.validity_start or c1.timestamp)
        end1 = _to_naive(c1.validity_end)
        start2 = _to_naive(c2.validity_start or c2.timestamp)
        end2 = _to_naive(c2.validity_end)

        if end1 is not None and start2 is not None and start2 > end1:
            return False
        if end2 is not None and start1 is not None and start1 > end2:
            return False
        return True

    def classify_pair(self, claim_a: Claim, claim_b: Claim, threshold: float = 0.7) -> ClaimPair:
        if self.dummy and self.classifier is None:
            return ClaimPair(claim_a=claim_a, claim_b=claim_b, relation="ambiguous", confidence=0.0, justification="dummy")

        premise = f"[{claim_a.timestamp.date()}] {claim_a.subject} {claim_a.predicate} {claim_a.object_}"
        hypothesis = f"[{claim_b.timestamp.date()}] {claim_b.subject} {claim_b.predicate} {claim_b.object_}"

        results = self.classifier({"text": premise, "text_pair": hypothesis})

        if isinstance(results, list) and isinstance(results[0], list):
            results = results[0]

        scores = {res["label"].upper(): res["score"] for res in results}

        mapped_scores: Dict[str, float] = {}
        for label, score in scores.items():
            mapped_lbl = self.label_map.get(label, self.label_map.get(label.lower(), label))
            mapped_scores[mapped_lbl] = score

        entailment = mapped_scores.get("ENTAILMENT", scores.get("ENTAILMENT", scores.get("LABEL_1", 0.0)))
        contradiction = mapped_scores.get("CONTRADICTION", scores.get("CONTRADICTION", scores.get("LABEL_2", scores.get("LABEL_0", 0.0))))

        if entailment > threshold:
            return ClaimPair(claim_a=claim_a, claim_b=claim_b, relation="consistent", confidence=entailment, justification="NLI entailment")

        t_a = _to_naive(claim_a.timestamp)
        t_b = _to_naive(claim_b.timestamp)
        if contradiction > threshold:
            if t_b is not None and t_a is not None and t_b > t_a and not self._intervals_overlap(claim_a, claim_b):
                return ClaimPair(claim_a=claim_a, claim_b=claim_b, relation="superseded", confidence=contradiction, justification="NLI contradiction, B is newer")
            else:
                return ClaimPair(claim_a=claim_a, claim_b=claim_b, relation="conflicting", confidence=contradiction, justification="NLI contradiction, overlapping intervals")

        return ClaimPair(claim_a=claim_a, claim_b=claim_b, relation="ambiguous", confidence=max(entailment, contradiction), justification="Below threshold")
