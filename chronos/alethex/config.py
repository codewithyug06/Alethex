"""Configuration loader for ALETHEX.

Precedence for every setting: explicit constructor/function argument >
``ALETHEX_*`` environment variable > value in ``configs/*.yaml`` > hardcoded
fallback defined here. This module is the single place fallback defaults
live; other modules must not redefine their own copy of the same default.
"""
import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

# Repo-root anchor so relative paths work regardless of the caller's CWD.
PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent

# Current recommended Anthropic model (see https://docs.claude.com/en/docs/about-claude/models)
DEFAULT_ANTHROPIC_MODEL = "claude-sonnet-5"


def _env(name: str) -> Optional[str]:
    value = os.environ.get(f"ALETHEX_{name}")
    return value if value else None


def default_output_dir() -> Path:
    return Path(_env("OUTPUT_DIR") or (PROJECT_ROOT / "output"))


def default_config_dir() -> Path:
    """Resolve the config directory.

    Precedence: ``ALETHEX_CONFIG_DIR`` env var > repo-layout ``<repo>/configs``
    (development checkout) > ``<package>/configs`` (bundled with a pip install,
    where the repo root above the package no longer exists).
    """
    env_dir = _env("CONFIG_DIR")
    if env_dir:
        return Path(env_dir)
    repo_configs = PROJECT_ROOT / "configs"
    if repo_configs.exists():
        return repo_configs
    return PACKAGE_ROOT / "configs"


def default_home_dir() -> Path:
    """Root directory for ALETHEX-managed data (logs, models, datasets)."""
    return Path(_env("HOME") or PROJECT_ROOT)


def load_config(config_dir: Path, name: str) -> Dict[str, Any]:
    """Load a YAML config file."""
    path = config_dir / f"{name}.yaml"
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


def load_all_configs(config_dir: Path) -> Dict[str, Dict[str, Any]]:
    """Load all configs from a directory."""
    configs: Dict[str, Dict[str, Any]] = {}
    for path in config_dir.glob("*.yaml"):
        name = path.stem
        with open(path, "r", encoding="utf-8") as f:
            configs[name] = yaml.safe_load(f) or {}
    return configs


class Config:
    """Global config holder."""

    def __init__(self, config_dir: Path):
        self.configs = load_all_configs(config_dir)
        self.config_dir = config_dir

    @property
    def extraction(self) -> Dict[str, Any]:
        return self.configs.get("extraction", {})

    @property
    def entity_linking(self) -> Dict[str, Any]:
        return self.configs.get("entity_linking", {})

    @property
    def temporal(self) -> Dict[str, Any]:
        return self.configs.get("temporal", {})

    @property
    def nli(self) -> Dict[str, Any]:
        return self.configs.get("nli", {})

    @property
    def staleness_threshold_days(self) -> int:
        return int(self.temporal.get("staleness_threshold_days", 30))

    @property
    def same_source_window_seconds(self) -> int:
        return int(self.temporal.get("same_source_window_seconds", 60))

    @property
    def min_confidence(self) -> float:
        return float(self.extraction.get("min_confidence", 0.4))

    @property
    def entity_linking_eps(self) -> float:
        return float(self.entity_linking.get("eps", 0.25))

    @property
    def entity_linking_min_samples(self) -> int:
        return int(self.entity_linking.get("min_samples", 2))

    @property
    def nli_model(self) -> str:
        return _env("NLI_MODEL") or str(self.nli.get("model", "cross-encoder/nli-deberta-v3-large"))

    @property
    def nli_device(self) -> str:
        return _env("DEVICE") or str(self.nli.get("device", "cpu"))

    @property
    def nli_use_onnx(self) -> bool:
        env_val = _env("NLI_USE_ONNX")
        if env_val is not None:
            return env_val.lower() in ("1", "true", "yes")
        return bool(self.nli.get("use_onnx", False))

    @property
    def nli_threshold(self) -> float:
        return float(_env("NLI_THRESHOLD") or self.nli.get("threshold", 0.7))

    @property
    def contradiction_top_n(self) -> int:
        return int(self.nli.get("contradiction_top_n", 50))

    @property
    def anthropic_model(self) -> str:
        return _env("ANTHROPIC_MODEL") or str(self.nli.get("anthropic_model", DEFAULT_ANTHROPIC_MODEL))

    @property
    def entity_linking_model(self) -> str:
        return _env("ENTITY_LINK_MODEL") or str(self.entity_linking.get("model", "all-MiniLM-L6-v2"))

    @property
    def spacy_model(self) -> str:
        return _env("SPACY_MODEL") or str(self.extraction.get("spacy_model", "en_core_web_sm"))

    @property
    def output_dir(self) -> Path:
        return default_output_dir()

    @property
    def home_dir(self) -> Path:
        return default_home_dir()

    @property
    def logs_dir(self) -> Path:
        path = self.home_dir / "logs"
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def default_extraction_confidence(self) -> float:
        return float(self.extraction.get("default_confidence", 0.8))

    @property
    def context_memory(self) -> Dict[str, Any]:
        return self.configs.get("context_memory", {})

    @property
    def context_memory_max_claims_per_session(self) -> int:
        return int(self.context_memory.get("max_claims_per_session", 20))

    @property
    def context_memory_top_k(self) -> int:
        return int(self.context_memory.get("top_k", 8))

    @property
    def context_memory_max_context_tokens(self) -> int:
        return int(self.context_memory.get("max_context_tokens", 512))

    @property
    def context_memory_high_salience_predicates(self) -> list:
        return list(self.context_memory.get("high_salience_predicates", []))
