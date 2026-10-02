"""ALETHEX - Temporal Belief Consistency Engine"""

import os
import warnings

# Suppress noisy external warnings from third-party dependencies (Pydantic 2.10+ & Torchaudio)
os.environ["PYTHONWARNINGS"] = "ignore"
warnings.filterwarnings("ignore")

from alethex.api import ConsistencyEngine, check, filter_context
from alethex.config import Config
from alethex.integrations.universal_rag import UniversalRAG

__all__ = [
    "ConsistencyEngine",
    "check",
    "filter_context",
    "Config",
    "UniversalRAG",
]