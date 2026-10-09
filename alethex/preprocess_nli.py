"""
ALETHEX NLI Dataset Preprocessing Entrypoint.
Unifies MultiNLI, SNLI, Temporal-NLI, and All-NLI into canonical 3-way format.

Usage:
    python preprocess_nli.py
    python -m alethex.data.preprocess_nli
    alethex preprocess-nli
    alethex verify-dataset
"""
import sys
from pathlib import Path

# Ensure alethex package is importable when executed directly
current_dir = Path(__file__).resolve().parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

from alethex.data.preprocess_nli import main

if __name__ == "__main__":
    main()
