#!/usr/bin/env python3
"""
Convenience CLI for retrieving, analyzing, and plotting ALETHEX training results.
Usage:
    python analyze_training_results.py
    python analyze_training_results.py --results-dir results/training_50_epochs
"""

import sys
from pathlib import Path
from alethex.evaluation.interpret_training import main

if __name__ == "__main__":
    main()
