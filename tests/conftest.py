import sys
from pathlib import Path

# Ensure alethex package directory is first on sys.path
alethex_pkg_dir = Path(__file__).resolve().parent.parent
if str(alethex_pkg_dir) not in sys.path:
    sys.path.insert(0, str(alethex_pkg_dir))
