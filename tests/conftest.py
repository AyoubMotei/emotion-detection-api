# tests/conftest.py
import sys
from pathlib import Path

# project root is one level above tests/
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
