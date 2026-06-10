"""Add semantic_graph root to sys.path for flat imports (dtype, graphrag, ...)."""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
