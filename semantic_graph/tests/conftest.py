import sys
from pathlib import Path

# semantic_graph modules use flat imports (graphrag, schemas, prompts).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
