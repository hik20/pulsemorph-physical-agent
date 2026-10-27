"""Launch the request-bound-goal version; preserve the first guarded failure."""
from pathlib import Path
import sys

root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(root / 'src'))
from pulsemorph.agentic.app.phrase.entrypoint_v2 import prepare_parent_imports
prepare_parent_imports(root)
from pulsemorph.agentic.expression.server_v3 import main
main()
