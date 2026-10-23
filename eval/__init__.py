"""HCX quality evaluation: run golden questions, record answers, human review, aggregate scores."""

import sys
from pathlib import Path

# ponytail: put src/ on the path so `python -m eval.*` can import kosis_ai; drop once T0-01 installs the package.
_SRC = str(Path(__file__).resolve().parents[1] / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
