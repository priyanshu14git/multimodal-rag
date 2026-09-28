"""
Makes `backend/` (for `from app.services... import ...`) and `notebooks/`
(for the eval script's pure functions) importable from tests without
installing this project as a package.

Order matters here: `backend/app/` is a package named `app`, and
`frontend/app.py` is a *module* also named `app`. If both were on
sys.path with frontend taking priority, `import app` would resolve to
the wrong one. backend goes first (highest priority); frontend is
appended at the end and is only there for `from utils import ...`,
which doesn't collide with anything.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT / "notebooks"))
sys.path.append(str(ROOT / "frontend"))