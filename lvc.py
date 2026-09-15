"""Convenience entry point for the command line surface.

The package lives under ``src/``, so importing it normally requires putting that
directory on ``sys.path`` first. ``python -m local_voice_companion`` therefore
fails from a fresh checkout, and every documented command would need an
``$env:PYTHONPATH = "src"`` preamble. This shim removes that step so the commands
in the README can be copied verbatim:

    python lvc.py doctor
    python lvc.py models status
    python lvc.py plan

It holds no logic of its own. ``app.py`` already does the same job for the server
entry point; anything new belongs in the package (RULE 15).
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from local_voice_companion.__main__ import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
