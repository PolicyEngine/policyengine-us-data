"""Shared fixtures for calibration tests.

Mocks microimpute at import time so tests can run without the
package installed (it requires Python >= 3.12).  Only install the
mock when the real package is unavailable.
"""

import sys
from unittest.mock import MagicMock

try:
    import microimpute  # noqa: F401
except ImportError:
    # microimpute is not installable on Python < 3.12.  Mock it
    # before any test module triggers the cps.py top-level import.
    sys.modules["microimpute"] = MagicMock()
    sys.modules["microimpute.models"] = MagicMock()
    sys.modules["microimpute.models.qrf"] = MagicMock()
