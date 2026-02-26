"""Shared fixtures for calibration tests.

Mocks microimpute at import time so tests can run without the
package installed (it requires Python >= 3.12).
"""

import sys
from unittest.mock import MagicMock

# microimpute is not installable on Python < 3.12.  Mock it before
# any test module triggers the cps.py top-level import.
if "microimpute" not in sys.modules:
    sys.modules["microimpute"] = MagicMock()
    sys.modules["microimpute.models"] = MagicMock()
    sys.modules["microimpute.models.qrf"] = MagicMock()
