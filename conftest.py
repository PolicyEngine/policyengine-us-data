"""Root conftest: mock optional dependencies before collection.

microimpute requires Python >= 3.12, so mock it for test
environments running an older interpreter.  Only install the
mock when the real package is unavailable — otherwise tests
that trigger CPS generation (which uses QRF imputation) would
silently get a MagicMock instead of the real model.
"""

import sys
from unittest.mock import MagicMock

try:
    import microimpute  # noqa: F401
except ImportError:
    _mock = MagicMock()
    sys.modules["microimpute"] = _mock
    sys.modules["microimpute.models"] = _mock
    sys.modules["microimpute.models.qrf"] = _mock
