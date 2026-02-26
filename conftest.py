"""Root conftest: mock optional dependencies before collection.

microimpute requires Python >= 3.12, so mock it for test
environments running an older interpreter.
"""

import sys
from unittest.mock import MagicMock

if "microimpute" not in sys.modules:
    _mock = MagicMock()
    sys.modules["microimpute"] = _mock
    sys.modules["microimpute.models"] = _mock
    sys.modules["microimpute.models.qrf"] = _mock
