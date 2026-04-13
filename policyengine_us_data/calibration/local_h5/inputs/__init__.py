"""Input-bundle contracts for local H5 publication.

This package owns typed descriptions of the artifact inputs needed for a
publication run. Future input-resolution helpers should live here.
"""

from .contracts import PublishingInputBundle

__all__ = ["PublishingInputBundle"]
