"""Batch-oriented verification of heat-extreme forecasts.

The public functions in this package deliberately work on canonical forecast
fields.  Source-specific naming and storage layout belong in model adapters.
"""

from .config import VerificationConfig, load_config

__all__ = ["VerificationConfig", "load_config"]
