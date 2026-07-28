"""Model adapters that translate source stores into canonical verification fields."""

from __future__ import annotations

from .aifs import AIFSEnsV2Adapter
from .base import CanonicalLead, ModelAdapter


def get_model_adapter(config) -> ModelAdapter:
    """Instantiate the configured adapter without leaking source names to metrics."""
    adapter = str(config.data["model"]["adapter"])
    if adapter in {"aifs_ens_v2", "compact_heat"}:
        return AIFSEnsV2Adapter(config)
    raise ValueError(f"Unsupported verification model adapter: {adapter}")


__all__ = ["AIFSEnsV2Adapter", "CanonicalLead", "ModelAdapter", "get_model_adapter"]
