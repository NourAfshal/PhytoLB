"""LRU cache for model bundles keyed by models directory path."""

from __future__ import annotations

from functools import lru_cache

from utils.model_loader import ModelBundle, load_artifacts


@lru_cache(maxsize=8)
def get_bundle(models_dir: str) -> ModelBundle:
    """Load (and cache) artifacts for ``models_dir``."""
    return load_artifacts(models_dir)


def invalidate_bundle_cache() -> None:
    """Clear cached bundles (useful in tests)."""
    get_bundle.cache_clear()
