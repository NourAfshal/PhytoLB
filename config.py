"""Application settings (env-overridable defaults)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

_BASE = Path(__file__).resolve().parent


def _path_from_env(var: str, default: Path) -> Path:
    raw = os.environ.get(var)
    return Path(raw).expanduser() if raw else default


@dataclass(frozen=True, slots=True)
class Settings:
    models_dir: Path
    worldclim_dir: Path
    elevation_dir: Path


@lru_cache
def get_settings() -> Settings:
    return Settings(
        models_dir=_path_from_env("PHYTO_MODELS_DIR", _BASE / "models"),
        worldclim_dir=_path_from_env(
            "PHYTO_WORLDCLIM_DIR", _BASE / "data" / "worldclim"
        ),
        elevation_dir=_path_from_env(
            "PHYTO_ELEVATION_DIR", _BASE / "data" / "elevation"
        ),
    )
