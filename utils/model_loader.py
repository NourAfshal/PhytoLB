"""Load trained model artifacts with joblib."""

from __future__ import annotations

from pathlib import Path
from typing import Any, TypedDict

import joblib

from utils.windows_lightgbm import ensure_lightgbm_importable


class ModelBundle(TypedDict):
    model: Any
    species_encoder: Any
    imputer: Any
    feature_names: list[str]


def load_artifacts(models_dir: str | Path) -> ModelBundle:
    """
    Load LightGBM classifier, species encoder, imputer, and feature name list.

    Expected files under ``models_dir``:
    phyto_lightgbm_model.pkl, species_encoder.pkl, imputer.pkl, feature_names.pkl
    """
    root = Path(models_dir)
    required = [
        "phyto_lightgbm_model.pkl",
        "species_encoder.pkl",
        "imputer.pkl",
        "feature_names.pkl",
    ]
    missing = [name for name in required if not (root / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"Missing artifact(s) in {root.resolve()}: {', '.join(missing)}"
        )

    feature_names = joblib.load(root / "feature_names.pkl")
    if not isinstance(feature_names, list) or not all(
        isinstance(x, str) for x in feature_names
    ):
        raise TypeError("feature_names.pkl must be a list of strings.")

    species_encoder = joblib.load(root / "species_encoder.pkl")
    imputer = joblib.load(root / "imputer.pkl")

    ensure_lightgbm_importable()
    model = joblib.load(root / "phyto_lightgbm_model.pkl")

    return {
        "model": model,
        "species_encoder": species_encoder,
        "imputer": imputer,
        "feature_names": feature_names,
    }
