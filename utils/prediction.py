"""Feature engineering, alignment, imputation, and suitability labels."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def _g(d: dict[str, float], key: str) -> float:
    v = d.get(key, float("nan"))
    try:
        x = float(v)
    except (TypeError, ValueError):
        return float("nan")
    return x if np.isfinite(x) else float("nan")


def compute_engineered_features(env: dict[str, float]) -> dict[str, float]:
    """
    Derived variables matching training-time engineering.

    Uses NaN-aware arithmetic so missing BIOs propagate as NaN where appropriate.
    """
    out: dict[str, float] = {}
    b = lambda k: _g(env, k)  # noqa: E731

    out["temp_annual_range"] = b("BIO5") - b("BIO6")
    out["warm_cold_quarter_diff"] = b("BIO10") - b("BIO11")
    out["precip_wet_dry_month_diff"] = b("BIO13") - b("BIO14")
    out["precip_wet_dry_quarter_diff"] = b("BIO16") - b("BIO17")
    out["dryness_ratio"] = b("BIO17") / (b("BIO16") + 1e-6)
    out["summer_winter_precip_ratio"] = b("BIO18") / (b("BIO19") + 1e-6)
    out["elevation_temp_interaction"] = b("elevation") * b("BIO1")
    out["elevation_precip_interaction"] = b("elevation") * b("BIO12")
    return out


def encode_species(species_encoder: Any, species_label: str) -> int:
    """Return integer species code; raises ValueError if unknown."""
    classes = getattr(species_encoder, "classes_", None)
    if classes is None:
        raise TypeError("species_encoder must expose classes_ (e.g. sklearn LabelEncoder).")
    if species_label not in set(classes):
        raise ValueError(f"Unknown species: {species_label!r}. Pick one from the training list.")
    idx = species_encoder.transform([species_label])[0]
    return int(idx)


def build_feature_frame(
    env: dict[str, float],
    species_label: str,
    species_encoder: Any,
    feature_names: list[str],
) -> pd.DataFrame:
    """
    Build a single-row DataFrame with columns exactly ``feature_names`` (same order).

    Base env keys (BIO*, elevation) plus ``species_id`` and any engineered names
    present in ``feature_names`` are filled; others are left NaN for the imputer.
    """
    row: dict[str, float] = {}
    for k, v in env.items():
        try:
            row[k] = float(v)
        except (TypeError, ValueError):
            row[k] = float("nan")

    engineered = compute_engineered_features(env)
    for name in feature_names:
        if name in engineered:
            row[name] = engineered[name]

    if "species_id" in feature_names:
        row["species_id"] = float(encode_species(species_encoder, species_label))

    data = {col: [row.get(col, float("nan"))] for col in feature_names}
    return pd.DataFrame(data, columns=feature_names)


def impute_and_predict_proba(
    model: Any,
    imputer: Any,
    X: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply imputer then return positive-class probability and full proba."""
    try:
        X_t = imputer.transform(X)
        X_t = pd.DataFrame(X_t, columns=X.columns)
    except AttributeError as e:
        # Common when the imputer was pickled with a different sklearn version:
        # e.g. 'SimpleImputer' object has no attribute '_fill_dtype'.
        if "_fill_dtype" not in str(e):
            raise

        stats = getattr(imputer, "statistics_", None)
        if stats is None:
            raise

        # Manual SimpleImputer.transform fallback using stored per-feature means.
        # This keeps behavior aligned with the fitted statistics and avoids
        # relying on internal sklearn attributes that may not exist after unpickle.
        X_arr = X.to_numpy(dtype=float, copy=True)
        missing_mask = np.isnan(X_arr)

        stats_arr = np.asarray(stats, dtype=float).reshape(-1)
        if stats_arr.shape[0] != X_arr.shape[1]:
            raise RuntimeError(
                f"Imputer statistics length mismatch: got {stats_arr.shape[0]}, expected {X_arr.shape[1]}"
            )

        # Replace NaNs with learned statistics per column.
        for j in range(X_arr.shape[1]):
            col_missing = missing_mask[:, j]
            if not np.any(col_missing):
                continue
            fill = float(stats_arr[j])
            if np.isfinite(fill):
                X_arr[col_missing, j] = fill
            # else: keep NaNs (imputer would also fail or propagate)

        X_t = X_arr

    proba = model.predict_proba(X_t)
    if proba.shape[1] < 2:
        pos = proba[:, 0]
    else:
        pos = proba[:, 1]
    return pos, proba


def suitability_labels(probability: float) -> tuple[str, str]:
    """
    Return (multiclass label, binary label) for suitability probability.

    Binary uses threshold 0.5 on the positive (suitable) class probability.
    """
    p = float(probability)
    if p >= 0.75:
        label = "Highly suitable"
    elif p >= 0.50:
        label = "Suitable"
    elif p >= 0.30:
        label = "Moderately suitable"
    else:
        label = "Low / not suitable"
    binary = "Suitable (≥0.5)" if p >= 0.5 else "Not suitable (<0.5)"
    return label, binary
