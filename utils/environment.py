"""WorldClim BIO1–BIO19 and elevation sampling from GeoTIFFs."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import rasterio
from rasterio.crs import CRS


def _is_nan_like(val: float) -> bool:
    return val is None or (isinstance(val, float) and np.isnan(val))


def _sanitize_sample_value(raw: float | np.floating, nodata: float | None) -> float:
    """Convert raster sample to float, treating nodata and non-finite as NaN."""
    try:
        v = float(np.asarray(raw).item())
    except (TypeError, ValueError):
        return float("nan")
    if not np.isfinite(v):
        return float("nan")
    if nodata is not None and np.isfinite(nodata) and np.isclose(v, float(nodata)):
        return float("nan")
    return v


def _read_band_value(dataset: rasterio.io.DatasetReader, lon: float, lat: float) -> float:
    """Sample a single-band dataset at lon/lat (assumes geographic or compatible CRS)."""
    try:
        xs, ys = float(lon), float(lat)
    except (TypeError, ValueError):
        return float("nan")

    crs: CRS | None = dataset.crs
    if crs is None or not crs.is_geographic:
        # Best-effort: still try index(); many Lebanon rasters are EPSG:4326
        pass

    try:
        row, col = dataset.index(xs, ys)
    except (ValueError, IndexError, OSError):
        return float("nan")

    h, w = int(dataset.height), int(dataset.width)
    if row < 0 or row >= h or col < 0 or col >= w:
        return float("nan")

    window = ((row, row + 1), (col, col + 1))
    data = dataset.read(1, window=window, masked=True)
    if np.ma.isMaskedArray(data):
        if bool(np.asarray(data.mask)[0, 0]):
            return float("nan")
        raw = data.data[0, 0]
    else:
        raw = data[0, 0]

    return _sanitize_sample_value(raw, dataset.nodata)


def discover_bio_rasters(worldclim_dir: str | Path) -> dict[str, Path]:
    """
    Map ``BIO1``..``BIO19`` to file paths under ``worldclim_dir``.

    Accepts common WorldClim naming patterns (e.g. ``wc2.1_30s_bio_1.tif``, ``BIO12.tif``).
    """
    root = Path(worldclim_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"WorldClim folder not found: {root.resolve()}")

    mapping: dict[str, Path] = {}
    patterns = [
        re.compile(r"(?:^|[_\-])(?:bio)[_\-]?(\d{1,2})(?:\.|$)", re.IGNORECASE),
        re.compile(r"BIO(\d{1,2})\b", re.IGNORECASE),
    ]

    for path in sorted(root.rglob("*")):
        if path.suffix.lower() not in {".tif", ".tiff"}:
            continue
        stem = path.stem
        idx: int | None = None
        for rx in patterns:
            m = rx.search(stem)
            if m:
                idx = int(m.group(1))
                break
        if idx is None or not (1 <= idx <= 19):
            continue
        key = f"BIO{idx}"
        if key not in mapping:
            mapping[key] = path

    return mapping


def discover_elevation_path(elevation_dir: str | Path) -> Path:
    """Return path to ``elevation.tif`` (case-insensitive) under ``elevation_dir``."""
    root = Path(elevation_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"Elevation folder not found: {root.resolve()}")

    candidates = list(root.rglob("*"))
    for path in candidates:
        if path.suffix.lower() not in {".tif", ".tiff"}:
            continue
        if path.stem.lower() == "elevation":
            return path
    raise FileNotFoundError(
        f"No elevation.tif (stem 'elevation') under {root.resolve()}"
    )


def sample_environment(
    worldclim_dir: str | Path,
    elevation_dir: str | Path,
    lon: float,
    lat: float,
) -> dict[str, float]:
    """
    Extract BIO1–BIO19 and elevation at (lon, lat).

    Missing or nodata cells become ``nan`` so downstream imputation can handle them.
    """
    bio_paths = discover_bio_rasters(worldclim_dir)
    elev_path = discover_elevation_path(elevation_dir)

    env: dict[str, float] = {}
    for i in range(1, 20):
        key = f"BIO{i}"
        path = bio_paths.get(key)
        if path is None:
            env[key] = float("nan")
            continue
        with rasterio.open(path) as ds:
            env[key] = _read_band_value(ds, lon, lat)

    with rasterio.open(elev_path) as ds:
        env["elevation"] = _read_band_value(ds, lon, lat)

    return env


def summarize_missing(env: dict[str, float]) -> list[str]:
    """Return list of feature keys that are NaN."""
    return [k for k, v in env.items() if _is_nan_like(v)]
