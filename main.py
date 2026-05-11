"""
PhytoLB — FastAPI backend + web UI for plant suitability (Lebanon).

Run: uvicorn main:app --reload --host 127.0.0.1 --port 8000
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

from config import Settings, get_settings
from utils.bundle_cache import get_bundle
from utils.constants import LEBANESE_CITIES
from utils.environment import discover_bio_rasters, sample_environment, summarize_missing
from utils.prediction import build_feature_frame, impute_and_predict_proba, suitability_labels

BASE_DIR = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))
app = FastAPI(
    title="PhytoLB",
    description="Plant suitability predictions for Lebanese locations.",
    version="1.0.0",
)
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")


class PathsOverride(BaseModel):
    models_dir: str | None = None
    worldclim_dir: str | None = None
    elevation_dir: str | None = None


class PathsBody(BaseModel):
    paths: PathsOverride | None = None


class PredictRequest(PathsBody):
    city: str | None = None
    latitude: float | None = None
    longitude: float | None = None
    species: str = Field(..., min_length=1)


class CompareRequest(PathsBody):
    city: str = Field(..., min_length=1)


class RasterStatus(BaseModel):
    worldclim_dir: str
    elevation_dir: str
    bio_rasters_found: int
    bio_expected: int = 19
    worldclim_exists: bool
    elevation_dir_exists: bool


def _effective_paths(settings: Settings, override: PathsOverride | None) -> tuple[str, str, str]:
    m = override.models_dir if override and override.models_dir else str(settings.models_dir)
    w = override.worldclim_dir if override and override.worldclim_dir else str(settings.worldclim_dir)
    e = override.elevation_dir if override and override.elevation_dir else str(settings.elevation_dir)
    return m, w, e


def _resolve_lon_lat(
    city: str | None,
    latitude: float | None,
    longitude: float | None,
) -> tuple[float, float, str]:
    if city and city in LEBANESE_CITIES:
        lat, lon = LEBANESE_CITIES[city]
        return lon, lat, city
    if city:
        raise HTTPException(status_code=400, detail=f"Unknown city: {city!r}.")
    raise HTTPException(
        status_code=400,
        detail="Provide either a known ``city``",
    )


@app.get("/", response_class=HTMLResponse)
async def index(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={},
    )


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/meta")
async def meta() -> JSONResponse:
    settings = get_settings()
    bundle_error: str | None = None
    species: list[str] = []
    try:
        b = get_bundle(str(settings.models_dir))
        classes = getattr(b["species_encoder"], "classes_", None)
        species = list(classes) if classes is not None else []
    except Exception as e:
        bundle_error = str(e)

    wc = settings.worldclim_dir
    bio_count = 0
    if wc.is_dir():
        try:
            bio_count = len(discover_bio_rasters(wc))
        except Exception:
            pass

    payload = {
        "cities": list(LEBANESE_CITIES.keys()),
        "species": species,
        "defaults": {
            "models_dir": str(settings.models_dir),
            "worldclim_dir": str(settings.worldclim_dir),
            "elevation_dir": str(settings.elevation_dir),
        },
        "raster_bio_count": bio_count,
        "model_ready": bundle_error is None,
        "model_error": bundle_error,
    }
    return JSONResponse(payload)


@app.get("/api/cities")
async def api_cities() -> dict[str, dict[str, float]]:
    """City name -> {\"lat\", \"lon\"}."""
    return {name: {"lat": t[0], "lon": t[1]} for name, t in LEBANESE_CITIES.items()}


@app.post("/api/raster-status")
async def raster_status(body: PathsBody = PathsBody()) -> RasterStatus:
    settings = get_settings()
    _, w, e = _effective_paths(settings, body.paths)
    wc_path = Path(w)
    el_path = Path(e)
    bio_count = 0
    if wc_path.is_dir():
        try:
            bio_count = len(discover_bio_rasters(wc_path))
        except FileNotFoundError:
            bio_count = 0
    return RasterStatus(
        worldclim_dir=w,
        elevation_dir=e,
        bio_rasters_found=bio_count,
        worldclim_exists=wc_path.is_dir(),
        elevation_dir_exists=el_path.is_dir(),
    )

@app.get("/api/feature-importance")
async def feature_importance() -> dict:
    settings = get_settings()
    try:
        bundle = get_bundle(str(settings.models_dir))
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Could not load model: {e}") from e

    model         = bundle["model"]
    feature_names = list(bundle["feature_names"])
    importances   = model.feature_importances_.tolist()

    return {"features": feature_names, "importances": importances}

@app.post("/api/predict")
async def predict(req: PredictRequest) -> dict:
    settings = get_settings()
    md, wd, ed = _effective_paths(settings, req.paths)
    try:
        bundle = get_bundle(md)
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Could not load model artifacts: {e}") from e

    lon, lat, label_used = _resolve_lon_lat(
        req.city,
        req.latitude,
        req.longitude,
    )

    try:
        env = sample_environment(wd, ed, lon, lat)
    except FileNotFoundError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Raster sampling failed: {e}") from e

    missing = summarize_missing(env)
    try:
        X = build_feature_frame(
            env, req.species, bundle["species_encoder"], bundle["feature_names"]
        )
        pos, proba = impute_and_predict_proba(bundle["model"], bundle["imputer"], X)
        p = float(pos[0])
        class_label, binary = suitability_labels(p)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}") from e

    return {
        "location_label": label_used,
        "latitude": lat,
        "longitude": lon,
        "species": req.species,
        "probability": round(p, 6),
        "class_label": class_label,
        "binary_prediction": binary,
        "positive_class_probability": p,
        "binary_at_05": p >= 0.5,
        "raster_missing_at_site": missing,
        "proba_full": (
            [[float(v) for v in row] for row in np.asarray(proba).tolist()]
            if hasattr(proba, "tolist")
            else None
        ),
    }


@app.post("/api/compare")
async def compare(req: CompareRequest) -> dict:
    settings = get_settings()
    md, wd, ed = _effective_paths(settings, req.paths)
    if req.city not in LEBANESE_CITIES:
        raise HTTPException(status_code=400, detail=f"Unknown city: {req.city!r}.")

    try:
        bundle = get_bundle(md)
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Could not load model artifacts: {e}") from e

    classes = getattr(bundle["species_encoder"], "classes_", None)
    species_list = list(classes) if classes is not None else []
    if not species_list:
        raise HTTPException(status_code=503, detail="Species encoder has no classes_.")

    lat, lon = LEBANESE_CITIES[req.city]
    try:
        env = sample_environment(wd, ed, float(lon), float(lat))
    except FileNotFoundError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Raster sampling failed: {e}") from e

    rows: list[dict] = []
    for sp in species_list:
        try:
            Xs = build_feature_frame(env, sp, bundle["species_encoder"], bundle["feature_names"])
            pos_s, _ = impute_and_predict_proba(bundle["model"], bundle["imputer"], Xs)
            pr = float(pos_s[0])
        except Exception:
            pr = float("nan")
        if np.isfinite(pr):
            lab, bin_ = suitability_labels(pr)
            rows.append(
                {
                    "species": sp,
                    "probability": round(pr, 6),
                    "class_label": lab,
                    "binary": bin_,
                }
            )
        else:
            rows.append(
                {
                    "species": sp,
                    "probability": None,
                    "class_label": "—",
                    "binary": "—",
                }
            )

    return {
        "city": req.city,
        "latitude": lat,
        "longitude": lon,
        "raster_missing_at_site": summarize_missing(env),
        "species_rows": rows,
    }
