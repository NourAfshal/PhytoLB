# PhytoLB

Plant suitability predictions for Lebanese locations using a trained **LightGBM** classifier, **WorldClim** BIO1–BIO19 rasters, and an **elevation** GeoTIFF. The app uses **FastAPI** for HTTP APIs and serves a browser UI (templates + Plotly JS). It loads saved artifacts only (no training in the app).

## Requirements

- Python 3.10+ recommended
- Trained artifacts in `models/` (see below)
- GeoTIFFs for WorldClim BIO1–BIO19 and `elevation.tif` in configurable folders

## Setup

Create a virtual environment (ignored by git) and install dependencies:

```bash
python -m venv venv
```

**Windows (PowerShell):**

```powershell
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

**macOS / Linux:**

```bash
source venv/bin/activate
pip install -r requirements.txt
```

### Windows: LightGBM / `lib_lightgbm.dll`

If you see **Could not find module** … `lib_lightgbm.dll` **(or one of its dependencies)**:

- The DLL from `pip install lightgbm` is often present, but a **system dependency** is missing—commonly **OpenMP** via **`vcomp140.dll`**, which ships with the **Microsoft Visual C++ Redistributable (x64)**.
- **Install (pick one):**
  - Download and run: [VC_redist.x64.exe](https://aka.ms/vs/17/release/vc_redist.x64.exe) ([Microsoft docs](https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist)).
  - Or in PowerShell (may prompt for admin):  
    `winget install Microsoft.VCRedist.2015+.x64 --source winget`
- Restart the terminal, activate `venv` again, then start the API (below).

Optional: use **Conda** (`conda install -c conda-forge lightgbm`) if pip + VC redist still misbehaves on your machine.

Your raster **data paths** are unrelated to this error—the failure happens when Python first loads the LightGBM native library during `joblib.load` of the pickled model.

## Run the server

From the project root:

```powershell
uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

Then open **http://127.0.0.1:8000** for the UI, or **http://127.0.0.1:8000/docs** for OpenAPI (“Swagger”) documentation.

### Default folders (environment variables)

| Variable | Default |
|----------|---------|
| `PHYTO_MODELS_DIR` | `<project>/models` |
| `PHYTO_WORLDCLIM_DIR` | `<project>/data/worldclim` |
| `PHYTO_ELEVATION_DIR` | `<project>/data/elevation` |

You can override these per request via the **`paths`** object in POST bodies (see `/docs`). The web form submits the folders you enter in the inputs.

### API endpoints (summary)

- `GET /` — Browser UI  
- `GET /api/meta` — Cities list, species list, defaults, BIO raster count, model-load status  
- `GET /api/cities` — City → lat/lon  
- `POST /api/raster-status` — Count BIO rasters given optional path overrides  
- `POST /api/predict` — Single suitability prediction  
- `POST /api/compare` — Table + chart payload for one city × all species  

## Model artifacts

Place these files under your model directory (default: `models/`):

| File | Role |
|------|------|
| `phyto_lightgbm_model.pkl` | Trained `LGBMClassifier` |
| `species_encoder.pkl` | Sklearn-style encoder with `classes_` (e.g. `LabelEncoder`) |
| `imputer.pkl` | Fitted `SimpleImputer` (or compatible) |
| `feature_names.pkl` | `list[str]` — exact column order for prediction |

Engineered features are computed when their names appear in `feature_names.pkl`; base rasters supply BIO1–BIO19 and elevation, and the selected species is encoded as `species_id` when that column is listed.

## Data layout

- **WorldClim:** one GeoTIFF per BIO variable. Names like `wc2.1_30s_bio_1.tif` or `BIO12.tif` are detected automatically under the chosen folder (including subfolders).
- **Elevation:** a file whose stem is `elevation` (e.g. `elevation.tif`) under the elevation folder.

## Project layout

```
PhytoLB/
├── main.py                # FastAPI app + routes
├── config.py              # Env-based default paths
├── requirements.txt
├── templates/
│   └── index.html         # Web UI
├── static/
│   └── style.css
├── models/                # Trained artifacts (you provide)
├── data/
│   ├── worldclim/         # BIO1–BIO19 GeoTIFFs (you provide)
│   └── elevation/         # elevation.tif (you provide)
└── utils/
    ├── bundle_cache.py    # Cached joblib bundle per models_dir
    ├── constants.py       # Lebanese city presets
    ├── model_loader.py    # joblib loads
    ├── windows_lightgbm.py # Windows DLL path + import check
    ├── environment.py     # raster discovery + sampling
    └── prediction.py      # features, imputer, predict_proba, labels
```

## License

Add your license here if applicable.
