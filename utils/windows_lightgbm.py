"""Windows: help LightGBM load ``lib_lightgbm.dll`` and surface clear fix hints."""

from __future__ import annotations

import sys
from pathlib import Path


def _site_packages_dirs() -> list[Path]:
    roots: list[Path] = []
    try:
        import site

        roots.extend(Path(p) for p in site.getsitepackages())
        user = site.getusersitepackages()
        if user:
            roots.append(Path(user))
    except Exception:
        pass
    roots.append(Path(sys.prefix) / "Lib" / "site-packages")

    seen: set[str] = set()
    out: list[Path] = []
    for r in roots:
        try:
            key = str(r.resolve())
        except OSError:
            continue
        if key not in seen and r.is_dir():
            seen.add(key)
            out.append(r)
    return out


def register_lightgbm_dll_directory() -> None:
    """Add LightGBM's ``bin`` folder to the DLL search path (Python 3.8+ on Windows)."""
    if sys.platform != "win32":
        return
    try:
        import os

        for base in _site_packages_dirs():
            bin_dir = (base / "lightgbm" / "bin").resolve()
            if bin_dir.is_dir():
                os.add_dll_directory(str(bin_dir))
    except Exception:
        pass


LIGHTGBM_WINDOWS_HINT = (
    "LightGBM could not load its native library on Windows. "
    "The usual cause is a missing Visual C++ runtime (often ``vcomp140.dll`` / OpenMP).\n\n"
    "**Fix:** install **Microsoft Visual C++ Redistributable (x64)**:\n"
    "- Download: https://aka.ms/vs/17/release/vc_redist.x64.exe\n"
    "- Or run (elevated PowerShell): "
    "`winget install Microsoft.VCRedist.2015+.x64 --source winget`\n\n"
    "Then restart the terminal and run `streamlit run app.py` again."
)


def ensure_lightgbm_importable() -> None:
    """
    Prepare DLL paths and verify ``import lightgbm`` works before unpickling the model.

    Raises RuntimeError with installation hints if the DLL or a dependency is missing.
    """
    register_lightgbm_dll_directory()
    try:
        import lightgbm  # noqa: F401
    except FileNotFoundError as e:
        low = str(e).lower()
        if "lightgbm" in low or "lib_lightgbm" in low:
            raise RuntimeError(LIGHTGBM_WINDOWS_HINT) from e
        raise
    except OSError as e:
        low = str(e).lower()
        if "lightgbm" in low or "lib_lightgbm" in low or "dll" in low:
            raise RuntimeError(LIGHTGBM_WINDOWS_HINT) from e
        raise
