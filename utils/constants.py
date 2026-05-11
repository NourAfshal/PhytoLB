"""Shared app constants (Lebanese city presets: lat, lon)."""

from __future__ import annotations

# Keys are display names; values are (latitude, longitude)
LEBANESE_CITIES: dict[str, tuple[float, float]] = {
    "Beirut": (33.8938, 35.5018),
    "Tripoli": (34.4367, 35.8497),
    "Byblos": (34.1230, 35.6519),
    "Bcharre": (34.2508, 36.0106),
    "Cedars of God": (34.2436, 36.0486),
    "Faraya": (33.9944, 35.8172),
    "Zahle": (33.8463, 35.9020),
    "Baalbek": (34.0060, 36.2181),
    "Sidon": (33.5630, 35.3688),
    "Tyre": (33.2700, 35.2038),
}
