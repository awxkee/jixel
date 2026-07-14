"""Per-crop baseline rate/quality curves.

For each source crop we encode with the *shipped defaults* across a dense
distance sweep, decode, and score. This gives a monotone rate->quality curve per
image. A candidate encode (which lands at some bitrate ``r``) is then compared to
the baseline *at the same rate* by interpolating the baseline score against
``log(bpp)`` — so a parameter set is never rewarded for merely spending more
bits.

Curves are cached to JSON so an overnight study never rebuilds them.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import config
from .corpus import Case, prepare_crops
from .encoder import run_case


def _cache_path(name: str) -> Path:
    return config.CACHE_DIR / f"{name}.json"


def build_baseline_for(name: str, crop: Path, distances: list[float], *, force: bool = False) -> dict:
    """Build (or load) one crop's baseline curve."""
    path = _cache_path(name)
    if path.exists() and not force:
        cached = json.loads(path.read_text())
        if set(cached.get("distance", [])) >= set(distances):
            return cached

    from PIL import Image

    with Image.open(crop) as im:
        w, h = im.size

    rows = []
    for d in sorted(distances):
        case = Case(name, crop, w, h, d, holdout=False)
        m = run_case(case, params=None)  # None => shipped defaults
        rows.append((d, m.bpp, m.ss2, m.encode_ms))

    curve = {
        "name": name,
        "distance": [r[0] for r in rows],
        "bpp": [r[1] for r in rows],
        "ss2": [r[2] for r in rows],
        "encode_ms": [r[3] for r in rows],
    }
    config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(curve, indent=2))
    return curve


def build_all_baselines(distances: list[float], *, force: bool = False, log=print) -> dict[str, dict]:
    """Build baselines for every prepared crop (train + holdout)."""
    curves: dict[str, dict] = {}
    prepared = prepare_crops()
    for i, (name, crop, _holdout) in enumerate(prepared, 1):
        log(f"[baseline {i}/{len(prepared)}] {name} ...")
        curves[name] = build_baseline_for(name, crop, distances, force=force)
    return curves


class Baseline:
    """Cached-curve accessor with log-rate interpolation."""

    def __init__(self, name: str):
        self.name = name
        data = json.loads(_cache_path(name).read_text())
        order = np.argsort(np.asarray(data["bpp"], dtype=np.float64))
        self._bpp = np.asarray(data["bpp"], dtype=np.float64)[order]
        self._ss2 = np.asarray(data["ss2"], dtype=np.float64)[order]
        self._dist = np.asarray(data["distance"], dtype=np.float64)
        self._time = np.asarray(data["encode_ms"], dtype=np.float64)

    def ss2_at_rate(self, bpp: float) -> float:
        """Baseline SSIMULACRA2 at a given bitrate (log-bpp interpolation,
        clamped to the measured range)."""
        lb = np.log(max(bpp, 1e-6))
        return float(np.interp(lb, np.log(self._bpp), self._ss2))

    def time_at_distance(self, distance: float) -> float:
        """Baseline encode time (ms) at (nearest) distance."""
        idx = int(np.argmin(np.abs(self._dist - distance)))
        return float(self._time[idx])


_CACHE: dict[str, Baseline] = {}


def get(name: str) -> Baseline:
    if name not in _CACHE:
        _CACHE[name] = Baseline(name)
    return _CACHE[name]
