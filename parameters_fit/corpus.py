"""Corpus preparation: one aligned square crop per source image, with a
deterministic train/holdout split.

Crops are taken from the image *center* and aligned down to a multiple of 256
(the codec's largest group size) so no partial-group padding differs between
train and validation. Holdout images never contribute a training crop — the
split is by *source image*, not by crop, so there is no content leakage.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from . import config


@dataclass(frozen=True)
class Case:
    """A single (crop, distance) evaluation unit."""

    name: str          # source image stem
    crop: Path         # prepared PNG crop
    width: int
    height: int
    distance: float
    holdout: bool

    @property
    def pixels(self) -> int:
        return self.width * self.height

    def key(self) -> str:
        return f"{self.name}@{self.distance:g}"


def _aligned(v: int, mult: int = 256) -> int:
    return (v // mult) * mult


def _sample_evenly(items: list, count: int | None) -> list:
    """Pick ``count`` items spread evenly across a sorted list (deterministic)."""
    if count is None or count >= len(items):
        return items
    if count <= 0:
        return []
    step = len(items) / count
    return [items[int(i * step)] for i in range(count)]


def _collect_sources() -> list[Path]:
    """Gather source image paths across all configured directories."""
    sources: list[Path] = []
    for directory, cap in config.SOURCE_DIRS:
        if not directory.is_dir():
            continue
        imgs = sorted(
            p for p in directory.iterdir()
            if p.suffix.lower() in {".png", ".jpg", ".jpeg"}
        )
        sources.extend(_sample_evenly(imgs, cap))
    if not sources:
        raise FileNotFoundError(
            f"no source images in {[str(d) for d, _ in config.SOURCE_DIRS]}"
        )
    return sources


def _crop_name(src: Path) -> str:
    """Unique, stable crop name: parent directory + stem (avoids collisions
    across source dirs)."""
    return f"{src.parent.name}__{src.stem}"


def prepare_crops() -> list[tuple[str, Path, bool]]:
    """Create/refresh one crop per source image. Returns
    ``(name, crop_path, is_holdout)`` for every prepared crop."""
    config.ensure_dirs()
    sources = _collect_sources()

    prepared: list[tuple[str, Path, bool]] = []
    for idx, src in enumerate(sources):
        holdout = (idx % config.HOLDOUT_EVERY) == (config.HOLDOUT_EVERY - 1)
        name = _crop_name(src)
        out = config.CROP_DIR / f"{name}.png"
        if not out.exists():
            with Image.open(src) as im:
                im = im.convert("RGB")
                w, h = im.size
                side = min(config.CROP_SIZE, _aligned(min(w, h)))
                if side < 256:
                    # Too small to yield an aligned crop; skip.
                    continue
                left = _aligned((w - side) // 2)
                top = _aligned((h - side) // 2)
                im.crop((left, top, left + side, top + side)).save(out)
        prepared.append((name, out, holdout))
    return prepared


def build_cases(distances: list[float], include_holdout: bool = False) -> list[Case]:
    """Expand prepared crops into (crop x distance) cases."""
    cases: list[Case] = []
    for name, crop, holdout in prepare_crops():
        if holdout and not include_holdout:
            continue
        with Image.open(crop) as im:
            w, h = im.size
        for d in distances:
            cases.append(Case(name, crop, w, h, d, holdout))
    return cases


def train_cases(distances: list[float]) -> list[Case]:
    return build_cases(distances, include_holdout=False)


def holdout_cases(distances: list[float]) -> list[Case]:
    return [c for c in build_cases(distances, include_holdout=True) if c.holdout]
