"""Subprocess glue: encode (jixel-tuner) -> decode (djxl) -> score
(ssimulacra2), returning a rate/quality/time measurement for one case.

Tuning parameters reach the encoder as environment overrides rendered by
``params.to_env`` (read by the jixel crate at first use). Passing
``params=None`` runs the shipped defaults, which is how baselines are built.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

from . import config
from .corpus import Case
from .params import MANAGED_ENV
from .params import to_env as params_to_env


class EncodeError(RuntimeError):
    """Any failure in the encode/decode/score chain for one case."""


@dataclass(frozen=True)
class Measurement:
    bpp: float
    ss2: float
    encode_ms: float
    bytes: int


def _run(cmd: list[str], env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd, capture_output=True, text=True, env=env, check=False
    )


def run_case(
    case: Case,
    params: dict[str, float | int] | None,
    tmp_dir: Path | None = None,
) -> Measurement:
    """Encode + decode + score one case. Cleans up its scratch files."""
    tmp_dir = tmp_dir or config.TMP_DIR
    tmp_dir.mkdir(parents=True, exist_ok=True)
    token = uuid.uuid4().hex[:12]
    stem = f"{case.name}_{case.distance:g}_{token}"
    jxl = tmp_dir / f"{stem}.jxl"
    png = tmp_dir / f"{stem}.png"

    env = dict(os.environ)
    for var in MANAGED_ENV:
        env.pop(var, None)
    env.update(params_to_env(params))

    # A "boost" key is not an env knob: it is passed straight to the tuner's
    # --boost CLI flag (DarkAqConfig CSV, e.g. "1.5" or "1.5,6,0.02,1").
    boost = (params or {}).get("boost")

    try:
        # 1) Encode.
        cmd = [
            str(config.ENCODER),
            str(case.crop),
            str(jxl),
            "--distance",
            str(case.distance),
            "--threads",
            str(config.ENCODE_THREADS),
            "--speed",
            config.SPEED,
        ]
        if boost is not None:
            cmd += ["--boost", str(boost)]
        enc = _run(cmd, env=env)
        if enc.returncode != 0 or not jxl.exists():
            raise EncodeError(f"encode failed ({case.key()}): {enc.stderr.strip()}")
        info = json.loads(enc.stdout.strip().splitlines()[-1])
        bpp = float(info["bpp"])
        nbytes = int(info["bytes"])
        encode_ms = float(info["encode_ms"])

        # 2) Decode.
        dec = _run([config.DJXL, str(jxl), str(png)])
        if dec.returncode != 0 or not png.exists():
            raise EncodeError(f"decode failed ({case.key()}): {dec.stderr.strip()}")

        # 3) Score.
        met = _run([config.SSIMULACRA2, str(case.crop), str(png)])
        if met.returncode != 0:
            raise EncodeError(f"ssimulacra2 failed ({case.key()}): {met.stderr.strip()}")
        ss2 = float(met.stdout.strip().split()[0])

        return Measurement(bpp=bpp, ss2=ss2, encode_ms=encode_ms, bytes=nbytes)
    finally:
        for f in (jxl, png):
            try:
                f.unlink()
            except FileNotFoundError:
                pass
