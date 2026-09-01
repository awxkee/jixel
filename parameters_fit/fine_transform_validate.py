"""Holdout validation of the fine-transform (IDENTITY/DCT2X2) fit.

Runs the fitted env (work/fine_transform.best.json, or --env-* overrides)
against the unset-env baseline on corpora the Optuna study never saw:
all 24 Kodak photos, the JPEG XL ClassD screen-content set, and a few
synthetic images. Prints per-image BD-SS2-rate (negative = fitted config
wins) plus mean / median / worst.

Usage:
    python -m parameters_fit.fine_transform_validate [--workers 8]
    python -m parameters_fit.fine_transform_validate --fine-off   # ablation
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

from parameters_fit.fine_transform_optuna import (
    DEFAULTS, ROOT, WORK, per_image_bd, run_point, trial_env,
)
from concurrent.futures import ThreadPoolExecutor

IMAGES = (
    sorted((ROOT / "assets/Kodak").glob("*.png"))
    + sorted((ROOT / "assets/jpeg_xl_png").glob("ClassD_*.png"))
    + [
        ROOT / "assets/ClassE_set70.png",
        ROOT / "assets/abstract_small.png",
        ROOT / "assets/buddhabrot1_small.png",
    ]
)
DISTANCES = [0.75, 1.0, 1.5, 2.0, 3.0, 5.0]
CACHE = WORK / "fine_transform_validate_baseline.json"


def run_grid(images, env_extra, workers):
    jobs = [(img, d) for img in images for d in DISTANCES]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = [pool.submit(run_point, img, d, env_extra) for img, d in jobs]
        return [f.result() for f in futs]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--rebuild-baseline", action="store_true")
    ap.add_argument("--fine-off", action="store_true",
                    help="measure the value of the fine transforms themselves "
                         "(max_d=0) instead of the fitted config")
    ap.add_argument("--best", type=Path,
                    default=WORK / "fine_transform.best.json")
    args = ap.parse_args()

    if args.rebuild_baseline or not CACHE.exists():
        print(f"building baseline over {len(IMAGES)} images ...")
        base = run_grid(IMAGES, {}, args.workers)
        CACHE.write_text(json.dumps(base, indent=1))
    base = json.loads(CACHE.read_text())

    if args.fine_off:
        env = {"JIXEL_FINE_SEL": "1.0427543:0.95:0.4:0.98:0"}
        label = "fine transforms OFF"
    else:
        params = {**DEFAULTS, **json.loads(args.best.read_text())}
        env = trial_env(params)
        label = f"fitted config from {args.best.name}"
    print(f"validating: {label}")
    for k, v in env.items():
        print(f"  {k}={v}")

    test = run_grid(IMAGES, env, args.workers)
    bd = per_image_bd(base, test)
    vals = [v for v in bd.values() if not math.isnan(v)]
    for img, v in sorted(bd.items(), key=lambda kv: kv[1]):
        print(f"  {v:+7.3f}%  {img}")
    print(f"mean {statistics.mean(vals):+.3f}%  "
          f"median {statistics.median(vals):+.3f}%  "
          f"worst {max(vals):+.3f}%  wins {sum(v < 0 for v in vals)}/{len(vals)}")


if __name__ == "__main__":
    main()
