"""Optuna fit of the CfL RDO constants (JIXEL_CFL_RDO, Slow-only).

NOTE 2026-08-29: the fit shipped and the JIXEL_CFL_RDO hook was removed from
the encoder (constants now live in color_correlation.rs::CFL_RDO). To re-run
this study, temporarily restore the env override in cfl_rdo_fit() first.

Baseline is the pre-RDO closed-form path (JIXEL_CFL_RDO=0), so BD-rates
measure the *total* RDO benefit for each constant set. The corpus is
chroma-stressed on purpose: two yellow-band crops of the burning-ship
fractal (the densest yellow this asset has — red/yellow/green speckle
bands, brutal for CfL slopes), kodim20 (the historical CfL image) and
kodim23 (parrots, saturated chroma).

Joint objective (desaturation shows in butteraugli, not SS2):

    score = mean_img( -bd_ss2 - 0.5*bd_ba3 - 0.15*bd_bamax )
            - 2.0 * max(worst_bd_ss2 - 1.0, 0)     # per-image SS2 damage cap

Fixed at fork values: corr gates (0.20/0.35) and dc_thr_x (0.10 — a no-op
at raw X scale; re-open it if the gate design changes). Tuned: rate/deadzone
weights, energy gates, dc_thr_b, max_d.

Usage:
    python -m parameters_fit.cfl_rdo_optuna --timeout 600 --workers 10
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import optuna

ROOT = Path(__file__).resolve().parents[1]
ENCODER = ROOT / "target/release/jixel-tuner"
WORK = ROOT / "parameters_fit/work"

IMAGES = [
    WORK / "ship_yellow_0.png",
    WORK / "ship_yellow_1.png",
    ROOT / "assets/Kodak/20.png",
    ROOT / "assets/Kodak/23.png",
]
DISTANCES = [0.5, 1.0, 1.5, 2.25, 3.0]
BASELINE_CACHE = WORK / "cfl_rdo_baseline.json"
LOG = WORK / "cfl_rdo_progress.log"

FORK_DEFAULTS = {
    "lambda_bits": 0.005,
    "mag_cost": 0.05,
    "oversat": 1.2,
    "dz_strong": 2.5,
    "dz_mild": 1.8,
    "energy_strong": 0.10,
    "energy_mild": 0.05,
    "dc_thr_b": 0.35,
    "max_d": 2.75,
}


def env_string(p: dict[str, float]) -> str:
    return (
        f"{p['lambda_bits']}:{p['mag_cost']}:{p['oversat']}:"
        f"{p['dz_strong']}:{p['dz_mild']}:0.20:0.35:"
        f"{p['energy_strong']}:{p['energy_mild']}:0.10:{p['dc_thr_b']}:"
        f"{p['max_d']}"
    )


def run_point(image: Path, distance: float, env_extra: dict[str, str]) -> dict:
    with tempfile.TemporaryDirectory(prefix="jixel-cflrdo-") as tmp:
        tmpp = Path(tmp)
        jxl = tmpp / "out.jxl"
        png = tmpp / "out.png"
        env = {k: v for k, v in os.environ.items() if not k.startswith("JIXEL_")}
        env.update(env_extra)
        enc = subprocess.run(
            [str(ENCODER), str(image), str(jxl), "-d", str(distance),
             "--threads", "1", "--speed", "slow"],
            env=env, text=True, capture_output=True, check=True,
        )
        info = json.loads(enc.stdout.strip().splitlines()[-1])
        subprocess.run(["djxl", str(jxl), str(png)], capture_output=True, check=True)
        ss2 = float(subprocess.run(
            ["ssimulacra2", str(image), str(png)],
            text=True, capture_output=True, check=True,
        ).stdout.split()[0])
        ba = subprocess.run(
            ["butteraugli_main", str(image), str(png), "--pnorm", "3"],
            text=True, capture_output=True, check=True,
        ).stdout.splitlines()
        return {
            "image": image.stem, "distance": distance, "bpp": float(info["bpp"]),
            "ss2": ss2, "bamax": float(ba[0]), "ba3": float(ba[1].split(":")[1]),
        }


def run_grid(env_extra: dict[str, str], workers: int) -> list[dict]:
    jobs = [(img, d) for img in IMAGES for d in DISTANCES]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = [pool.submit(run_point, img, d, env_extra) for img, d in jobs]
        return [f.result() for f in futs]


def bd_rate(base: list[tuple[float, float]], test: list[tuple[float, float]]) -> float:
    """BD-rate %% over (quality, log_bpp) points; negative = test cheaper."""
    base, test = sorted(base), sorted(test)
    lo = max(base[0][0], test[0][0])
    hi = min(base[-1][0], test[-1][0])
    if hi <= lo:
        return float("nan")

    def integ(points):
        total = 0.0
        for (q0, r0), (q1, r1) in zip(points, points[1:]):
            a, b = max(q0, lo), min(q1, hi)
            if b <= a or q1 == q0:
                continue
            ra = r0 + (r1 - r0) * (a - q0) / (q1 - q0)
            rb = r0 + (r1 - r0) * (b - q0) / (q1 - q0)
            total += 0.5 * (ra + rb) * (b - a)
        return total / (hi - lo)

    return (math.exp(integ(test) - integ(base)) - 1.0) * 100.0


def per_image_bd(base_pts: list[dict], test_pts: list[dict]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for img in sorted({p["image"] for p in base_pts}):
        b = [p for p in base_pts if p["image"] == img]
        t = [p for p in test_pts if p["image"] == img]
        pts = lambda src, key, neg: [
            ((-p[key] if neg else p[key]), math.log(p["bpp"])) for p in src
        ]
        out[img] = {
            "ss2": bd_rate(pts(b, "ss2", False), pts(t, "ss2", False)),
            "ba3": bd_rate(pts(b, "ba3", True), pts(t, "ba3", True)),
            "bamax": bd_rate(pts(b, "bamax", True), pts(t, "bamax", True)),
        }
    return out


def score_of(bd: dict[str, dict[str, float]]) -> float:
    vals = list(bd.values())
    mean = lambda key: sum(v[key] for v in vals) / len(vals)
    score = -mean("ss2") - 0.5 * mean("ba3") - 0.15 * mean("bamax")
    worst_ss2 = max(v["ss2"] for v in vals)
    return score - 2.0 * max(worst_ss2 - 1.0, 0.0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--trials", type=int, default=None)
    ap.add_argument("--workers", type=int, default=10)
    args = ap.parse_args()

    if BASELINE_CACHE.exists():
        base_pts = json.loads(BASELINE_CACHE.read_text())
    else:
        print("building baseline grid (JIXEL_CFL_RDO=0)...", flush=True)
        base_pts = run_grid({"JIXEL_CFL_RDO": "0"}, args.workers)
        BASELINE_CACHE.write_text(json.dumps(base_pts))

    storage = f"sqlite:///{WORK}/cfl_rdo_optuna.sqlite3"
    study = optuna.create_study(
        study_name="cfl_rdo_joint", storage=storage,
        direction="maximize", load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=7, multivariate=True),
    )

    def objective(trial: optuna.Trial) -> float:
        p = {
            "lambda_bits": trial.suggest_float("lambda_bits", 0.001, 0.05, log=True),
            "mag_cost": trial.suggest_float("mag_cost", 0.005, 0.3, log=True),
            "oversat": trial.suggest_float("oversat", 1.0, 1.6),
            "dz_strong": trial.suggest_float("dz_strong", 1.0, 4.0),
            "dz_mild": trial.suggest_float("dz_mild", 1.0, 2.5),
            "energy_strong": trial.suggest_float("energy_strong", 0.02, 0.4, log=True),
            "energy_mild": trial.suggest_float("energy_mild", 0.01, 0.2, log=True),
            "dc_thr_b": trial.suggest_float("dc_thr_b", 0.1, 0.8),
            "max_d": trial.suggest_float("max_d", 2.0, 4.5),
        }
        pts = run_grid({"JIXEL_CFL_RDO": env_string(p)}, args.workers)
        bd = per_image_bd(base_pts, pts)
        s = score_of(bd)
        mean = lambda key: sum(v[key] for v in bd.values()) / len(bd)
        trial.set_user_attr("bd_ss2", mean("ss2"))
        trial.set_user_attr("bd_ba3", mean("ba3"))
        trial.set_user_attr("bd_bamax", mean("bamax"))
        with LOG.open("a") as f:
            f.write(
                f"trial {trial.number}: {env_string(p)} score={s:.4f} "
                f"ss2={mean('ss2'):+.3f} ba3={mean('ba3'):+.3f} "
                f"bamax={mean('bamax'):+.3f} "
                f"per_img_ss2={ {k: round(v['ss2'], 2) for k, v in bd.items()} }\n"
            )
        return s

    # Seeds: fork defaults, a no-deadzone variant, and a rate-heavy variant.
    study.enqueue_trial(dict(FORK_DEFAULTS))
    study.enqueue_trial({**FORK_DEFAULTS, "dz_strong": 1.0, "dz_mild": 1.0})
    study.enqueue_trial({**FORK_DEFAULTS, "lambda_bits": 0.02, "mag_cost": 0.15})
    study.optimize(objective, n_trials=args.trials, timeout=args.timeout)

    best = study.best_trial
    print("BEST:", json.dumps(best.params), best.value, best.user_attrs)
    print("env:", env_string({**FORK_DEFAULTS, **best.params}))


if __name__ == "__main__":
    main()
