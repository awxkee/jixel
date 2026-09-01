"""Joint Optuna fit of the IDENTITY/DCT2X2 (fine transform) cycle.

Tunes the JIXEL_FINE_SEL selection constants (entropy biases, the libjxl
favor multiplier, the reconstruction admission margin, the distance gate)
jointly with the quant tables via JIXEL_ID_QM / JIXEL_DCT2_QM, parameterized
as per-channel scales + shape knobs on the spec tables. Custom tables are
signalled in the frame header, so their cost is part of the measurement.

Baseline = env unset (spec tables + libjxl biases, the shipped state).
Corpus = images where the fine transforms measurably fire (screenshot,
illustration, fractal, digital art, photos as guards).

Objective: maximize mean per-image -BD-SS2-rate over 4-point curves, with a
penalty on the worst image so no class regresses.

Usage:
    python -m parameters_fit.fine_transform_optuna --trials 250 --workers 8
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
    ROOT / "assets/Screenshot 2026-07-18 at 16.09.09.png",  # screen content
    ROOT / "assets/small_carrot.png",                       # illustration
    ROOT / "assets/Burning_Ship_Fractal_small.png",         # fractal
    ROOT / "assets/digital_art_portrait_small.png",         # digital art
    ROOT / "assets/train0/00004_TE_1808x1352.png",          # photo, fine-active
    ROOT / "assets/Kodak/05.png",                           # photo guard
]
DISTANCES = [0.75, 1.5, 3.0, 5.0]
BASELINE_CACHE = WORK / "fine_transform_baseline.json"
LOG = WORK / "fine_transform_progress.log"
DB = WORK / "fine_transform.db"

# Spec tables (must match quant_weights.rs).
ID_SPEC = [[280.0, 3160.0, 3160.0], [60.0, 864.0, 864.0], [18.0, 200.0, 200.0]]
DCT2_SPEC = [
    [3840.0, 2560.0, 1280.0, 640.0, 480.0, 300.0],
    [960.0, 640.0, 320.0, 180.0, 140.0, 120.0],
    [640.0, 320.0, 128.0, 64.0, 32.0, 16.0],
]
CH = ["x", "y", "b"]


def run_point(image: Path, distance: float, env_extra: dict[str, str]) -> dict:
    with tempfile.TemporaryDirectory(prefix="jixel-fine-") as tmp:
        tmpp = Path(tmp)
        jxl = tmpp / "out.jxl"
        png = tmpp / "out.png"
        env = {k: v for k, v in os.environ.items() if not k.startswith("JIXEL_")}
        env.update(env_extra)
        enc = subprocess.run(
            [str(ENCODER), str(image), str(jxl), "-d", str(distance),
             "--threads", "2", "--speed", "slow"],
            env=env, text=True, capture_output=True, check=True,
        )
        info = json.loads(enc.stdout.strip().splitlines()[-1])
        subprocess.run(["djxl", str(jxl), str(png)], capture_output=True, check=True)
        ss2 = float(subprocess.run(
            ["ssimulacra2", str(image), str(png)],
            text=True, capture_output=True, check=True,
        ).stdout.split()[0])
        return {"image": image.stem, "distance": distance,
                "bpp": float(info["bpp"]), "ss2": ss2}


def run_grid(env_extra: dict[str, str], workers: int) -> list[dict]:
    jobs = [(img, d) for img in IMAGES for d in DISTANCES]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = [pool.submit(run_point, img, d, env_extra) for img, d in jobs]
        return [f.result() for f in futs]


def bd_rate(base: list[tuple[float, float]], test: list[tuple[float, float]]) -> float:
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


def per_image_bd(base_pts: list[dict], test_pts: list[dict]) -> dict[str, float]:
    out: dict[str, float] = {}
    for img in sorted({p["image"] for p in base_pts}):
        b = [(p["ss2"], math.log(p["bpp"])) for p in base_pts if p["image"] == img]
        t = [(p["ss2"], math.log(p["bpp"])) for p in test_pts if p["image"] == img]
        out[img] = bd_rate(b, t)
    return out


def trial_env(params: dict) -> dict[str, str]:
    sel = ":".join(
        f"{params[k]:.6g}"
        for k in ("id_bias", "dct2_bias", "favor_mul", "recon_margin", "max_d")
    )
    id_vals = []
    for c in range(3):
        s = params[f"id_scale_{CH[c]}"]
        edge = params["id_edge"]
        id_vals += [ID_SPEC[c][0] * s, ID_SPEC[c][1] * s * edge, ID_SPEC[c][2] * s * edge]
    d2_vals = []
    for c in range(3):
        s = params[f"d2_scale_{CH[c]}"]
        tilt = params["d2_tilt"]
        for i in range(6):
            d2_vals.append(DCT2_SPEC[c][i] * s * tilt ** ((i - 2.5) / 2.5))
    return {
        "JIXEL_FINE_SEL": sel,
        "JIXEL_ID_QM": ",".join(f"{v:.6g}" for v in id_vals),
        "JIXEL_DCT2_QM": ",".join(f"{v:.6g}" for v in d2_vals),
    }


DEFAULTS = {
    "id_bias": 1.0427543, "dct2_bias": 0.95, "favor_mul": 0.4,
    "recon_margin": 0.98, "max_d": 5.0,
    "id_scale_x": 1.0, "id_scale_y": 1.0, "id_scale_b": 1.0, "id_edge": 1.0,
    "d2_scale_x": 1.0, "d2_scale_y": 1.0, "d2_scale_b": 1.0, "d2_tilt": 1.0,
}


def suggest(trial: optuna.Trial) -> dict:
    p = {
        "id_bias": trial.suggest_float("id_bias", 0.6, 1.6),
        "dct2_bias": trial.suggest_float("dct2_bias", 0.5, 1.5),
        "favor_mul": trial.suggest_float("favor_mul", 0.0, 1.0),
        "recon_margin": trial.suggest_float("recon_margin", 0.90, 1.04),
        "max_d": trial.suggest_float("max_d", 1.0, 5.5),
    }
    for k in ("id_scale_x", "id_scale_y", "id_scale_b",
              "d2_scale_x", "d2_scale_y", "d2_scale_b"):
        p[k] = trial.suggest_float(k, 0.4, 2.5, log=True)
    p["id_edge"] = trial.suggest_float("id_edge", 0.5, 2.0, log=True)
    p["d2_tilt"] = trial.suggest_float("d2_tilt", 0.6, 1.6, log=True)
    return p


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=250)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--rebuild-baseline", action="store_true")
    args = ap.parse_args()

    WORK.mkdir(exist_ok=True)
    if args.rebuild_baseline or not BASELINE_CACHE.exists():
        print("building baseline (env unset)...")
        base = run_grid({}, args.workers)
        BASELINE_CACHE.write_text(json.dumps(base, indent=1))
    base = json.loads(BASELINE_CACHE.read_text())

    def objective(trial: optuna.Trial) -> float:
        params = suggest(trial)
        pts = run_grid(trial_env(params), args.workers)
        bd = per_image_bd(base, pts)
        vals = [v for v in bd.values() if not math.isnan(v)]
        if len(vals) != len(IMAGES):
            raise optuna.TrialPruned()
        worst = max(vals)
        score = -sum(vals) / len(vals) - 2.0 * max(worst - 0.5, 0.0)
        with LOG.open("a") as f:
            f.write(f"trial {trial.number}: score={score:+.4f} "
                    f"mean_bd={sum(vals)/len(vals):+.3f} worst={worst:+.3f} "
                    f"bd={json.dumps({k: round(v, 3) for k, v in bd.items()})} "
                    f"params={json.dumps({k: round(v, 5) for k, v in params.items()})}\n")
        return score

    study = optuna.create_study(
        study_name="fine_transform_v1",
        storage=f"sqlite:///{DB}",
        direction="maximize",
        load_if_exists=True,
    )
    if not study.trials:
        study.enqueue_trial(DEFAULTS)  # anchor: must score ~0
        # A few structured starts: coarser tables (rate-saving direction) and
        # stronger/weaker admission.
        study.enqueue_trial({**DEFAULTS, "d2_scale_x": 0.7, "d2_scale_y": 0.7,
                             "d2_scale_b": 0.7, "id_scale_x": 0.7,
                             "id_scale_y": 0.7, "id_scale_b": 0.7})
        study.enqueue_trial({**DEFAULTS, "recon_margin": 1.02, "favor_mul": 0.6})
        study.enqueue_trial({**DEFAULTS, "recon_margin": 0.94, "favor_mul": 0.2})
    study.optimize(objective, n_trials=args.trials)

    best = study.best_trial
    print(f"best trial #{best.number}: score={best.value:+.4f}")
    print(json.dumps(best.params, indent=1))
    (WORK / "fine_transform.best.json").write_text(json.dumps(best.params, indent=1))
    print("env:", json.dumps(trial_env({**DEFAULTS, **best.params}), indent=1))


if __name__ == "__main__":
    main()
