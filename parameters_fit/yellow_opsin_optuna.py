"""Optuna fit of the adaptive yellow-opsin selector (JIXEL_YELLOW_OPSIN=auto).

NOTE 2026-08-29: the two-tier selector shipped unconditionally and the
JIXEL_YELLOW_OPSIN / JIXEL_YELLOW_FIT hooks were removed from the encoder
(constants live in yellow_opsin.rs). To re-tune, temporarily restore the env
overrides there first — and use the tail-chroma-loss metric (p95 / frac>20
over the yellow mask), not SS2: it is the one that matches visible
desaturation.

Tunes the JIXEL_YELLOW_FIT knobs
(rel_cost_ratio:min_spec_cost:regression_weight:bias_mid:bias_hi:tail_weight)
so the auto mode fires only where a biased B row pays. Baseline is the shipped
default (env unset) — auto-no-fire is byte-identical to it, so a trial's score
is exactly the net effect of its firing decisions.

Objective per case (yellow images + fractal), following the b8-v2 lesson that
chroma wins must be taxed with an all-content bit term:

    case_score = d_chroma_pp - 1.0 * d_bits_pct - 2.0 * max(-d_ss2 - 0.25, 0)

where d_chroma_pp = reduction of risk-weighted mean |relative yellow-chroma
error| in percentage points (positive = trial better), d_bits_pct = byte
inflation %, and the last term caps per-case SS2 damage. Neutral Kodak guards
enter with the same formula (their chroma term is ~0, so any spurious firing
just pays its bit/SS2 cost). Score = mean over cases.

Usage:
    python -m parameters_fit.yellow_opsin_optuna --timeout 600 --workers 8
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import optuna
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
ENCODER = ROOT / "target/release/jixel-tuner"
WORK = ROOT / "parameters_fit/work"
CORPUS = WORK / "yellow_fit"

YELLOW_IMAGES = [
    CORPUS / "fire_fractal.png",
    CORPUS / "pexels-huznimhmd-956769.png",
    CORPUS / "pexels-jedrzej-koralewski-14125120-17502287.png",
    CORPUS / "pexels-content-prod-co-7067195.png",
    CORPUS / "pexels-dkeats-33624937.png",
]
GUARD_IMAGES = [
    ROOT / "assets/Kodak/08.png",
    ROOT / "assets/Kodak/23.png",
]
DISTANCES = [1.0, 2.0, 3.0]

BASELINE_CACHE = WORK / "yellow_opsin_baseline.json"
LOG = WORK / "yellow_opsin_progress.log"

SHIPPED = {
    "rel_cost_ratio": 0.55,
    "min_spec_cost": 0.02,
    "regression_weight": 4.0,
    "bias_mid": 0.70,
    "bias_hi": 0.85,
    "tail_weight": 0.4,
}


def env_string(p: dict[str, float]) -> str:
    return (
        f"{p['rel_cost_ratio']}:{p['min_spec_cost']}:{p['regression_weight']}:"
        f"{p['bias_mid']}:{p['bias_hi']}:{p['tail_weight']}"
    )


def srgb_to_linear(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float64) / 255.0
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def risk(lin: np.ndarray) -> np.ndarray:
    r, g, b = lin[..., 0], lin[..., 1], lin[..., 2]
    rg_hi = np.maximum(r, g)
    rg_lo = np.minimum(r, g)
    yellow = np.maximum(rg_lo - b, 0.0)
    inv_hi = 1.0 / np.maximum(rg_hi, 1e-5)

    def ramp(x, lo, hi):
        return np.clip((x - lo) / (hi - lo), 0.0, 1.0)

    return (
        ramp(rg_hi, 0.55, 0.85)
        * ramp(yellow * inv_hi, 0.15, 0.45)
        * ramp(1.0 - np.abs(r - g) * inv_hi, 0.20, 0.80)
    )


_SRC_CACHE: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}


def src_planes(image: Path):
    key = str(image)
    if key not in _SRC_CACHE:
        lin = srgb_to_linear(np.asarray(Image.open(image).convert("RGB")))
        cs = np.maximum(np.minimum(lin[..., 0], lin[..., 1]) - lin[..., 2], 0.0)
        mask = (risk(lin) > 0.05) & (cs > 0.02)
        _SRC_CACHE[key] = (lin, cs, mask)
    return _SRC_CACHE[key]


def chroma_err_pct(image: Path, recon: Path) -> float:
    """Risk-weighted mean |relative yellow-chroma error| in percent."""
    _, cs, mask = src_planes(image)
    if mask.sum() == 0:
        return 0.0
    rec = srgb_to_linear(np.asarray(Image.open(recon).convert("RGB")))
    cr = np.maximum(np.minimum(rec[..., 0], rec[..., 1]) - rec[..., 2], 0.0)
    rel = np.abs(cr[mask] - cs[mask]) / np.maximum(cs[mask], 0.02)
    return float(rel.mean()) * 100.0


def run_point(image: Path, distance: float, env_extra: dict[str, str]) -> dict:
    with tempfile.TemporaryDirectory(prefix="jixel-yopsin-") as tmp:
        tmpp = Path(tmp)
        jxl = tmpp / "out.jxl"
        png = tmpp / "out.png"
        env = {k: v for k, v in os.environ.items() if not k.startswith("JIXEL_")}
        env.update(env_extra)
        subprocess.run(
            [str(ENCODER), str(image), str(jxl), "-d", str(distance),
             "--threads", "2", "--speed", "slow"],
            env=env, text=True, capture_output=True, check=True,
        )
        nbytes = jxl.stat().st_size
        subprocess.run(["djxl", str(jxl), str(png)], capture_output=True, check=True)
        ss2 = float(subprocess.run(
            ["ssimulacra2", str(image), str(png)],
            text=True, capture_output=True, check=True,
        ).stdout.split()[0])
        return {
            "image": image.stem,
            "distance": distance,
            "bytes": nbytes,
            "ss2": ss2,
            "chroma_err": chroma_err_pct(image, png),
        }


def run_grid(env_extra: dict[str, str], workers: int) -> list[dict]:
    jobs = [(img, d) for img in YELLOW_IMAGES + GUARD_IMAGES for d in DISTANCES]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = [pool.submit(run_point, img, d, env_extra) for img, d in jobs]
        return [f.result() for f in futs]


def score_of(base_pts: list[dict], test_pts: list[dict]) -> tuple[float, dict]:
    base = {(p["image"], p["distance"]): p for p in base_pts}
    scores = []
    fired = 0
    detail = {}
    for t in test_pts:
        b = base[(t["image"], t["distance"])]
        d_bits = (t["bytes"] - b["bytes"]) / b["bytes"] * 100.0
        d_chroma = b["chroma_err"] - t["chroma_err"]
        d_ss2 = t["ss2"] - b["ss2"]
        s = d_chroma - 1.0 * d_bits - 2.0 * max(-d_ss2 - 0.25, 0.0)
        if t["bytes"] != b["bytes"]:
            fired += 1
            detail[f"{t['image']}@{t['distance']:g}"] = (
                f"dC={d_chroma:+.2f}pp dbits={d_bits:+.2f}% dss2={d_ss2:+.2f} s={s:+.2f}"
            )
        scores.append(s)
    return sum(scores) / len(scores), {"fired": fired, **detail}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--trials", type=int, default=None)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    if BASELINE_CACHE.exists():
        base_pts = json.loads(BASELINE_CACHE.read_text())
    else:
        print("building baseline grid (yellow opsin off)...", flush=True)
        base_pts = run_grid({}, args.workers)
        BASELINE_CACHE.write_text(json.dumps(base_pts))

    storage = f"sqlite:///{WORK}/yellow_opsin_optuna.sqlite3"
    study = optuna.create_study(
        study_name="yellow_opsin", storage=storage,
        direction="maximize", load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=7, multivariate=True),
    )
    if not any(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials):
        study.enqueue_trial(SHIPPED)

    def objective(trial: optuna.Trial) -> float:
        p = {
            "rel_cost_ratio": trial.suggest_float("rel_cost_ratio", 0.3, 1.0),
            "min_spec_cost": trial.suggest_float("min_spec_cost", 0.0, 0.04),
            "regression_weight": trial.suggest_float("regression_weight", 0.0, 10.0),
            "bias_mid": trial.suggest_float("bias_mid", 0.60, 0.75),
            "bias_hi": trial.suggest_float("bias_hi", 0.75, 0.88),
            "tail_weight": trial.suggest_float("tail_weight", 0.0, 0.8),
        }
        pts = run_grid(
            {"JIXEL_YELLOW_OPSIN": "auto", "JIXEL_YELLOW_FIT": env_string(p)},
            args.workers,
        )
        s, detail = score_of(base_pts, pts)
        trial.set_user_attr("fired", detail["fired"])
        with LOG.open("a") as f:
            f.write(f"trial {trial.number}: {env_string(p)} score={s:+.4f} {detail}\n")
        return s

    study.optimize(objective, timeout=args.timeout, n_trials=args.trials)

    best = study.best_trial
    print(f"\nbest trial {best.number}: score={best.value:+.4f}")
    print(f"JIXEL_YELLOW_FIT={env_string(best.params)}")
    print(f"fired on {best.user_attrs.get('fired', '?')} cases")


if __name__ == "__main__":
    main()
