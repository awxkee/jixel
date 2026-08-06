"""Short Pareto study for the restored DCT64 selector.

The three objectives are all oriented so that larger is better:

* matched-rate SSIMULACRA2 delta;
* matched-rate Butteraugli reduction;
* sampled BD-rate saving at equal SSIMULACRA2.

Run from the repository root::

    python -m parameters_fit.dct64_multi --timeout 400
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import optuna

from . import baseline, config
from .corpus import train_cases
from .encoder import run_case
from .params import DEFAULTS


def _robust(values: list[float]) -> float:
    """Reward the mean while charging for broad and worst regressions."""
    a = np.asarray(values, dtype=np.float64)
    return float(
        np.mean(a)
        + 1.5 * min(float(np.percentile(a, 10)), 0.0)
        + 0.25 * min(float(np.min(a)), 0.0)
    )


def _objective(cases, parameter: str):
    def objective(trial: optuna.Trial) -> tuple[float, float, float]:
        bounds = {
            "dct64_accept": (0.70, 1.20),
            "dct64_rect_accept": (0.30, 1.00),
        }
        candidate = dict(DEFAULTS)
        if parameter == "dct64_rect_qm":
            candidate["dct64_rect_qm_scale"] = trial.suggest_float(
                "dct64_rect_qm_scale", 0.65, 1.45
            )
            candidate["dct64_rect_y_hf"] = trial.suggest_float(
                "dct64_rect_y_hf", 0.60, 1.80
            )
        else:
            candidate[parameter] = trial.suggest_float(parameter, *bounds[parameter])
        ss2_deltas: list[float] = []
        ba_deltas: list[float] = []
        bd_savings: list[float] = []
        rows: list[dict] = []

        for case in cases:
            measurement = run_case(case, candidate)
            curve = baseline.get(case.name)
            ss2_delta = measurement.ss2 - curve.ss2_at_rate(measurement.bpp)
            ba_delta = curve.ba_at_rate(measurement.bpp) - measurement.ba
            baseline_bpp = curve.rate_at_ss2(measurement.ss2)
            bd_saving = 100.0 * (1.0 - measurement.bpp / baseline_bpp)
            ss2_deltas.append(ss2_delta)
            ba_deltas.append(ba_delta)
            bd_savings.append(bd_saving)
            rows.append(
                {
                    "name": case.name,
                    "d": case.distance,
                    "ss2_delta": round(ss2_delta, 5),
                    "ba_delta": round(ba_delta, 6),
                    "bd_saving_pct": round(bd_saving, 5),
                }
            )

        for name, values in (
            ("ss2", ss2_deltas),
            ("ba", ba_deltas),
            ("bd", bd_savings),
        ):
            trial.set_user_attr(f"{name}_mean", float(np.mean(values)))
            trial.set_user_attr(f"{name}_p10", float(np.percentile(values, 10)))
            trial.set_user_attr(f"{name}_worst", float(np.min(values)))
        trial.set_user_attr("per_case", rows)
        return _robust(ss2_deltas), _robust(ba_deltas), _robust(bd_savings)

    return objective


def _report(study: optuna.Study, parameter: str) -> None:
    complete = [
        trial
        for trial in study.trials
        if trial.state == optuna.trial.TrialState.COMPLETE
    ]
    print(f"\ncompleted {len(complete)}/{len(study.trials)} trials")
    print("Pareto front (robust SS2, BA reduction, BD-rate saving):")
    payload = []
    sort_key = (
        (lambda trial: tuple(trial.params.values()))
        if parameter == "dct64_rect_qm"
        else (lambda trial: (trial.params[parameter],))
    )
    for trial in sorted(study.best_trials, key=sort_key):
        attrs = trial.user_attrs
        values = trial.values or [float("nan")] * 3
        print(
            f"  params={trial.params} "
            f"robust=({values[0]:+.5f}, {values[1]:+.6f}, {values[2]:+.4f}%) "
            f"means=({attrs['ss2_mean']:+.5f}, {attrs['ba_mean']:+.6f}, "
            f"{attrs['bd_mean']:+.4f}%)"
        )
        payload.append(
            {
                "trial": trial.number,
                "params": trial.params,
                "robust": {
                    "ss2": values[0],
                    "butteraugli_reduction": values[1],
                    "bd_rate_saving_pct": values[2],
                },
                "summary": attrs,
            }
        )

    out = config.STUDY_DIR / f"{study.study_name}.pareto.json"
    out.write_text(json.dumps(payload, indent=2))
    print(f"written -> {out}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--study", default="dct64_multi_7m_20260803")
    parser.add_argument("--timeout", type=float, default=400.0)
    parser.add_argument("--trials", type=int, default=1000)
    parser.add_argument(
        "--parameter",
        choices=("dct64_accept", "dct64_rect_accept", "dct64_rect_qm"),
        default="dct64_accept",
    )
    args = parser.parse_args(argv)

    config.ensure_dirs()
    storage_path = config.STUDY_DIR / f"{args.study}.sqlite3"
    sampler = optuna.samplers.TPESampler(seed=19, n_startup_trials=8)
    study = optuna.create_study(
        study_name=args.study,
        directions=["maximize", "maximize", "maximize"],
        sampler=sampler,
        storage=f"sqlite:///{storage_path}",
        load_if_exists=True,
    )
    if args.parameter == "dct64_rect_qm":
        seeds = [
            {"dct64_rect_qm_scale": 1.0, "dct64_rect_y_hf": 1.0},
            {"dct64_rect_qm_scale": 0.85, "dct64_rect_y_hf": 1.0},
            {"dct64_rect_qm_scale": 1.15, "dct64_rect_y_hf": 1.0},
            {"dct64_rect_qm_scale": 1.0, "dct64_rect_y_hf": 0.80},
            {"dct64_rect_qm_scale": 1.0, "dct64_rect_y_hf": 1.30},
        ]
    else:
        values = (
            (0.84, 0.919205, 0.951763)
            if args.parameter == "dct64_accept"
            else (0.30, 0.50, 0.70, 0.85, 0.945)
        )
        seeds = [{args.parameter: value} for value in values]
    for seed in seeds:
        study.enqueue_trial(seed, skip_if_exists=True)

    cases = sorted(train_cases([3.0, 4.0, 5.0]), key=lambda c: (c.distance, c.name))
    print(
        f"study '{args.study}': {len(cases)} cases/trial, "
        f"timeout={args.timeout:g}s"
    )
    study.optimize(
        _objective(cases, args.parameter),
        n_trials=args.trials,
        timeout=args.timeout,
    )
    _report(study, args.parameter)


if __name__ == "__main__":
    main()
