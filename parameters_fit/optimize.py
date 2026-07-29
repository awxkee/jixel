"""Optuna study driver and CLI.

Subcommands::

    python -m parameters_fit.optimize baseline           # build/refresh RD curves
    python -m parameters_fit.optimize search [opts]      # run a TPE study
    python -m parameters_fit.optimize best  --study NAME # print best params JSON
    python -m parameters_fit.optimize validate --study NAME  # holdout + wide sweep

The objective encodes every training case with a candidate parameter set,
computes the rate-matched SSIMULACRA2 delta against the cached baseline, and
aggregates with the robust score in ``objective.py``. Cheap high-quality cases
run first so unpromising trials are pruned before the whole corpus is spent.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import optuna

from . import baseline, config, params as P
from .corpus import Case, holdout_cases, train_cases
from .encoder import EncodeError, run_case
from . import objective as obj_cfg
from .objective import refine_score, robust_score

optuna.logging.set_verbosity(optuna.logging.WARNING)


def _storage(study_name: str) -> str:
    config.ensure_dirs()
    return f"sqlite:///{config.STUDY_DIR / (study_name + '.sqlite3')}"


def _ordered(cases: list[Case]) -> list[Case]:
    """Cheapest/most-diagnostic first: low distance (the problematic HQ band)
    before high distance, then by image name for determinism."""
    return sorted(cases, key=lambda c: (c.distance, c.name))


def make_objective(cases: list[Case], active_keys: list[str] | None):
    ordered = _ordered(cases)

    def objective(trial: optuna.Trial) -> float:
        candidate = P.suggest(trial, active_keys)
        deltas: list[float] = []
        time_ratios: list[float] = []
        per_case: list[dict] = []

        for step, case in enumerate(ordered):
            try:
                m = run_case(case, candidate)
            except (EncodeError, ValueError, OSError) as err:
                trial.set_user_attr("failure", str(err))
                raise optuna.TrialPruned() from err

            bl = baseline.get(case.name)
            delta = m.ss2 - bl.ss2_at_rate(m.bpp)
            tratio = m.encode_ms / max(bl.time_at_distance(case.distance), 1e-6)
            deltas.append(delta)
            time_ratios.append(tratio)
            per_case.append(
                {"name": case.name, "d": case.distance, "bpp": round(m.bpp, 5),
                 "ss2": round(m.ss2, 4), "delta": round(delta, 4)}
            )

            if delta < obj_cfg.PRUNE_DELTA:
                trial.set_user_attr("pruned_on", case.key())
                raise optuna.TrialPruned()

            trial.report(robust_score(deltas, time_ratios), step)
            if trial.should_prune():
                raise optuna.TrialPruned()

        trial.set_user_attr("per_case", per_case)
        trial.set_user_attr("mean_delta", sum(deltas) / len(deltas))
        return robust_score(deltas, time_ratios)

    return objective


VAL_EVERY = 4  # every 4th optimization crop is held back as internal validation


def make_refine_objective(cases: list[Case], active_keys: list[str] | None):
    """Refinement objective: clamped deltas, median-based, with an internal
    train/validation split (the reviewer's anti-overfit guard). The held-out
    validation images never influence which config is chosen except through the
    penalty terms."""
    import numpy as np

    names = sorted({c.name for c in cases})
    val_names = {n for i, n in enumerate(names) if i % VAL_EVERY == (VAL_EVERY - 1)}
    opt = _ordered([c for c in cases if c.name not in val_names])
    val = _ordered([c for c in cases if c.name in val_names])
    print(f"  refine split: {len(names) - len(val_names)} opt / {len(val_names)} "
          f"internal-val images")

    def eval_cases(case_list, candidate, trial, tag, deltas, tratios):
        for case in case_list:
            try:
                m = run_case(case, candidate)
            except (EncodeError, ValueError, OSError) as err:
                trial.set_user_attr("failure", f"{tag}:{err}")
                raise optuna.TrialPruned() from err
            bl = baseline.get(case.name)
            deltas.append(m.ss2 - bl.ss2_at_rate(m.bpp))
            tratios.append(m.encode_ms / max(bl.time_at_distance(case.distance), 1e-6))

    def objective(trial: optuna.Trial) -> float:
        candidate = P.suggest(trial, active_keys)
        opt_d: list[float] = []
        val_d: list[float] = []
        tr: list[float] = []
        eval_cases(opt, candidate, trial, "opt", opt_d, tr)
        eval_cases(val, candidate, trial, "val", val_d, tr)
        # Hard reject configs that break any validation image.
        if val_d and min(val_d) < obj_cfg.VAL_REJECT:
            trial.set_user_attr("val_worst", round(min(val_d), 3))
            raise optuna.TrialPruned()
        score = refine_score(opt_d, val_d, tr)
        trial.set_user_attr("opt_median", round(float(np.median(opt_d)), 4))
        trial.set_user_attr("val_mean", round(float(np.mean(val_d)), 4))
        trial.set_user_attr("val_worst", round(float(np.min(val_d)), 4))
        return score

    return objective


def cmd_baseline(args: argparse.Namespace) -> None:
    baseline.build_all_baselines(config.BASELINE_DISTANCES, force=args.force)
    print(f"baselines cached in {config.CACHE_DIR}")


def cmd_search(args: argparse.Namespace) -> None:
    obj_cfg.PRUNE_DELTA = args.prune_delta
    obj_cfg.VAL_REJECT = args.val_reject
    active_keys = args.params.split(",") if args.params else None
    if active_keys:
        for k in active_keys:
            if k not in P.SPACE:
                sys.exit(f"unknown parameter: {k}")

    distances = (
        [float(x) for x in args.distances.split(",")]
        if args.distances else config.SEARCH_DISTANCES
    )

    # Baselines must exist for every training crop.
    cases = train_cases(distances)
    names = {c.name for c in cases}
    missing = [n for n in names if not (config.CACHE_DIR / f"{n}.json").exists()]
    if missing:
        print(f"building missing baselines for {len(missing)} crops ...")
        baseline.build_all_baselines(config.BASELINE_DISTANCES)

    sampler = optuna.samplers.TPESampler(
        seed=args.seed,
        multivariate=True,
        group=True,
        constant_liar=True,
        n_startup_trials=args.startup,
    )
    pruner = optuna.pruners.MedianPruner(n_startup_trials=8, n_warmup_steps=4)
    study = optuna.create_study(
        study_name=args.study,
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        storage=_storage(args.study),
        load_if_exists=True,
    )
    # Always evaluate the shipped defaults (deduped on resume).
    enqueue = {k: P.DEFAULTS[k] for k in (active_keys or P.SPACE.keys())}
    study.enqueue_trial(enqueue, skip_if_exists=True)

    mode = "refine (clamped/median + internal-val)" if args.refine else "standard"
    print(f"study '{args.study}': {len(cases)} cases/trial, {args.trials} trials, "
          f"n_jobs={args.jobs}, objective={mode}")
    obj = (make_refine_objective if args.refine else make_objective)(cases, active_keys)
    study.optimize(obj, n_trials=args.trials, n_jobs=args.jobs, show_progress_bar=False)
    _report(study, refine=args.refine)


def _report(study: optuna.Study, refine: bool = False) -> None:
    done = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    print(f"\ncompleted {len(done)}/{len(study.trials)} trials")
    if not done:
        return
    best = study.best_trial
    if refine:
        a = best.user_attrs
        print(f"best score = {best.value:.4f}  (opt_median="
              f"{a.get('opt_median', float('nan'))}, val_mean={a.get('val_mean', float('nan'))}, "
              f"val_worst={a.get('val_worst', float('nan'))})")
    else:
        print(f"best score = {best.value:.4f}  (mean_delta="
              f"{best.user_attrs.get('mean_delta', float('nan')):.4f})")
    merged = P.normalize({**P.DEFAULTS, **best.params})
    print("best params:")
    print(json.dumps(merged, indent=2))
    out = config.STUDY_DIR / f"{study.study_name}.best.json"
    out.write_text(json.dumps(merged, indent=2))
    print(f"written -> {out}")

    try:
        imp = optuna.importance.get_param_importances(study)
        print("\nparameter importance:")
        for k, v in imp.items():
            print(f"  {v:6.3f}  {k}")
    except Exception as e:  # importance needs >1 completed trial with variance
        print(f"(importance unavailable: {e})")


def cmd_best(args: argparse.Namespace) -> None:
    study = optuna.load_study(study_name=args.study, storage=_storage(args.study))
    _report(study)


def cmd_validate(args: argparse.Namespace) -> None:
    best_path = config.STUDY_DIR / f"{args.study}.best.json"
    if not best_path.exists():
        sys.exit(f"no best params at {best_path}; run 'search' first")
    candidate = json.loads(best_path.read_text())

    # Ensure baselines cover the (wider) validation distances.
    baseline.build_all_baselines(
        sorted(set(config.BASELINE_DISTANCES) | set(config.VALIDATION_DISTANCES))
    )

    cases = holdout_cases(config.VALIDATION_DISTANCES)
    print(f"validating on {len(cases)} holdout cases ...")
    deltas = []
    for c in cases:
        m = run_case(c, candidate)
        bl = baseline.get(c.name)
        delta = m.ss2 - bl.ss2_at_rate(m.bpp)
        deltas.append(delta)
        flag = "  <-- REGRESSION" if delta < -0.2 else ""
        print(f"  {c.name:28s} d={c.distance:<4g} bpp={m.bpp:6.3f} "
              f"ss2={m.ss2:7.3f} delta={delta:+.3f}{flag}")
    import numpy as np
    print(f"\nholdout mean delta = {np.mean(deltas):+.4f}, "
          f"p10 = {np.percentile(deltas, 10):+.4f}, "
          f"worst = {np.min(deltas):+.4f}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="parameters_fit.optimize")
    sub = parser.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("baseline", help="build/refresh baseline RD curves")
    b.add_argument("--force", action="store_true")
    b.set_defaults(func=cmd_baseline)

    s = sub.add_parser("search", help="run a TPE study")
    s.add_argument("--study", default="jixel_hq_v1")
    s.add_argument("--trials", type=int, default=200)
    s.add_argument("--jobs", type=int, default=1)
    s.add_argument("--seed", type=int, default=7)
    s.add_argument("--startup", type=int, default=16)
    s.add_argument("--params", default="", help="comma-separated subset to tune")
    s.add_argument("--distances", default="",
                   help="comma-separated search distances (default: config.SEARCH_DISTANCES)")
    s.add_argument("--prune-delta", type=float, default=obj_cfg.PRUNE_DELTA,
                   help="prune a trial when any single case regresses past this")
    s.add_argument("--val-reject", type=float, default=obj_cfg.VAL_REJECT,
                   help="refine mode: reject when any internal-val case regresses past this")
    s.add_argument("--refine", action="store_true",
                   help="use the clamped/median objective with an internal train/val split")
    s.set_defaults(func=cmd_search)

    g = sub.add_parser("best", help="print best params for a study")
    g.add_argument("--study", default="jixel_hq_v1")
    g.set_defaults(func=cmd_best)

    v = sub.add_parser("validate", help="evaluate best params on holdout")
    v.add_argument("--study", default="jixel_hq_v1")
    v.set_defaults(func=cmd_validate)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
