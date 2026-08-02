"""Evaluate one explicit config (or the defaults) and print per-distance deltas.

Unlike ``optimize``, this does no searching: it runs the corpus once and reports
the rate-matched SSIMULACRA2 delta broken down by distance, which is how a
distance-scheduled knob is actually judged.

    python -m parameters_fit.probe --params '{"dct8_only_max_distance":0.0}'
    python -m parameters_fit.probe --file work/studies/x.best.json --holdout
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from . import baseline, config
from .corpus import holdout_cases, train_cases
from .encoder import run_case
from .params import DEFAULTS, normalize


def evaluate(
    params: dict | None,
    distances: list[float],
    holdout: bool,
    images: list[str] | None = None,
) -> dict:
    cases = holdout_cases(distances) if holdout else train_cases(distances)
    if images:
        cases = [c for c in cases if any(s in c.name for s in images)]
    by_distance: dict[float, list[float]] = defaultdict(list)
    by_case: list[tuple[str, float, float, float, float]] = []
    time_ratio: list[float] = []
    out_of_range: list[str] = []
    for case in cases:
        m = run_case(case, params)
        bl = baseline.get(case.name)
        if not bl.covers_rate(m.bpp):
            out_of_range.append(f"{case.key()} @ {m.bpp:.4f} bpp")
        delta = m.ss2 - bl.ss2_at_rate(m.bpp)
        by_distance[case.distance].append(delta)
        by_case.append((case.name, case.distance, m.bpp, m.ss2, delta))
        time_ratio.append(m.encode_ms / max(bl.time_at_distance(case.distance), 1e-6))
    return {
        "by_distance": by_distance,
        "by_case": by_case,
        "time_ratio": time_ratio,
        "out_of_range": out_of_range,
    }


def report(result: dict, label: str) -> None:
    print(f"\n=== {label} ===")
    print(f"{'distance':>9}  {'mean':>7} {'median':>7} {'worst':>7} {'best':>7}  n")
    all_deltas: list[float] = []
    for d in sorted(result["by_distance"]):
        v = np.asarray(result["by_distance"][d])
        all_deltas.extend(v.tolist())
        print(
            f"{d:>9g}  {v.mean():+7.3f} {np.median(v):+7.3f} "
            f"{v.min():+7.3f} {v.max():+7.3f}  {len(v)}"
        )
    a = np.asarray(all_deltas)
    tr = np.asarray(result["time_ratio"])
    print(
        f"{'ALL':>9}  {a.mean():+7.3f} {np.median(a):+7.3f} {a.min():+7.3f} "
        f"{a.max():+7.3f}  {len(a)}   time x{tr.mean():.2f}"
    )
    if result["out_of_range"]:
        print(
            f"!! {len(result['out_of_range'])} case(s) outside the baseline rate range "
            f"(deltas there are extrapolation artifacts): "
            f"{', '.join(result['out_of_range'][:4])}"
        )
    worst = sorted(result["by_case"], key=lambda r: r[4])[:5]
    print("worst cases:")
    for name, d, bpp, ss2, delta in worst:
        print(f"  {name:30s} d={d:<5g} bpp={bpp:7.4f} ss2={ss2:7.3f} delta={delta:+.3f}")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="parameters_fit.probe")
    p.add_argument("--params", default="", help="inline JSON of overrides")
    p.add_argument("--file", default="", help="path to a JSON config")
    p.add_argument("--label", default="")
    p.add_argument("--distances", default="")
    p.add_argument("--holdout", action="store_true")
    p.add_argument("--defaults", action="store_true", help="run the shipped defaults (sanity: ~0)")
    p.add_argument("--images", default="", help="comma-separated name substrings to restrict the corpus")
    args = p.parse_args(argv)

    distances = (
        [float(x) for x in args.distances.split(",")]
        if args.distances
        else config.SEARCH_DISTANCES
    )

    if args.defaults:
        params = None
        label = "shipped defaults (expect ~0)"
    else:
        overrides = {}
        if args.file:
            overrides.update(json.loads(Path(args.file).read_text()))
        if args.params:
            overrides.update(json.loads(args.params))
        if not overrides:
            p.error("give --params, --file or --defaults")
        params = normalize({**DEFAULTS, **overrides})
        changed = {k: v for k, v in params.items() if DEFAULTS.get(k) != v}
        label = args.label or json.dumps(changed)

    images = [s for s in args.images.split(",") if s] or None
    result = evaluate(params, distances, args.holdout, images)
    report(result, f"{label}{' [holdout]' if args.holdout else ''}")


if __name__ == "__main__":
    main()
