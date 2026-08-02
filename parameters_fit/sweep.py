"""1-D sweep of a single knob — the check that decides whether a study's winner
is real.

A joint study optimizes a scalar aggregate, and a clamped/median objective can
rank a knob highly while that knob is neutral or harmful on its own (it happened
twice in the transform-merge study). Sweeping one key at a time over the corpus,
with unclamped per-distance deltas, shows the actual shape.

    python -m parameters_fit.sweep --key dc_refine_peak \
        --values 1.15,1.25,1.35,1.45 --distances 1.5,2.5,4.0
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from . import config
from .params import DEFAULTS, normalize
from .probe import evaluate


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="parameters_fit.sweep")
    p.add_argument("--key", required=True)
    p.add_argument("--values", required=True, help="comma-separated values to try")
    p.add_argument("--distances", default="")
    p.add_argument("--base", default="", help="JSON of other overrides held fixed")
    p.add_argument("--holdout", action="store_true")
    args = p.parse_args(argv)

    if args.key not in DEFAULTS:
        p.error(f"unknown key {args.key!r}; known: {', '.join(sorted(DEFAULTS))}")
    distances = (
        [float(x) for x in args.distances.split(",")]
        if args.distances else config.SEARCH_DISTANCES
    )
    base = json.loads(args.base) if args.base else {}
    values = [float(v) for v in args.values.split(",")]

    print(f"sweeping {args.key} over {values}"
          + (f", holding {base}" if base else "")
          + f", d={distances}{' [holdout]' if args.holdout else ''}\n")
    header = f"{args.key:>22} " + " ".join(f"{d:>8g}" for d in distances) + f" {'ALL':>8} {'worst':>8}"
    print(header)
    print("-" * len(header))
    for v in values:
        params = normalize({**DEFAULTS, **base, args.key: v})
        r = evaluate(params, distances, args.holdout)
        per_d = [float(np.mean(r["by_distance"][d])) for d in distances]
        allv = np.concatenate([np.asarray(r["by_distance"][d]) for d in distances])
        flag = "  !! out-of-range" if r["out_of_range"] else ""
        print(f"{v:>22g} " + " ".join(f"{x:+8.3f}" for x in per_d)
              + f" {allv.mean():+8.3f} {allv.min():+8.3f}{flag}")
    print(f"\n(shipped default: {args.key} = {DEFAULTS[args.key]})")


if __name__ == "__main__":
    main()
