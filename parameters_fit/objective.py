"""Rate-matched objective and robust aggregate score.

For each case we compute a *rate-matched* quality delta::

    delta = candidate_ss2 - baseline_ss2_at(candidate_bpp)

i.e. how much better (or worse) the candidate is than the shipped encoder *at
the bitrate the candidate actually produced*. Positive is better.

The aggregate rewards mean gain but strongly penalizes broad regressions and
mildly penalizes speed loss, so the optimizer can't win by helping easy images
while wrecking hard ones.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Hard-rejection thresholds (see plan). During search a trial is pruned if any
# single case regresses past PRUNE_DELTA; validation uses a tighter bar.
PRUNE_DELTA = -2.0
VALIDATION_MAX_REGRESSION = -0.2

# Refinement mode: a config is rejected outright if any internal-validation case
# regresses past this. Steep-RD bands (high quality especially) swing by several
# points per case, so studies restricted to those bands relax it via
# ``optimize.py --val-reject``.
VAL_REJECT = -0.5


@dataclass
class CaseResult:
    name: str
    distance: float
    bpp: float
    ss2: float
    delta: float
    time_ratio: float


def clamp_deltas(deltas: list[float], clamp: float = 1.5) -> np.ndarray:
    """Clamp per-case deltas to +/-clamp. SSIMULACRA2 matched-rate deltas swing
    by many points on steep-RD images and at very low quality; clamping keeps a
    single outlier (good or bad) from dominating selection."""
    return np.clip(np.asarray(deltas, dtype=np.float64), -clamp, clamp)


def refine_score(
    opt_deltas: list[float],
    val_deltas: list[float],
    time_ratios: list[float],
    clamp: float = 1.5,
) -> float:
    """Robust objective for the refinement round (reviewer's structure).

    Uses the *median* of the optimization set (outlier-proof central tendency)
    plus an internal-validation term that rewards broad val gain and strongly
    penalizes val regressions — so a config that overfits the optimization
    images is not selected. All deltas are clamped first.
    """
    if not opt_deltas or not val_deltas:
        return float("-inf")
    o = clamp_deltas(opt_deltas, clamp)
    v = clamp_deltas(val_deltas, clamp)
    ratios = np.asarray(time_ratios, dtype=np.float64)
    time_penalty = max(float(np.mean(np.log(np.clip(ratios, 1e-6, None)))), 0.0)

    return (
        float(np.median(o))
        + 1.0 * float(np.mean(v))
        + 1.5 * min(float(np.percentile(v, 10)), 0.0)
        + 0.5 * min(float(np.min(v)), 0.0)
        - 0.10 * time_penalty
    )


def robust_score(deltas: list[float], time_ratios: list[float]) -> float:
    """Aggregate per-case rate-matched deltas into a single scalar to maximize.

    score = mean(delta)
            + 1.5 * min(p10(delta), 0)     # punish broad regressions
            + 0.25 * min(min(delta), 0)    # punish the worst case
            - 0.10 * max(mean(log time_ratio), 0)  # discourage slowdowns
    """
    if not deltas:
        return float("-inf")
    values = np.asarray(deltas, dtype=np.float64)
    ratios = np.asarray(time_ratios, dtype=np.float64)

    mean_delta = float(np.mean(values))
    p10_delta = float(np.percentile(values, 10))
    worst_delta = float(np.min(values))
    time_penalty = max(float(np.mean(np.log(np.clip(ratios, 1e-6, None)))), 0.0)

    return (
        mean_delta
        + 1.5 * min(p10_delta, 0.0)
        + 0.25 * min(worst_delta, 0.0)
        - 0.10 * time_penalty
    )
