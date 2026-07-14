"""The tuning parameter space.

Every key here must match a field understood by ``jixel``'s runtime tuning
loader (``src/tuning.rs``). The *defaults* reproduce the shipped encoder exactly
— enqueue them into the study so the baseline is always a trial.

The dead-zone thresholds are parameterized as ``base`` plus non-negative deltas
so the four-quadrant ordering (t0 <= t1 <= t2 <= t3) stays valid by
construction, exactly as the plan requires.
"""

from __future__ import annotations

from typing import Any

import optuna

# The shipped constants — one Optuna trial should always evaluate these.
DEFAULTS: dict[str, float | int] = {
    "ac_dc_cap": 1.60,
    "ac_dc_hq_cap": 2.30,
    "hf_aq_mul": -0.38,
    "hf_aq_offset": 0.42,
    "gamma_aq_mul": 0.1005613337192697,
    "blue_aq_mul": 0.90590804735610064,
    "bias_rect": 0.92,
    "bias_16x16": 0.86,
    "bias_rect32": 1.10,
    "deadzone_base": 0.58,
    "deadzone_d1": 0.055,   # -> 0.635
    "deadzone_d2": 0.025,   # -> 0.660
    "deadzone_d3": 0.040,   # -> 0.700
    "x_high_add": 0.08,
    "b_high_val": 0.75,
    "x_qm_scale_base": 2,
}

# Search bounds. (lo, hi) for continuous, or a list for categorical/int.
# Chosen to bracket the defaults with room on both sides.
SPACE: dict[str, Any] = {
    "ac_dc_cap": (1.40, 2.60),
    "ac_dc_hq_cap": (1.80, 2.80),
    "hf_aq_mul": (-0.50, -0.10),
    "hf_aq_offset": (0.30, 0.55),
    "gamma_aq_mul": (0.02, 0.22),
    "blue_aq_mul": (0.40, 1.60),
    "bias_rect": (0.85, 1.06),
    "bias_16x16": (0.80, 1.08),
    "bias_rect32": (0.95, 1.20),
    "deadzone_base": (0.48, 0.66),
    "deadzone_d1": (0.0, 0.10),
    "deadzone_d2": (0.0, 0.08),
    "deadzone_d3": (0.0, 0.08),
    "x_high_add": (0.0, 0.14),
    "b_high_val": (0.60, 0.85),
    "x_qm_scale_base": [2, 3],  # discrete
}


def suggest(trial: optuna.Trial, keys: list[str] | None = None) -> dict[str, float | int]:
    """Ask Optuna for one point in the (sub)space defined by ``keys``.

    Parameters not in ``keys`` are frozen to their default (Stage-B focusing).
    """
    keys = keys if keys is not None else list(SPACE.keys())
    out: dict[str, float | int] = dict(DEFAULTS)
    for key in keys:
        spec = SPACE[key]
        if isinstance(spec, list):
            out[key] = trial.suggest_categorical(key, spec)
        else:
            lo, hi = spec
            out[key] = trial.suggest_float(key, lo, hi)
    return out
