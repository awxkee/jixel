"""The tuning parameter space — sub-8x8 transform selection.

`fill_ac_strategy` commits DCT4X4/DCT4X8/DCT8X4 wherever the per-block RD
comparison beats the 8x8 incumbent. Unlike every large transform (which got
distance-`Banded` biases plus explicit merge margins during the transform-merge
work), the sub-8 path shipped with **flat `BIAS_4X4 = BIAS_4X8 = 1.0`, no
acceptance margin, and no upper distance gate** — it was never fitted.

Measured before fitting (BD-rate of DISABLING sub-8 entirely; negative means
turning it off wins):

    Kodak   HQ -0.356%  low -0.363%  full -0.341%   (22 of 24 images)
    train0  HQ +0.130%  low -0.123%  full -0.035%   (10 of 16 help at HQ)

So the family genuinely pays on large photographic content at high quality but
is too permissive elsewhere. Both references matter: a fit has to beat the
current settings *and* beat switching sub-8 off.

`bias_*` scale each candidate's cost before it is compared with DCT8, so >1
makes that strategy harder to select. An explicit `margin` knob would be exactly
redundant with `bias_4x8` (DCT4X4 is selected for ~0% of blocks, so the 4x8/8x4
bias alone sets the acceptance bar), and is deliberately not in the space.
"""

from __future__ import annotations

from typing import Any

import optuna

# Shipped: flat 1.0 biases, no distance gate.
DEFAULTS: dict[str, float | int] = {
    "sub8_bias_4x4": 1.0,
    "sub8_bias_4x8": 1.0,
    "sub8_max_distance": 100.0,   # effectively "never gated"
    "dct64_accept": 0.945,
    "dct64_rect_accept": 0.642,
    "dct64_rect_qm_scale": 1.0,
    "dct64_rect_y_hf": 1.0,
}

SPACE: dict[str, Any] = {
    "sub8_bias_4x4": (0.85, 1.35),
    "sub8_bias_4x8": (0.95, 1.40),
    # Below ~0.5 sub-8 would be off almost everywhere; 100 leaves it always on.
    "sub8_max_distance": (0.5, 100.0),
    # Higher values admit progressively more 64x64 merges. The old selector
    # used 0.84; values above ~1.1 are intentionally included to reveal the
    # point where rate/quality starts to collapse rather than hiding it.
    "dct64_accept": (0.70, 1.20),
    "dct64_rect_accept": (0.70, 1.10),
    "dct64_rect_qm_scale": (0.75, 1.30),
    "dct64_rect_y_hf": (0.75, 1.60),
}

_ENV = {
    "sub8_bias_4x4": "JIXEL_SUB8_BIAS4X4",
    "sub8_bias_4x8": "JIXEL_SUB8_BIAS4X8",
    "sub8_max_distance": "JIXEL_SUB8_MAXD",
    "dct64_accept": "JIXEL_DCT64_ACCEPT",
    "dct64_rect_accept": "JIXEL_DCT64_RECT_ACCEPT",
    "dct64_rect_qm_scale": "JIXEL_DCT64_RECT_QM_SCALE",
    "dct64_rect_y_hf": "JIXEL_DCT64_RECT_Y_HF",
}


#: Every environment variable this space can set. ``encoder.run_case`` clears
#: all of them before each encode so a stale value from the parent process (or
#: a previous study) can never leak into a measurement.
MANAGED_ENV: frozenset[str] = frozenset(_ENV.values())


def suggest(trial: optuna.Trial, keys: list[str] | None = None) -> dict[str, float | int]:
    """Ask Optuna for one point in the (sub)space defined by ``keys``."""
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


def normalize(params: dict[str, float | int]) -> dict[str, float | int]:
    """No cross-knob orderings to enforce in this space."""
    return dict(params)


def to_env(params: dict[str, float | int] | None) -> dict[str, str]:
    """Render a parameter point as the encoder's tuning environment.

    ``None`` means the shipped defaults, which is how baselines are built.
    """
    if params is None:
        return {}
    out = {}
    for key, var in _ENV.items():
        if key not in params:
            continue
        # A 1.0 matrix knob is the unsignalled library default. Skipping the
        # environment variable preserves that state and its zero header cost.
        if key in {"dct64_rect_qm_scale", "dct64_rect_y_hf"} and float(params[key]) == 1.0:
            continue
        out[var] = repr(float(params[key]))
    return out
