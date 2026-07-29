# parameters_fit

```
parameters ──► encode corpus ──► size + SSIMULACRA2 + time ──► optimizer proposes better parameters
```

## How the pieces fit

> **The encoder currently has NO runtime tuning module.** Winning configs get
> folded into constants and the override module is deleted, so `params.py` keys
> are inert until someone re-adds a `JIXEL_TUNING_JSON` reader (see
> `src/ac_tuning.rs` in the transform-merge study, git history). Always confirm
> the encoder actually reads a key — run `probe --defaults` and check the deltas
> are exactly 0, then flip one knob and check they are not.

| Layer                          | What it does                                                                                                                                                                     |
|--------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| tuning module (jixel)          | Reads a flat JSON of tuning knobs from `JIXEL_TUNING_JSON` **once per process**. Unset ⇒ shipped defaults, byte-for-byte. Removed once a study lands.                            |
| `tuner/` crate (`jixel-tuner`) | Encode-only CLI: image → `.jxl`, prints `{bytes, encode_ms, bpp,…}`. Never recompiled between trials.                                                                            |
| `encoder.py`                   | encode (`jixel-tuner`) → decode (`djxl`) → score (`ssimulacra2`) for one case.                                                                                                   |
| `corpus.py`                    | One aligned 768² crop per source image (16 from `train0` + an evenly-spaced sample of `DIV2K_train_HR`); deterministic 75/25 train/holdout split **by image** (no crop leakage). |
| `baseline.py`                  | Per-crop rate→quality curve at the shipped defaults, cached to JSON. `covers_rate()` flags candidates outside the measured bitrate range — `np.interp` clamps there and fabricates huge fake deltas, so **BASELINE_DISTANCES must stay strictly wider than anything you probe**. |
| `probe.py`                     | Per-distance unclamped deltas for one explicit config. Run this on a study winner's top knob before believing it — the clamped/median refine objective can hide a single image's collapse. |
| `objective.py`                 | Rate-matched delta + robust aggregate score.                                                                                                                                     |
| `optimize.py`                  | Optuna study driver + CLI.                                                                                                                                                       |

## The parameters (all in `params.py` / `src/tuning.rs`)

| key                 | default        | meaning                                                                   |
|---------------------|----------------|---------------------------------------------------------------------------|
| `ac_dc_cap`         | 1.60           | AC/DC quant floor (`distance ≥ 1`)                                        |
| `ac_dc_hq_cap`      | 2.30           | AC/DC quant on the HQ plateau (`distance ≤ 0.75`)                         |
| `hf_aq_mul`         | −0.38          | HF adaptive-quant slope                                                   |
| `hf_aq_offset`      | 0.42           | HF adaptive-quant offset (↑ ⇒ more bpp)                                   |
| `bias_rect`         | 0.92           | merge bias, rectangular 16×8 / 8×16                                       |
| `bias_16x16`        | 0.86           | merge bias, DCT16×16                                                      |
| `bias_rect32`       | 1.10           | merge bias, rect-32                                                       |
| `deadzone_base`     | 0.58           | AC dead-zone quadrant 0                                                   |
| `deadzone_d1/d2/d3` | .055/.025/.040 | **non-negative** deltas → quadrants 1..3 (ordering valid by construction) |
| `x_high_add`        | 0.08           | extra dead-zone on X high bands                                           |
| `b_high_val`        | 0.75           | flat high-band dead-zone for B                                            |
| `x_qm_scale_base`   | 2              | base X quant-matrix scale (discrete 2/3)                                  |

## Why rate-matched, not raw SSIMULACRA2

Changing a parameter changes bitrate, so optimizing raw SSIMULACRA2 at a fixed
`distance` would just reward configurations that spend more bits. Instead, for
each crop we precompute a baseline rate/quality curve, and score a candidate by

```
delta = candidate_ss2 − baseline_ss2_interpolated_at(candidate_bpp)   # log-bpp interpolation
```

The aggregate (in `objective.py`) rewards mean gain but strongly penalizes broad
regressions (p10) and the worst case, and mildly penalizes slowdowns — so the
optimizer can't win by helping easy images while wrecking hard ones. A single
case regressing past −2.0 prunes the whole trial.

**Sanity check built in:** the shipped defaults are always enqueued as a trial;
because the baseline *is* the defaults, that trial scores `mean_delta ≈ 0`.

## Usage

```bash
# 0) one-time: build the encoder + Python env
cargo build --release -p tuner            # from repo root
python3 -m venv parameters_fit/.venv
parameters_fit/.venv/bin/pip install -r parameters_fit/requirements.txt

# run everything from the repo root:
PY=parameters_fit/.venv/bin/python

# 1) build baseline RD curves (cached; ~45s for 16 crops)
$PY -m parameters_fit.optimize baseline

# 2) Stage A — sensitivity scan (all knobs, find what matters)
$PY -m parameters_fit.optimize search --study stageA --trials 80

# 3) Stage B — focus on the parameters that mattered
$PY -m parameters_fit.optimize search --study stageB --trials 300 \
    --params hf_aq_mul,hf_aq_offset,deadzone_base,ac_dc_cap,bias_16x16

# inspect / resume (studies are SQLite, resumable)
$PY -m parameters_fit.optimize best     --study stageB

# 4) Stage C — validate the winner on held-out images + a wider sweep
$PY -m parameters_fit.optimize validate --study stageB
```

The winning parameters are written to `work/studies/<study>.best.json`. To ship
them, fold the values back into the defaults in `src/tuning.rs`.

## Applying a config manually

Any `.json` of the keys above can be handed straight to the encoder:

```bash
JIXEL_TUNING_JSON=my_params.json ./target/release/jixel-tuner in.png out.jxl -d 0.7
```

## Staging

1. **A — sensitivity:** ~64–100 trials, all knobs, `d = 0.1/0.3/0.7`. Read the
   Optuna importance table; drop inactive knobs.
2. **B — focus:** ~150–300 trials on the survivors.
3. **C — validate:** top configs on holdout images + full sweep; pick from the
   Pareto frontier, not blindly trial #1.
