"""Offline parameter optimization for the jixel VarDCT encoder.

Treats a handful of encoder tuning constants as an expensive black-box
hyperparameter problem: encode a corpus, measure rate-matched SSIMULACRA2 gain,
and let Optuna's TPE sampler propose better values. See ``README.md``.
"""
