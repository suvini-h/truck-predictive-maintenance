"""Shared pieces of the risk model (notebooks 03 and 04).

- model_matrix(): the numeric input matrix both models use (80 features + 8 spec codes).
- Recalibration: temperature + class biases on log-probabilities, optionally with a
  quadratic age term on the "repair due" classes; fitted on validation (cross-fitted there).
- pretty(): readable names for the anonymized features ("counter 427, rate over last 5 readouts").
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from make_features_sql import FEATURES

SPEC_COLS = [f"Spec_{i}" for i in range(8)]
INPUTS = FEATURES + SPEC_COLS                  # column order of the model matrix
AGE_IDX = FEATURES.index("time_step")
AGE_CENTER, AGE_SCALE = 170.0, 100.0           # fixed scaling for the calibrator's age term
MIN_GROUP = 200                                # spec categories below this are pooled ("Other")


def signed_log(x):
    """Sign-preserving log1p: tames the heavy right tails of counters for the linear model."""
    return np.sign(x) * np.log1p(np.abs(x))


def spec_levels_from(train: pd.DataFrame, weight_col: str = "w_vehicle") -> dict:
    """Spec categories with >= MIN_GROUP training vehicles; everything else -> 'Other' (code last)."""
    levels = {}
    for c in SPEC_COLS:
        vehicles = train.groupby(c)[weight_col].sum()
        levels[c] = sorted(vehicles[vehicles >= MIN_GROUP - 1e-9].index) + ["Other"]
    return levels


def model_matrix(df: pd.DataFrame, spec_levels: dict) -> np.ndarray:
    """(n, 88) float matrix: 80 numeric features (NaN kept) + 8 integer spec codes."""
    num = df[FEATURES].to_numpy(dtype=float)
    codes = []
    for c in SPEC_COLS:
        lv = spec_levels[c]
        s = df[c].where(df[c].isin(lv[:-1]), "Other")
        codes.append(pd.Categorical(s, categories=lv).codes.astype(float))
    return np.column_stack([num] + codes)


def sqrt_balanced_class_weights(y, w) -> dict:
    """Class weight = sqrt(N / (5 * N_k)) on vehicle-weighted counts: helps rare classes
    without the extreme distortion of full balancing; calibration later restores priors."""
    y, w = np.asarray(y), np.asarray(w)
    total = w.sum()
    return {k: float(np.sqrt(total / (5 * w[y == k].sum()))) for k in range(5)}


# ---------------------------------------------------------------- recalibration
def n_calib_params(age_terms: bool) -> int:
    return 7 if age_terms else 5


def apply_calibrator(logp, age, params, age_terms: bool) -> np.ndarray:
    """z = log p / T + b (b_0 = 0); with age_terms, the due classes 1-4 also get
    a1*u + a2*u^2 with u = (age - 170) / 100. Returns calibrated probabilities."""
    T = np.exp(params[0])
    z = np.asarray(logp) / T + np.r_[0.0, params[1:5]]
    if age_terms:
        u = (np.asarray(age) - AGE_CENTER) / AGE_SCALE
        z[:, 1:] += (params[5] * u + params[6] * u ** 2)[:, None]
    z = z - z.max(axis=1, keepdims=True)
    p = np.exp(z)
    return p / p.sum(axis=1, keepdims=True)


def fit_calibrator(logp, age, y, age_terms: bool, ridge: float = 1e-3) -> np.ndarray:
    y = np.asarray(y, dtype=int)
    idx = np.arange(len(y))

    def loss(params):
        p = apply_calibrator(logp, age, params, age_terms)
        return -np.log(p[idx, y] + 1e-12).mean() + ridge * np.sum(params[1:] ** 2)

    return minimize(loss, np.zeros(n_calib_params(age_terms)), method="L-BFGS-B").x


def due_log_odds(p) -> np.ndarray:
    """log P(repair due within 48 steps) / P(not due) = log (1 - p0) / p0."""
    p0 = np.clip(np.asarray(p)[:, 0], 1e-9, 1 - 1e-9)
    return np.log((1 - p0) / p0)


# ---------------------------------------------------------------- readable names
_SUFFIX = {
    "level": "level",
    "delta_prev": "change since previous readout",
    "rate_prev": "rate since previous readout",
    "delta_5": "change over last 5 readouts",
    "rate_5": "rate over last 5 readouts",
    "total": "total count",
    "mean_bin": "mean bin (lifetime)",
    "top2_share": "share in top 2 bins (lifetime)",
    "mean_bin_change_5": "change in mean bin over last 5 readouts",
    "recent5_mean_bin": "mean bin (last 5 readouts)",
    "recent5_top2_share": "share in top 2 bins (last 5 readouts)",
}
_TIMING = {
    "time_step": "age (time_step)",
    "readouts_so_far": "readouts so far",
    "dt_prev": "gap since previous readout",
    "dt_last5": "span of last 5 readouts",
}


def pretty(feature: str) -> str:
    if feature in _TIMING:
        return _TIMING[feature]
    if feature.startswith("Spec_"):
        return feature.replace("_", " ")
    m = re.fullmatch(r"([ch])(\d+)_(.+)", feature)
    if m:
        kind = "counter" if m.group(1) == "c" else "histogram"
        return f"{kind} {m.group(2)}, {_SUFFIX.get(m.group(3), m.group(3))}"
    return feature
