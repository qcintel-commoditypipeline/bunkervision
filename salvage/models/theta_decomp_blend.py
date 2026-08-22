"""
Candidate level-model: theta_decomp_blend
Family: ensemble / weighted average.

Idea
----
Blend the two lowest-overfit, near-unbiased models in the bake-off, drawn from
DIFFERENT error families so their mistakes are as decorrelated as the panel allows
and their small opposite-signed biases cancel:

  * theta_lag_ensemble    -- Theta/damped-ETS + lag-regression blend.
                             MAE% 4.01, bias -0.08 (slight UNDER-prediction).
  * decomp_regression_stack -- STL-decomposition backbone + AR(1) on the backbone
                             residual.  MAE% 4.26, bias +0.36 (slight OVER).

theta_lag_ensemble is a smoother/lag forecaster on the y-history; the decomp stack
is a structural decomposition that corrects its own one-step error via a residual
AR(1). They reach the same level from genuinely different mechanisms, so averaging
them shaves variance (validated: RMSE% 5.20 vs 5.28 / 5.35 for the parents) and the
+0.36 / -0.08 biases net to roughly +0.14 -- still near zero.

Weighting choice: a FIXED equal weight (0.5/0.5).
-----------------------------------------------
A weight sweep on the 40 OOS folds is FLAT across wa in [0.4, 0.7] for both MAE%
(3.98-4.04) and RMSE% (~5.20): the blend is insensitive to the exact weight, the
signature of a non-overfit ensemble. A non-leaky train-only inverse-MAE scheme was
also tried; because the two components have nearly identical MAE it lands at
wa~=0.5 and produces MAE% 4.00 -- statistically indistinguishable from the fixed
0.5 blend but at the cost of a nested per-fold walk-forward (much slower) and extra
moving parts. With only ~40 folds, fitting the weight would itself be an in-sample
choice; an equal average of two comparable, near-unbiased models is the classic
variance reducer and the most defensible low-overfit option, so we ship the fixed
weight.

Validated (walk_forward, min_train=36, 40 OOS folds):
    MAE%=4.01  bias=+0.14  rmse=5.20  within5=26/40  within10=37/40  worst=15.28
beating the benchmarks (seasonal_naive 8.35, trailing-12-mean 5.27) and matching
the best parent's MAE with strictly lower RMSE and near-zero bias.

Leakage discipline
------------------
This model adds NO new data access. It only calls the two component
forecast_one functions, each of which already obeys the rules:
  - fits only on `train` (months strictly before the test month, expanding window),
  - reads from test_row only SAFE_TEST_COLS (calendar + prior-actual lags),
  - never touches test_row['y'] or same-month port/price (sg_*) columns.
The blend is a deterministic function of the two component outputs (a fixed-weight
average + a recent-level sanity band), so it inherits their leakage-safety. Refit
happens every fold via the harness. Parameter count added by the blend: ZERO fitted
parameters (the weight is a fixed constant, not estimated).
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

# Import the two component models. They live in the same directory; make that
# directory importable whether this file is run as a script or imported.
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import theta_lag_ensemble        # noqa: E402
import decomp_regression_stack   # noqa: E402

# Equal weight on the two near-unbiased components from different error families.
# Fixed constant (NOT fold-tuned); the OOS sweep is flat around 0.5.
_W_THETA = 0.5


def _seasonal_fallback(train: pd.DataFrame, test_row) -> float:
    """Same-month trailing mean if BOTH components return nothing usable."""
    m = int(test_row["month_num"])
    same = train.loc[train["month_num"] == m, "y"]
    if len(same):
        return float(same.tail(5).mean())
    return float(train["y"].tail(12).mean())


def forecast_one(train: pd.DataFrame, test_row) -> float:
    """Equal-weight blend of theta_lag_ensemble and decomp_regression_stack."""
    preds = []
    weights = []

    try:
        a = float(theta_lag_ensemble.forecast_one(train, test_row))
        if np.isfinite(a) and a > 0:
            preds.append(a)
            weights.append(_W_THETA)
    except Exception:
        pass

    try:
        b = float(decomp_regression_stack.forecast_one(train, test_row))
        if np.isfinite(b) and b > 0:
            preds.append(b)
            weights.append(1.0 - _W_THETA)
    except Exception:
        pass

    if not preds:
        return _seasonal_fallback(train, test_row)

    w = np.asarray(weights, dtype=float)
    w = w / w.sum()  # renormalise if one component dropped out
    pred = float(np.dot(w, np.asarray(preds, dtype=float)))

    # Generous sanity band around the recent level (inherited discipline; does
    # not bind on this panel, guards a pathological fold).
    recent = float(np.mean(np.asarray(train["y"], dtype=float)[-12:]))
    lo, hi = 0.5 * recent, 1.8 * recent
    return float(min(max(pred, lo), hi))


if __name__ == "__main__":
    sys.path.insert(0, "salvage")
    import btlib

    df = btlib.load_panel()
    res = btlib.walk_forward(df, forecast_one, min_train=36)
    print(btlib.metrics(res))
