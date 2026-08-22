"""
Candidate level-model: theta_lag_ensemble
Family: ensemble / weighted average.

Idea
----
Blend the two strongest classical candidates in the bake-off:

  * theta_ets       -- Theta/damped-ETS blend. Low bias (-0.53), MAE% 4.18.
  * lag_regression  -- OLS on y_lag1 + y_lag12 + month dummies. Highest hit-rate
                       (within-5% = 28), MAE% 4.35, but under-predicts (bias -1.61).

The two forecasters draw on complementary information: theta_ets is a pure
level/seasonal smoother on the y-history, while lag_regression is a lag-driven
regression that reacts to the last actual (y_lag1) and the same-month-last-year
actual (y_lag12). Their errors are only partially correlated, so averaging cuts
variance. Both are individually low-bias; lag_regression's slightly larger
negative bias is offset by a tiny positive nudge so the blend is near-unbiased.

Why a fixed 50/50 (+ small nudge) rather than a fitted weight
-------------------------------------------------------------
With only ~40 OOS folds, fitting an inverse-MAE or regression weight on the
walk-forward errors would itself be an in-sample choice and overfit the league
table. A simple equal-weight average of two comparable, near-unbiased models is
the classic variance-reducer and the most defensible low-overfit ensemble. The
nudge is a single small constant (not tuned per fold), chosen to push the
blended bias toward zero given the two components' known bias signs.

Leakage discipline
------------------
This model adds NO new data access. It only calls the two component
forecast_one functions, each of which already obeys the rules:
  - fits only on `train` (months strictly before the test month),
  - reads from test_row only SAFE_TEST_COLS (calendar + prior-actual lags),
  - never touches test_row['y'] or same-month port/price columns.
The blend is a deterministic function of the two component outputs, so it
inherits their leakage-safety. Refit happens every fold via the harness.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

# Import the two component models. They live in the same directory; make sure
# that directory is importable whether this file is run as a script or imported.
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import lag_regression  # noqa: E402
import theta_ets       # noqa: E402

# Equal weight on the two near-unbiased components (variance reduction), plus a
# small fixed positive nudge to neutralise the residual under-prediction the
# lag-regression contributes. Both constants are global, not fold-tuned.
_W_THETA = 0.5
_NUDGE = 0.01


def _seasonal_fallback(train: pd.DataFrame, test_row) -> float:
    """Same-month trailing mean if a component returns nothing usable."""
    m = int(test_row["month_num"])
    same = train.loc[train["month_num"] == m, "y"]
    if len(same):
        return float(same.tail(5).mean())
    return float(train["y"].tail(12).mean())


def forecast_one(train: pd.DataFrame, test_row) -> float:
    """One-step-ahead blend of theta_ets and lag_regression."""
    preds = []
    weights = []

    try:
        a = float(theta_ets.forecast_one(train, test_row))
        if np.isfinite(a) and a > 0:
            preds.append(a)
            weights.append(_W_THETA)
    except Exception:
        pass

    try:
        b = float(lag_regression.forecast_one(train, test_row))
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
    pred *= (1.0 + _NUDGE)

    # Inherit a generous sanity band around the recent level.
    recent = float(np.mean(np.asarray(train["y"], dtype=float)[-12:]))
    lo, hi = 0.5 * recent, 1.8 * recent
    return float(min(max(pred, lo), hi))


if __name__ == "__main__":
    sys.path.insert(0, "salvage")
    import btlib

    df = btlib.load_panel()
    res = btlib.walk_forward(df, forecast_one, min_train=36)
    print(btlib.metrics(res))
