"""
Candidate level-model: sarima (SARIMAX family)

Approach
--------
Fit a small SARIMA model on the log of the official Singapore monthly bunker
TOTAL. Logs stabilise the multiplicative-looking seasonality/trend and keep the
forecast strictly positive. Orders are kept deliberately low to avoid overfitting
on only ~36-75 training points:

    order          = (1, 1, 1)
    seasonal_order = (1, 1, 0, 12)

This captures: a short-memory AR/MA term, a first difference for the upward
trend, and a seasonal difference + seasonal AR for the strong annual cycle.
(Both this and the (0,1,1,12) MA variant validate well; the seasonal-AR form
edged out slightly lower OOS MAE in a small order sweep while staying just as
parsimonious -- a single seasonal parameter.)

Leakage discipline
-------------------
- The model is fit ONLY on train['y'] (all months strictly before the test month).
- The walk-forward refits each fold (expanding window).
- We forecast exactly one step ahead, so we never touch test_row['y'] nor any
  same-month other-port column. We don't even need the lag columns: the SARIMA
  state already encodes the history.

Robustness
----------
- statsmodels convergence/warnings are fully suppressed.
- If the fit fails (singular matrix, non-convergence to a usable result), we fall
  back to a seasonal-naive style estimate (same calendar month mean, then last
  value) so a fold never crashes the harness.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

# statsmodels emits a lot of convergence / frequency warnings on short series.
warnings.simplefilter("ignore")
try:
    from statsmodels.tools.sm_exceptions import (
        ConvergenceWarning,
        ValueWarning,
        HessianInversionWarning,
    )

    for _w in (ConvergenceWarning, ValueWarning, HessianInversionWarning):
        warnings.simplefilter("ignore", _w)
except Exception:
    pass

from statsmodels.tsa.statespace.sarimax import SARIMAX

ORDER = (1, 1, 1)
SEASONAL_ORDER = (1, 1, 0, 12)


def _fallback(train: pd.DataFrame, test_row) -> float:
    """Seasonal mean for this calendar month; else the last observed level."""
    m = test_row["month_num"]
    same = train[train["month_num"] == m]["y"]
    if len(same) > 0:
        return float(same.tail(5).mean())
    return float(train["y"].iloc[-1])


def forecast_one(train: pd.DataFrame, test_row) -> float:
    y = pd.Series(np.asarray(train["y"], dtype=float)).reset_index(drop=True)

    # Need at least a couple of seasonal cycles for a seasonal difference of 12.
    if len(y) < 24:
        return _fallback(train, test_row)

    logy = np.log(y)

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = SARIMAX(
                logy,
                order=ORDER,
                seasonal_order=SEASONAL_ORDER,
                trend="n",
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            fit = model.fit(disp=0, maxiter=200, method="lbfgs")
        pred_log = float(fit.forecast(steps=1).iloc[0])
        pred = float(np.exp(pred_log))
    except Exception:
        return _fallback(train, test_row)

    # Sanity guard: SARIMA on short logs can occasionally produce a wild value.
    # If the prediction is implausibly far from the recent level, fall back.
    recent = float(y.tail(12).mean())
    if not np.isfinite(pred) or pred <= 0 or pred > 4.0 * recent or pred < 0.25 * recent:
        return _fallback(train, test_row)

    return pred


if __name__ == "__main__":
    import sys
    import os

    sys.path.insert(0, "salvage")
    import btlib

    df = btlib.load_panel()
    res = btlib.walk_forward(df, forecast_one, min_train=36)
    print(btlib.metrics(res))
