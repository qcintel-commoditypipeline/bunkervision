"""
Candidate level-model: stl_trend
Family: STL decomposition + trend extrapolation.

Approach
--------
For each fold we have `train` = all monthly observations strictly before the
target month (expanding window). We:

  1. Build the contiguous monthly series y_t from train (already sorted, no gaps).
  2. Run statsmodels STL(period=12, robust=True) to split it into
        trend_t + seasonal_t + resid_t.
  3. Extrapolate the TREND forward to the target month. STL trend is a smooth
     local level, so a short local-linear extrapolation off the last few trend
     points (fall back to last value) gives the expected level.
  4. Add the SEASONAL component for the target calendar month. STL seasonal is
     (nearly) periodic; we take the most recent estimate of that month_num's
     seasonal factor.
  5. forecast = extrapolated_trend + seasonal[target_month_num].

No same-month / future columns are read from test_row (only calendar + lags are
even available, and we don't need y). Strictly fit-on-train, refit each fold.

Low parameter count: STL is non-parametric smoothing with fixed period=12; the
only "tunings" are the trend extrapolation horizon (small, fixed) and number of
points used for the local linear slope.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import STL


def forecast_one(train: pd.DataFrame, test_row) -> float:
    y = train["y"].to_numpy(dtype=float)
    n = len(y)

    # Need at least a couple of seasonal cycles for a stable STL.
    if n < 24:
        # Fall back to same-month mean (seasonal naive) early on.
        m = int(test_row["month_num"])
        same = train[train["month_num"] == m]["y"]
        if len(same):
            return float(same.mean())
        return float(y[-1])

    series = pd.Series(y)

    # STL with annual period. robust=False validated better here (MAE% 4.61 vs
    # 4.84 and near-zero bias): the non-robust seasonal adapts faster to the
    # evolving seasonal shape, which matters more than outlier-downweighting on
    # this short 36-75 month window.
    try:
        stl = STL(series, period=12, robust=False).fit()
    except Exception:
        m = int(test_row["month_num"])
        same = train[train["month_num"] == m]["y"]
        return float(same.mean()) if len(same) else float(y[-1])

    trend = stl.trend.to_numpy(dtype=float)
    seasonal = stl.seasonal.to_numpy(dtype=float)

    # --- Extrapolate the trend to the target month --------------------------
    # The target month is exactly one step after the last train month.
    horizon = 1

    # Local linear slope from the last `w` trend points; trend is smooth so a
    # short window captures the current drift without chasing noise.
    w = min(6, n)
    recent = trend[-w:]
    xs = np.arange(w, dtype=float)
    # slope via least squares; robust enough for a smooth monotone-ish trend
    slope = np.polyfit(xs, recent, 1)[0]
    trend_fcast = trend[-1] + slope * horizon

    # Guard against an explosive slope (trend is a level, keep extrapolation
    # within a sane band relative to recent observed level).
    last_level = trend[-1]
    cap = 0.10 * last_level  # at most ~10% one-month trend move
    step = trend_fcast - last_level
    if step > cap:
        trend_fcast = last_level + cap
    elif step < -cap:
        trend_fcast = last_level - cap

    # --- Seasonal factor for the target calendar month ----------------------
    # Map each train index to its month_num, then take the most recent seasonal
    # estimate for the target month_num (STL seasonal evolves slowly).
    target_m = int(test_row["month_num"])
    months = train["month_num"].to_numpy(dtype=int)
    mask = months == target_m
    if mask.any():
        # average the last up-to-3 estimates of this month's seasonal factor
        seas_vals = seasonal[mask]
        seasonal_fac = float(np.mean(seas_vals[-3:]))
    else:
        seasonal_fac = 0.0

    return float(trend_fcast + seasonal_fac)


if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, "salvage")
    import btlib

    df = btlib.load_panel()
    res = btlib.walk_forward(df, forecast_one, min_train=36)
    print(btlib.metrics(res))
