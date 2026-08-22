"""
Candidate level-model: damped_theta_decomp_stack
Family: hybrid -- damped-Theta level/trend backbone + multiplicative STL seasonal
        factor + lag-regression residual correction (one stacked model, not an
        ensemble of three).

Idea
----
The bake-off's three best-performing mechanisms are, individually:
  * a damped trend (Theta/ETS smoothers, low bias),
  * seasonal decomposition (STL trend + month factor),
  * a lag correction on the residual (AR on the backbone error).

Rather than averaging three model outputs, we STACK these three mechanisms into
a single forecaster so each stage cleans up what the previous one leaves behind:

  STAGE 1 -- multiplicative STL seasonal factor.
      Decompose y with STL (period=12, multiplicative via log). The seasonal
      factor s_m for the target calendar month is the geometric-mean of the most
      recent estimates for that month. Deseasonalise the history: d_t = y_t/s_t.

  STAGE 2 -- damped-Theta backbone on the deseasonalised level.
      Fit a damped-trend exponential smoother (Theta's level/trend backbone:
      Holt with a damping phi<1) to the deseasonalised series d_t and take the
      one-step-ahead forecast d_hat. Damping is the key guard: it pulls the
      trend term toward flat, so the multi-day "phantom-event" volume spikes
      noted in the project methodology cannot inflate a runaway trend.
      Backbone B = d_hat * s_m  (re-seasonalise).

  STAGE 3 -- lag-regression on the backbone residual.
      Reconstruct the SAME one-step backbone B_hat_k for every train month k
      (using only y[:k]) and form r_k = y_k - B_hat_k. Fit a no-intercept OLS
      r_t ~ r_lag1 (AR(1) on the backbone error). The no-intercept fit preserves
      the backbone's near-zero bias; the slope captures error persistence (if we
      under-shot last month we tend to under-shoot this month).
      forecast = B_test + slope * r_lag1 .

Each stage targets a distinct error source: STL fixes seasonality, damped-Theta
fixes the level/trend (unbiased, spike-robust), the residual AR fixes the
short-horizon dynamics.

Leakage discipline
------------------
- Every fit (STL, the damped smoother, the residual OLS, all reconstructions)
  uses `train` rows only -- months strictly before the test month. Refit each
  fold (expanding window).
- From test_row we read ONLY calendar fields (month_num). The residual AR uses
  r_lag1 built from prior train actuals. NEVER test_row['y']; NEVER same-month
  port/sg_* columns.
- Parameter count is small: STL is non-parametric (fixed period=12); the damped
  smoother's params are fit by statsmodels' likelihood (damped Holt); the
  residual OLS has a SINGLE parameter (AR(1) slope, no intercept).

Robustness
----------
- Seasonal/level fallbacks when train is too short (<24) or a fit fails.
- Damping bound away from 1 keeps the trend from running away on a spike.
- A std-based clamp on the residual correction guards a pathological OLS fold.
- Final sanity band vs recent levels; degenerate cases return the backbone.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.seasonal import STL

warnings.simplefilter("ignore")

_PERIOD = 12
_MIN_STL = 24  # need >=2 seasons for a stable STL


def _seasonal_factors(logy: np.ndarray):
    """Multiplicative STL seasonal: returns (seasonal_component_logspace).

    Works in log space so the STL additive seasonal exponentiates to a
    multiplicative factor. Returns the log-seasonal array aligned to logy.
    """
    series = pd.Series(logy)
    stl = STL(series, period=_PERIOD, robust=False).fit()
    return stl.seasonal.to_numpy(dtype=float)


def _backbone(y: np.ndarray, months: np.ndarray, target_month: int) -> float:
    """One-step-ahead damped-Theta + multiplicative-STL backbone.

    y       : contiguous monthly level series (sorted, >0).
    months  : month_num for each element of y.
    Returns d_hat * seasonal_factor[target_month].
    """
    n = len(y)
    logy = np.log(y)

    # --- multiplicative STL seasonal (log-additive) ---
    log_seasonal = _seasonal_factors(logy)

    # Seasonal factor for the target month: geometric mean (mean in log space)
    # of the most recent up-to-3 estimates for that calendar month.
    mask = months == target_month
    if mask.any():
        s_log = float(np.mean(log_seasonal[mask][-3:]))
    else:
        s_log = 0.0

    # Deseasonalise (in log space) then back to levels.
    d = np.exp(logy - log_seasonal)

    # --- damped-Theta backbone on the deseasonalised level ---
    # Damped Holt (additive trend, damped) is the Theta level/trend backbone.
    d_series = pd.Series(d)
    d_hat = None
    try:
        fit = ExponentialSmoothing(
            d_series,
            trend="add",
            damped_trend=True,
            seasonal=None,
            initialization_method="estimated",
        ).fit()
        phi = float(fit.params.get("damping_trend", 0.98))
        # Guard: keep damping comfortably below 1 so a spike can't drive a
        # runaway trend. statsmodels already bounds it, but cap defensively.
        if not np.isfinite(phi) or phi > 0.995:
            raise ValueError("damping too close to 1")
        d_hat = float(fit.forecast(1).iloc[0])
    except Exception:
        d_hat = None

    if d_hat is None or not np.isfinite(d_hat) or d_hat <= 0:
        # Fallback damped extrapolation: local-linear slope off last points,
        # damped by phi=0.9, on the deseasonalised series.
        w = min(6, n)
        recent = d[-w:]
        xs = np.arange(w, dtype=float)
        slope = np.polyfit(xs, recent, 1)[0]
        d_hat = d[-1] + 0.9 * slope

    # Cap an explosive one-month deseasonalised move at 10% of level.
    last = d[-1]
    cap = 0.10 * last
    step = d_hat - last
    if step > cap:
        d_hat = last + cap
    elif step < -cap:
        d_hat = last - cap

    # Re-seasonalise.
    return float(d_hat * np.exp(s_log))


def _seasonal_fallback(train: pd.DataFrame, test_row) -> float:
    m = int(test_row["month_num"])
    same = train.loc[train["month_num"] == m, "y"]
    if len(same):
        return float(same.tail(5).mean())
    return float(train["y"].tail(12).mean())


def forecast_one(train: pd.DataFrame, test_row) -> float:
    y = train["y"].to_numpy(dtype=float)
    months = train["month_num"].to_numpy(dtype=int)
    n = len(y)

    if n < _MIN_STL or np.any(y <= 0):
        return _seasonal_fallback(train, test_row)

    target_m = int(test_row["month_num"])

    # --- backbone forecast for the test month ---
    try:
        B_test = _backbone(y, months, target_m)
    except Exception:
        return _seasonal_fallback(train, test_row)

    if not np.isfinite(B_test) or B_test <= 0:
        return _seasonal_fallback(train, test_row)

    # --- in-sample backbone residual series ---
    # Reconstruct the SAME one-step backbone using only y[:k] so the OLS learns
    # the genuine one-step error structure.
    start = _MIN_STL
    if n - start < 8:
        return float(B_test)

    r = np.full(n, np.nan)
    for k in range(start, n):
        try:
            r[k] = y[k] - _backbone(y[:k], months[:k], int(months[k]))
        except Exception:
            r[k] = np.nan

    # --- AR(1) on residual, NO intercept ---
    rows_r, rows_x = [], []
    for k in range(start + 1, n):
        if np.isfinite(r[k]) and np.isfinite(r[k - 1]):
            rows_r.append(r[k])
            rows_x.append([r[k - 1]])

    if len(rows_r) < 8:
        return float(B_test)

    Xtr = np.asarray(rows_x, dtype=float)
    rtr = np.asarray(rows_r, dtype=float)
    try:
        ols = sm.OLS(rtr, Xtr).fit()  # no constant -> preserve backbone bias
    except Exception:
        return float(B_test)

    r_lag1 = r[n - 1]
    if not np.isfinite(r_lag1):
        return float(B_test)
    r_hat = float(ols.predict(np.array([[r_lag1]]))[0])

    # Safety clamp on the residual correction.
    sd = float(np.std(rtr))
    if sd > 0:
        r_hat = float(np.clip(r_hat, -2.0 * sd, 2.0 * sd))

    pred = B_test + r_hat

    # Final sanity band vs recent levels.
    recent = train["y"].tail(24)
    lo, hi = recent.min() * 0.6, recent.max() * 1.4
    if not np.isfinite(pred) or pred < lo or pred > hi:
        return float(B_test)
    return float(pred)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, "salvage")
    import btlib

    df = btlib.load_panel()
    res = btlib.walk_forward(df, forecast_one, min_train=36)
    print(btlib.metrics(res))
