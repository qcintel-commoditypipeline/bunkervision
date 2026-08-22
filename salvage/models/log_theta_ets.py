"""
Candidate level-model: log_theta_ets

Family: Theta / damped ETS in LOG space (low-parameter classical forecasters).

Idea
----
This is a direct refinement of the winning `theta_ets` candidate (MAE% 4.18,
level space). Bunker tonnage is strictly positive (~3.5M-5.5M mt) and its swing
scales with the level, so seasonality is multiplicative and forecast errors are
proportional. The bake-off metric (MAE%) is itself proportional, so a model
whose loss matches it should do better on the high-volume months that dominate.

Working in log space gives all of that natively:
  * additive structure on log(y)  <=>  multiplicative structure on y
    (multiplicative trend + multiplicative season for free, no `mul` option
     needed and no negativity / zero-protection pathologies inside the fit);
  * an additive ETS / Theta error on log(y) is a *proportional* error on y, so
    the optimiser is effectively minimising a %-style loss, aligned with MAE%.

Two low-parameter classical forecasters, both fit on log(y):
  * Theta (Assimakopoulos & Nikolopoulos) via statsmodels ThetaModel:
    deseasonalise (now ADDITIVE in log space = multiplicative in level),
    forecast the theta-line by SES + drift, re-seasonalise.
  * Damped-trend additive ETS with additive seasonality on log(y) (= damped
    multiplicative trend + multiplicative season in level space).
We average the two log forecasts, exponentiate, and apply the lognormal
half-variance bias correction  E[y] = exp(mu + 0.5*sigma^2)  using the in-sample
residual variance of the blend in log space. Averaging two classical forecasters
trims variance without adding parameters.

Design choices for a 76-point series (avoid overfit):
  * period = 12, additive-in-log deseasonalising (= multiplicative in level).
  * Theta: deseasonalize=True with the built-in seasonality significance test.
  * ETS: additive trend (damped) + additive season on log(y); estimated init.
  * Bias correction capped (sigma clipped) so a noisy short window can't inflate.
  * Blend = mean of the two log forecasts; clamp the level point to a sane band
    around the recent level so a pathological MLE fit can't emit an absurd point.
  * Seasonal trailing-mean fallback when history is too short for a seasonal fit.

LEAKAGE
-------
Only train['y'] (months strictly before the test month) feeds the fit. From
test_row nothing but the implicit "one step after train" position is used: no
test_row['y'], no same-month port columns (sg_*, fujairah/rotterdam/panama).
Refit each fold via the expanding-window harness.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

try:
    from statsmodels.tsa.forecasting.theta import ThetaModel
except Exception:  # pragma: no cover
    ThetaModel = None

try:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
except Exception:  # pragma: no cover
    ExponentialSmoothing = None


def _seasonal_fallback(train: pd.DataFrame, test_row, k: int = 3) -> float:
    """Trailing-k-year same-month mean, lightly trend-adjusted (log-space nudge).

    Captures seasonality (same calendar month) plus a half-weight nudge toward
    the recent 12-month level so the trend is not ignored. Used when there is
    not enough history for a stable seasonal fit.
    """
    m = int(test_row["month_num"])
    same = train[train["month_num"] == m]["y"]
    if len(same) == 0:
        return float(train["y"].tail(12).mean())
    seas = float(same.tail(k).mean())
    recent = float(train["y"].tail(12).mean())
    hist = float(train["y"].mean())
    if hist > 0:
        seas *= (recent / hist) ** 0.5
    return seas


def _theta_log_pred(ly: pd.Series) -> float | None:
    """One-step Theta forecast on log(y); returns the log-scale point forecast.

    (ThetaModelResults exposes no residual series, so the lognormal bias
    correction is driven solely by the ETS in-sample residual variance below;
    Theta's role here is a minority ensemble member / structural fallback.)
    """
    if ThetaModel is None:
        return None
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # Additive deseasonalising in log space == multiplicative in level.
            tm = ThetaModel(ly, period=12, deseasonalize=True,
                            use_test=True, method="additive")
            res = tm.fit()
            lpred = float(res.forecast(1).iloc[0])
        if np.isfinite(lpred):
            return lpred
    except Exception:
        return None
    return None


def _ets_log_pred(ly_vals: np.ndarray) -> tuple[float, float] | None:
    """One-step damped additive ETS on log(y); returns (log_pred, resid_var)."""
    if ExponentialSmoothing is None:
        return None
    s = pd.Series(np.asarray(ly_vals, dtype=float)).reset_index(drop=True)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = ExponentialSmoothing(
                s,
                trend="add",
                damped_trend=True,
                seasonal="add",          # additive in log == multiplicative in level
                seasonal_periods=12,
                initialization_method="estimated",
            )
            fit = model.fit(optimized=True)
            lpred = float(fit.forecast(1).iloc[0])
            try:
                resid = np.asarray(fit.resid, dtype=float)
                resid = resid[np.isfinite(resid)]
                rv = float(np.var(resid)) if resid.size > 2 else 0.0
            except Exception:
                rv = 0.0
        if np.isfinite(lpred):
            return lpred, rv
    except Exception:
        return None
    return None


def forecast_one(train: pd.DataFrame, test_row) -> float:
    """One-step-ahead log-space Theta/damped-ETS blend for the test month."""
    y_vals = np.asarray(train["y"], dtype=float)
    n = len(y_vals)

    # Need >= ~2.5 seasonal cycles and strictly positive y for a log fit.
    if n < 30 or (y_vals <= 0).any():
        return _seasonal_fallback(train, test_row)

    ly_vals = np.log(y_vals)

    # Month-start indexed log series so ThetaModel infers period/freq cleanly.
    months = pd.to_datetime(train["month"]).values
    ly = pd.Series(ly_vals, index=pd.DatetimeIndex(months, freq=None))
    try:
        ly.index = pd.DatetimeIndex(ly.index).to_period("M").to_timestamp()
        ly = ly.asfreq("MS")
    except Exception:
        ly = pd.Series(ly_vals)

    # Damped additive ETS on log(y) is the dominant member: it cleanly models
    # multiplicative trend+season and a proportional error (= the MAE% metric),
    # and on this panel it is the single strongest log-space forecaster. Theta
    # enters as a minority member (W_THETA) for ensemble robustness and as a
    # structural fallback if the ETS optimiser fails on an edge fold.
    W_ETS = 0.85

    tp = _theta_log_pred(ly)
    ep = _ets_log_pred(ly_vals)

    if ep is None and tp is None:
        return _seasonal_fallback(train, test_row)
    if ep is None:
        mu, sigma2 = tp, 0.0
    elif tp is None:
        mu, sigma2 = ep[0], ep[1]
    else:
        mu = W_ETS * ep[0] + (1.0 - W_ETS) * tp
        sigma2 = ep[1]

    # Lognormal half-variance bias correction: E[y] = exp(mu + 0.5*sigma^2),
    # driven by the ETS in-sample log residual variance. Clipped at sigma=0.10
    # (~10%) so a noisy short window cannot inflate the forecast.
    sigma2 = min(max(float(sigma2), 0.0), 0.10 ** 2)
    pred = float(np.exp(mu + 0.5 * sigma2))

    # Guard against pathological extrapolation.
    recent = float(np.mean(y_vals[-12:]))
    lo, hi = 0.5 * recent, 1.8 * recent
    pred = min(max(pred, lo), hi)
    return float(pred)


if __name__ == "__main__":
    import sys

    sys.path.insert(0, "salvage")
    import btlib

    df = btlib.load_panel()
    res = btlib.walk_forward(df, forecast_one, min_train=36)
    print(btlib.metrics(res))
