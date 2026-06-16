"""
Candidate level-model: theta_ets

Family: Theta / damped ETS (low-parameter classical forecasters).

Idea
----
The benchmarks each capture only half of the signal: seasonal_naive nails the
12-month seasonality but ignores the upward trend (bias -8.3), while the
trailing-12-mean tracks the level/trend but ignores seasonality. The Theta
method (Assimakopoulos & Nikolopoulos) is the bake-off-classic that does BOTH
with very few parameters: it deseasonalises the series (multiplicative,
period 12), forecasts the long-term theta-line by SES + drift, and re-applies
the seasonal factors. statsmodels' ThetaModel implements exactly this and is
robust on short series.

We pair the Theta forecast with a damped-trend ETS forecast (also low-param,
also trend+season aware) and average the two. Averaging two unbiased-ish
classical forecasters reduces variance without adding parameters, and the blend
is more stable fold-to-fold than either alone. Both are MLE/closed-form fitted
on the train window only and refit every fold by the expanding-window harness.

Design choices for a 76-point series (avoid overfit):
  * period = 12, multiplicative deseasonalising (swing scales with level in the
    ~3.5M-5.5M range).
  * Theta: deseasonalize=True, with a significance test so a flat series isn't
    over-seasonalised on very short windows.
  * ETS: additive damped trend + multiplicative season (same rationale as the
    HW candidate), AIC-selected against an additive-season backstop.
  * Blend = mean of the two point forecasts; clamp to a sane band around the
    recent level so a pathological MLE fit can't emit an absurd point.
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
    """Trailing-k-year same-month mean, lightly trend-adjusted.

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


def _theta_pred(y: pd.Series) -> float | None:
    """One-step Theta forecast on a monthly series indexed by month-start."""
    if ThetaModel is None:
        return None
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tm = ThetaModel(y, period=12, deseasonalize=True,
                            use_test=True, method="mul")
            res = tm.fit()
            pred = float(res.forecast(1).iloc[0])
        if np.isfinite(pred) and pred > 0:
            return pred
    except Exception:
        return None
    return None


def _ets_pred(y_vals: np.ndarray) -> float | None:
    """One-step damped-ETS forecast; AIC-selects mul vs add season."""
    if ExponentialSmoothing is None:
        return None
    s = pd.Series(np.asarray(y_vals, dtype=float)).reset_index(drop=True)
    candidates = []
    configs = [
        dict(seasonal="mul"),
        dict(seasonal="add"),
    ]
    for cfg in configs:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = ExponentialSmoothing(
                    s,
                    trend="add",
                    damped_trend=True,
                    seasonal=cfg["seasonal"],
                    seasonal_periods=12,
                    initialization_method="estimated",
                )
                fit = model.fit(optimized=True)
                pred = float(fit.forecast(1).iloc[0])
                if np.isfinite(pred) and pred > 0:
                    candidates.append((fit.aic, pred))
        except Exception:
            continue
    if not candidates:
        return None
    candidates.sort(key=lambda c: c[0])
    return candidates[0][1]


def forecast_one(train: pd.DataFrame, test_row) -> float:
    """One-step-ahead Theta/damped-ETS blend for the test month."""
    y_vals = np.asarray(train["y"], dtype=float)
    n = len(y_vals)

    # Need >= 2 full seasonal cycles + headroom for a stable seasonal fit.
    if n < 30 or (y_vals <= 0).any():
        return _seasonal_fallback(train, test_row)

    # Build a month-start indexed series so ThetaModel infers period/freq cleanly.
    months = pd.to_datetime(train["month"]).values
    y = pd.Series(y_vals, index=pd.DatetimeIndex(months, freq=None))
    try:
        y.index = pd.DatetimeIndex(y.index).to_period("M").to_timestamp()
        y = y.asfreq("MS")
    except Exception:
        y = pd.Series(y_vals)

    preds = []
    tp = _theta_pred(y)
    if tp is not None:
        preds.append(tp)
    ep = _ets_pred(y_vals)
    if ep is not None:
        preds.append(ep)

    if not preds:
        return _seasonal_fallback(train, test_row)

    pred = float(np.mean(preds))

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
