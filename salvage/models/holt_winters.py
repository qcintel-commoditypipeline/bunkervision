"""
Candidate level-model: holt_winters

Family: ExponentialSmoothing (Holt-Winters).

Idea
----
Fit a single Holt-Winters model on the train target series (monthly, ordered)
and forecast exactly one step ahead (the test month). Holt-Winters captures
BOTH the upward trend and the strong 12-month seasonality the panel exhibits,
which is the explicit weakness of the seasonal_naive / trailing-mean
benchmarks (each captures only one of the two).

Design choices for a 76-point series (avoid overfit):
  * seasonal_periods = 12 (needs >= 2 full cycles; we only forecast once we have
    enough history, otherwise we fall back to a seasonal trailing mean).
  * Damped additive trend: keeps the multi-year trend from extrapolating
    explosively on short windows.
  * Multiplicative seasonal: bunker volume scales with the level, and the
    seasonal swing grows with the trend, so a multiplicative season fits the
    ~3.5M-5.5M range better than additive.
  * Parameters (alpha/beta/gamma/phi) are estimated by statsmodels MLE on the
    train window only -> refit every fold via the harness.

LEAKAGE
-------
Only the train target `y` (months strictly before the test month) is used to
fit. From test_row nothing but the implicit "one step after train" position is
needed. No same-month port columns, no test_row['y']. Refit each fold by the
expanding-window harness.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

try:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
except Exception:  # pragma: no cover
    ExponentialSmoothing = None


def _seasonal_fallback(train: pd.DataFrame, test_row, k: int = 3) -> float:
    """Trailing-k-year same-month mean, lightly trend-adjusted.

    Used when there is not enough history for a stable HW fit. Captures
    seasonality (same calendar month) plus a small level correction toward the
    most recent 12-month mean so the trend is not ignored.
    """
    m = int(test_row["month_num"])
    same = train[train["month_num"] == m]["y"]
    if len(same) == 0:
        return float(train["y"].tail(12).mean())
    seas = float(same.tail(k).mean())
    # nudge toward recent level to follow trend
    recent = float(train["y"].tail(12).mean())
    hist = float(train["y"].mean())
    if hist > 0:
        seas *= (recent / hist) ** 0.5  # half-weight trend drift
    return seas


def forecast_one(train: pd.DataFrame, test_row) -> float:
    """One-step-ahead Holt-Winters forecast for the test month.

    Fits ExponentialSmoothing on the train target only and returns the first
    forecast step. Falls back to a seasonal trailing mean when history is too
    short for a reliable seasonal fit.
    """
    y = pd.Series(np.asarray(train["y"], dtype=float))
    y = y.reset_index(drop=True)
    n = len(y)

    # Need at least 2 full seasonal cycles + a little headroom for a stable fit.
    if ExponentialSmoothing is None or n < 30 or (y <= 0).any():
        return _seasonal_fallback(train, test_row)

    candidates = []
    # Multiplicative seasonal is the primary; additive as a robustness backstop.
    configs = [
        dict(trend="add", damped_trend=True, seasonal="mul"),
        dict(trend="add", damped_trend=True, seasonal="add"),
        dict(trend="add", damped_trend=False, seasonal="mul"),
    ]
    for cfg in configs:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = ExponentialSmoothing(
                    y,
                    trend=cfg["trend"],
                    damped_trend=cfg["damped_trend"],
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
        return _seasonal_fallback(train, test_row)

    # Pick the configuration with the best (lowest) AIC on the train window.
    candidates.sort(key=lambda c: c[0])
    pred = candidates[0][1]

    # Guard against pathological extrapolation: clamp to a sane band around the
    # recent level so a bad MLE fit can't produce an absurd point forecast.
    recent = float(y.tail(12).mean())
    lo, hi = 0.5 * recent, 1.8 * recent
    pred = min(max(pred, lo), hi)
    return float(pred)


if __name__ == "__main__":
    import os
    import sys

    sys.path.insert(0, "salvage")
    import btlib

    df = btlib.load_panel()
    res = btlib.walk_forward(df, forecast_one, min_train=36)
    print(btlib.metrics(res))
