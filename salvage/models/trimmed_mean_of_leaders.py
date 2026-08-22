"""
Candidate level-model: trimmed_mean_of_leaders

Family: robust ensemble (combination forecast).

Idea
----
Four independent, individually-sound one-step forecasters each capture the
trend+seasonality of the official Singapore monthly bunker TOTAL in a different
way, and -- crucially -- their biases partially cancel:

    theta_ets         bias ~ -0.5   (Theta / damped-ETS blend)
    sarima            bias ~ +0.1   (log SARIMA (1,1,1)(1,1,0,12))
    lag_regression    bias ~ -1.6   (OLS on y_lag1 + y_lag12 + month dummies)
    seasonal_x_trend  bias ~ -2.1   (trailing-12 level x multiplicative season)

Two are near-zero-biased, two are mildly negative. No single one is reliably the
best year-to-year (the league table reshuffles by OOS year, and the project's
"phantom-event" volatility makes any single engine's worst-case fragile). A
robust central estimator across the panel of forecasts is therefore the best
*year-robust* hedge: it cannot be dragged by one member's bad fold, and the
member biases partially cancel.

Combination rule
----------------
We collect the point forecasts of the members that returned a finite value, then
take a robust centre:
  * >= 4 members -> trimmed mean: drop the single lowest and single highest,
    average the remaining 2 (equivalent to the mean of the two interior values).
    This rejects the most extreme member each fold (the usual cause of a blown
    fold) while keeping more information than the bare median.
  * 3 members   -> median (the natural robust centre of 3).
  * 1-2 members -> plain mean.
  * 0 members   -> seasonal fallback.

This adds ZERO fitted parameters of its own (equal weights, fixed trimming rule),
so there is no extra overfit on the 76-point panel -- the only "learning" is in
the constituent models, each refit per fold by the harness.

Leakage discipline
------------------
Every member obeys the harness contract: fit only on `train` (months strictly
before the test month) and read only SAFE_TEST_COLS from `test_row` (month,
year, month_num, t, y_lag1, y_lag3, y_lag6, y_lag12). No same-month port / sg_*
columns are used. The ensemble merely averages member outputs, so it inherits
their non-leakiness. The member implementations are imported from their own
files; if an import fails the member is silently skipped (and the trimmed-mean /
median rule adapts to the available count), so the ensemble can never crash a
fold.
"""
from __future__ import annotations

import importlib
import warnings

import numpy as np
import pandas as pd

warnings.simplefilter("ignore")

# Member models, in a fixed order. Each is loaded lazily and cached.
_MEMBER_NAMES = ["theta_ets", "sarima", "lag_regression", "seasonal_x_trend"]
_MEMBERS: list = []


def _load_members() -> list:
    """Import each member's forecast_one once and cache the callables."""
    global _MEMBERS
    if _MEMBERS:
        return _MEMBERS
    fns = []
    for name in _MEMBER_NAMES:
        fn = None
        for modpath in (f"salvage.models.{name}", f"models.{name}", name):
            try:
                mod = importlib.import_module(modpath)
                fn = getattr(mod, "forecast_one", None)
                if fn is not None:
                    break
            except Exception:
                continue
        if fn is not None:
            fns.append(fn)
    _MEMBERS = fns
    return _MEMBERS


def _seasonal_fallback(train: pd.DataFrame, test_row) -> float:
    """Trailing-3-year same-month mean; else trailing-12 level."""
    m = int(test_row["month_num"])
    same = train.loc[train["month_num"] == m, "y"]
    if len(same):
        return float(same.tail(3).mean())
    return float(train["y"].tail(12).mean())


def _robust_centre(preds: list[float]) -> float:
    """Robust central estimator over the member point forecasts."""
    arr = np.array([p for p in preds if np.isfinite(p)], dtype=float)
    k = len(arr)
    if k == 0:
        return float("nan")
    if k == 1:
        return float(arr[0])
    if k == 2:
        return float(arr.mean())
    if k == 3:
        return float(np.median(arr))
    # k >= 4: symmetric trimmed mean -- drop the single min and single max,
    # average the interior. Rejects the most extreme member this fold.
    arr_sorted = np.sort(arr)
    interior = arr_sorted[1:-1]
    return float(interior.mean())


def forecast_one(train: pd.DataFrame, test_row) -> float:
    members = _load_members()

    preds: list[float] = []
    for fn in members:
        try:
            p = float(fn(train, test_row))
            if np.isfinite(p) and p > 0:
                preds.append(p)
        except Exception:
            continue

    centre = _robust_centre(preds)
    if not np.isfinite(centre) or centre <= 0:
        return _seasonal_fallback(train, test_row)

    # Final sanity clamp around the recent level (cheap defence in depth; the
    # members already clamp, so this almost never binds).
    recent = float(np.asarray(train["y"], dtype=float)[-12:].mean())
    lo, hi = 0.5 * recent, 1.8 * recent
    return float(min(max(centre, lo), hi))


if __name__ == "__main__":
    import sys
    import os

    sys.path.insert(0, "salvage")
    # Make 'salvage.models.<name>' importable from the repo root.
    sys.path.insert(0, os.getcwd())
    import btlib

    df = btlib.load_panel()
    res = btlib.walk_forward(df, forecast_one, min_train=36)
    print(btlib.metrics(res))
