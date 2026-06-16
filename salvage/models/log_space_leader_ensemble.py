"""
Candidate level-model: log_space_leader_ensemble
Family: geometric-mean ensemble.

Idea
----
Blend the bake-off's two current champions in LOG space (a geometric mean of
their level forecasts):

  * theta_lag_ensemble -- the level-space leader (MAE% 4.01, bias -0.08). An
    arithmetic blend of theta_ets and lag_regression; reacts to the last actual
    (y_lag1) and same-month-last-year (y_lag12) via its lag-regression member.
  * log_theta_ets      -- the log-space leader (MAE% 4.07, bias -0.42). A
    damped-ETS/Theta blend fit on log(y) with a lognormal bias correction.

These two are the strongest survivors and draw on genuinely different
machinery (level-space lag regression + theta smoother vs. log-space ETS), so
their errors are only partially correlated and averaging trims variance.

Why a GEOMETRIC mean (average in log space)
-------------------------------------------
The target is strictly positive (~3.5M-5.5M mt) and the bake-off metric (MAE%)
is proportional, so the natural averaging space is log, not level. The geometric
mean  sqrt(a*b) = exp((log a + log b)/2)  is the variance-reducer that matches a
proportional loss. It is also <= the arithmetic mean (AM-GM), with the gap
growing as the two forecasts diverge -- so on folds where one champion spikes
upward, the geometric mean is automatically pulled BELOW the arithmetic blend.
That is the built-in down-weighting of upside outliers the hint asks for: no new
parameter, it falls straight out of working in log space.

Because both inputs are already near-unbiased (-0.08 and -0.42 %), their
geometric mean sits near zero bias as well; the slight downward pull of the GM
is tiny here (the two champions usually agree to within a couple of %), and is
offset by the half-variance term below, so we expect bias to stay near zero.

A small lognormal half-variance correction  *exp(0.5*sigma^2)  is added, where
sigma is the (clipped) log-gap between the two champions on THIS fold. When the
champions disagree a lot the geometric point sits low; this term nudges the
expectation back up just enough to keep the blend unbiased, and is zero when the
two agree. sigma is clipped at 0.08 so a single wild fold cannot inflate it.

Leakage discipline
------------------
This model adds NO new data access. It only calls the two champion
forecast_one functions, each of which already obeys the rules (fit on `train`
only, read only SAFE_TEST_COLS from test_row, never test_row['y'] or same-month
port/price columns). The blend is a deterministic function of their two outputs,
so it inherits their leakage-safety. Refit happens every fold via the harness.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

# Component champions live in the same directory; make it importable whether this
# file is run as a script or imported by the harness.
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import log_theta_ets        # noqa: E402  log-space champion
import theta_lag_ensemble   # noqa: E402  level-space champion

# Equal weight on the two near-unbiased champions in log space. Not fold-tuned.
_W_LEVEL = 0.5
# Clip on the per-fold log-gap used in the half-variance bias correction.
_SIGMA_CAP = 0.08


def _seasonal_fallback(train: pd.DataFrame, test_row) -> float:
    """Same-month trailing mean (trend-nudged) if a champion returns nothing."""
    m = int(test_row["month_num"])
    same = train.loc[train["month_num"] == m, "y"]
    if len(same):
        seas = float(same.tail(5).mean())
        recent = float(train["y"].tail(12).mean())
        hist = float(train["y"].mean())
        if hist > 0:
            seas *= (recent / hist) ** 0.5
        return seas
    return float(train["y"].tail(12).mean())


def forecast_one(train: pd.DataFrame, test_row) -> float:
    """One-step-ahead geometric-mean blend of the two champions (log space)."""
    logs = []
    weights = []

    # Level-space champion.
    try:
        a = float(theta_lag_ensemble.forecast_one(train, test_row))
        if np.isfinite(a) and a > 0:
            logs.append(np.log(a))
            weights.append(_W_LEVEL)
    except Exception:
        pass

    # Log-space champion.
    try:
        b = float(log_theta_ets.forecast_one(train, test_row))
        if np.isfinite(b) and b > 0:
            logs.append(np.log(b))
            weights.append(1.0 - _W_LEVEL)
    except Exception:
        pass

    if not logs:
        return _seasonal_fallback(train, test_row)

    w = np.asarray(weights, dtype=float)
    w = w / w.sum()                       # renormalise if one champion dropped out
    logs = np.asarray(logs, dtype=float)
    mu = float(np.dot(w, logs))           # geometric mean in log space

    # Lognormal half-variance correction driven by THIS fold's disagreement
    # between the champions: sigma = |log a - log b| / 2 when both present.
    # Zero when they agree; clipped so one wild fold can't inflate the point.
    if logs.size >= 2:
        sigma = 0.5 * abs(float(logs[0] - logs[1]))
    else:
        sigma = 0.0
    sigma = min(sigma, _SIGMA_CAP)
    pred = float(np.exp(mu + 0.5 * sigma * sigma))

    # Guard against pathological extrapolation (inherited sane band).
    recent = float(np.mean(np.asarray(train["y"], dtype=float)[-12:]))
    lo, hi = 0.5 * recent, 1.8 * recent
    return float(min(max(pred, lo), hi))


if __name__ == "__main__":
    sys.path.insert(0, "salvage")
    import btlib

    df = btlib.load_panel()
    res = btlib.walk_forward(df, forecast_one, min_train=36)
    print(btlib.metrics(res))
