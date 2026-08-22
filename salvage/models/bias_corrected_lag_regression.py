"""
bias_corrected_lag_regression
=============================
Family: regression on lags + month dummies, debiased.

Base estimator
--------------
OLS in LOG space of the target on two lags that between them carry both the
upward trend (y_lag1, the most recent level) and the seasonal shape (y_lag12,
the same calendar month a year earlier):

    log(y) ~ 1 + log(y_lag1) + log(y_lag12)

Working in logs makes the model multiplicative, which is the natural geometry for
a series that grows ~proportionally and has proportional seasonal swings. It also
keeps the parameter count tiny (3) for a 76-row panel, so it does not overfit.
y_lag12 plays the role of the month dummies (it IS the seasonal level), so explicit
month dummies are redundant and were dropped after testing (they did not help MAE).

Why this needs debiasing
------------------------
A raw lag regression of this shape has a small but stable NEGATIVE level bias
(the reference build measured ~ -1.6%): least squares fit in logs, then naively
exponentiated, systematically under-predicts because (a) Jensen / retransformation
and (b) the trend means the newest unseen month tends to sit slightly above the
fitted surface. The hint: remove that ~ -1.61% tilt WITHOUT inflating variance and
the MAE should fall (the family already has the best hit-rate).

Debiasing (two stacked, leakage-free corrections)
--------------------------------------------------
1. Duan smearing factor  mean(exp(residual))  — the standard non-parametric
   retransformation correction, estimated from in-sample log residuals on train.
2. A partial OUT-OF-SAMPLE recentre: we run a rolling internal backtest of the
   SAME base model on the train window, take the MEDIAN log error, and add a
   damped (0.5x) fraction of it. The damping is deliberate: a full OOS recentre
   over-corrects on noisy early folds and inflates variance (verified), whereas a
   half-weight median recentre removes most of the bias while NUDGING MAE DOWN
   rather than up.

Both corrections are computed only from `train`. Nothing same-month is read.

NO LEAKAGE
----------
Fit on `train` only. From test_row we read solely month_num and the lag columns
(all in SAFE_TEST_COLS). The same-month port/price columns are never touched.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

_LAGS = ["y_lag1", "y_lag12"]
_OOS_WEIGHT = 0.5          # damping on the OOS median-log-error recentre
_OOS_WINDOW = 36          # rolling internal-backtest length for the bias estimate
_BIAS_CLIP = 0.10         # guard: |log correction| never exceeds ~10%


def _design(d: pd.DataFrame):
    """(X, y) in log space from rows whose lags are present."""
    d = d.dropna(subset=_LAGS)
    X = np.column_stack([
        np.ones(len(d)),
        np.log(d["y_lag1"].to_numpy(dtype=float)),
        np.log(d["y_lag12"].to_numpy(dtype=float)),
    ])
    y = np.log(d["y"].to_numpy(dtype=float))
    return X, y


def _row(test_row) -> np.ndarray:
    return np.array([1.0,
                     np.log(float(test_row["y_lag1"])),
                     np.log(float(test_row["y_lag12"]))])


def _fit(train: pd.DataFrame, test_row):
    """Fit log-OLS; return (yhat_log, in_sample_residuals) or (None, None)."""
    X, y = _design(train)
    if len(y) < 8 or X.shape[0] <= X.shape[1]:
        return None, None
    lam = 1e-6 * np.trace(X.T @ X) / X.shape[1]      # tiny ridge for stability
    try:
        beta = np.linalg.solve(X.T @ X + lam * np.eye(X.shape[1]), X.T @ y)
    except np.linalg.LinAlgError:
        return None, None
    resid = y - X @ beta
    yhat = float(_row(test_row) @ beta)
    if not np.isfinite(yhat):
        return None, None
    return yhat, resid


def _oos_median_log_error(train: pd.DataFrame) -> float:
    """Rolling internal backtest of the base model on train; median log error.

    For the last ~_OOS_WINDOW feasible train months, fit the base model on the
    rows strictly before and record (log actual - log predicted). The median of
    these is a robust estimate of the model's own OOS tilt. Train-only -> no leak.
    """
    feas = train.dropna(subset=_LAGS).reset_index(drop=True)
    if len(feas) < 18:
        return 0.0
    errs = []
    start = max(12, len(feas) - _OOS_WINDOW)
    for i in range(start, len(feas)):
        sub = train[train["month"] < feas.loc[i, "month"]]
        yhat, _ = _fit(sub, feas.loc[i])
        if yhat is None:
            continue
        errs.append(float(np.log(feas.loc[i, "y"]) - yhat))
    if len(errs) < 6:
        return 0.0
    return float(np.median(errs))


def forecast_one(train: pd.DataFrame, test_row) -> float:
    yhat, resid = _fit(train, test_row)

    # Fallback for very short / infeasible train windows.
    if yhat is None:
        m = int(test_row["month_num"])
        same = train[train["month_num"] == m]["y"]
        if len(same) > 0:
            return float(same.tail(3).mean())
        return float(train["y"].tail(12).mean())

    # Correction 1: Duan smearing (retransformation debias), additive in log space.
    smear = float(np.log(np.mean(np.exp(resid))))

    # Correction 2: damped OOS median-log-error recentre (removes the residual tilt
    # without over-correcting / inflating variance).
    oos = _OOS_WEIGHT * _oos_median_log_error(train)

    correction = float(np.clip(smear + oos, -_BIAS_CLIP, _BIAS_CLIP))
    return float(np.exp(yhat + correction))


if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, "salvage")
    import btlib
    df = btlib.load_panel()
    res = btlib.walk_forward(df, forecast_one, min_train=36)
    print(btlib.metrics(res))
