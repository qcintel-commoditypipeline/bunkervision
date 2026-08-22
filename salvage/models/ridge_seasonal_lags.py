"""
ridge_seasonal_lags
-------------------
Regularized (ridge) level-model for the monthly bunker-demand nowcast.

Design: a SINGLE linear design matrix per fold:
    intercept + linear trend (t) + 11 month dummies + y_lag1 + y_lag12.
The month dummies are the usual overfit driver on 76 points; ridge shrinks them
toward the trend so the across-year stability improves. lambda (ridge penalty) is
chosen each fold by an expanding-window OOS CV *inside the training set* (no peek at
the test row). The intercept is never penalised; features are standardised on train
so the single penalty is comparable across columns.

Leakage discipline:
  * Fit only on `train` (months strictly before the test month).
  * From test_row only month_num, t, y_lag1, y_lag12 are read (all SAFE cols).
  * No same-month other-port columns are used.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# ---- design matrix ---------------------------------------------------------

def _design(t, month_num, y_lag1, y_lag12):
    """Build one feature row (no intercept; that's added by the solver).
    Columns: t, month dummies for months 2..12 (Jan is baseline), y_lag1, y_lag12."""
    feats = [float(t)]
    for m in range(2, 13):
        feats.append(1.0 if int(month_num) == m else 0.0)
    feats.append(float(y_lag1))
    feats.append(float(y_lag12))
    return feats


def _build_matrix(frame):
    """Return (X, y) for rows of `frame` that have both lags available."""
    X, y = [], []
    for _, r in frame.iterrows():
        l1, l12 = r["y_lag1"], r["y_lag12"]
        if pd.isna(l1) or pd.isna(l12):
            continue
        X.append(_design(r["t"], r["month_num"], l1, l12))
        y.append(float(r["y"]))
    return np.asarray(X, float), np.asarray(y, float)


# ---- ridge solve (closed form, standardised, intercept unpenalised) --------

def _fit_ridge(X, y, lam):
    """Closed-form ridge. Standardise X, don't penalise intercept.
    Returns a predictor closure pred(x_raw_row)."""
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd[sd == 0] = 1.0
    Xs = (X - mu) / sd
    ybar = y.mean()
    yc = y - ybar
    p = Xs.shape[1]
    A = Xs.T @ Xs + lam * np.eye(p)
    beta = np.linalg.solve(A, Xs.T @ yc)

    def pred(x_raw):
        xs = (np.asarray(x_raw, float) - mu) / sd
        return float(ybar + xs @ beta)

    return pred


def _cv_lambda(X, y, lambdas):
    """Expanding-window OOS CV inside train to pick lambda.
    Predict each held-out point from all earlier points; sum abs error.
    Falls back to a mid lambda when too few points."""
    n = len(y)
    if n < 14:
        return 10.0  # not enough to CV reliably; modest shrinkage
    best_lam, best_err = lambdas[0], np.inf
    start = max(8, n // 3)  # need a usable base train
    for lam in lambdas:
        err = 0.0
        cnt = 0
        for i in range(start, n):
            try:
                pr = _fit_ridge(X[:i], y[:i], lam)
                err += abs(pr(X[i]) - y[i])
                cnt += 1
            except np.linalg.LinAlgError:
                err = np.inf
                break
        if cnt and err < best_err:
            best_err, best_lam = err, lam
    return best_lam


# ---- public forecaster -----------------------------------------------------

def forecast_one(train: pd.DataFrame, test_row) -> float:
    train = train.sort_values("month")

    # test-row features (SAFE columns only)
    t = test_row["t"]
    m = test_row["month_num"]
    l1 = test_row["y_lag1"]
    l12 = test_row["y_lag12"]

    # Early folds (<~24 months) lack a full year of y_lag12 history -> the seasonal
    # design is unidentified. Fall back to a robust trailing seasonal/level blend.
    X, y = _build_matrix(train)
    have_l12 = not pd.isna(l12)
    if len(X) < 18 or not have_l12:
        # trailing-12 mean (captures trend/level) blended with same-month recent avg
        last12 = float(train["y"].tail(12).mean())
        same = train[train["month_num"] == m]["y"]
        if len(same):
            seas = float(same.tail(3).mean())
            return 0.5 * last12 + 0.5 * seas
        return last12

    lambdas = [0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0]
    lam = _cv_lambda(X, y, lambdas)
    pred = _fit_ridge(X, y, lam)

    # if either lag is missing on the test row, impute from train tails
    if pd.isna(l1):
        l1 = float(train["y"].iloc[-1])
    x = _design(t, m, l1, l12)
    yhat = pred(x)

    # guard against pathological extrapolation: clamp to a sane band around
    # recent level (no future info; uses train only).
    lo = float(train["y"].tail(24).min()) * 0.80
    hi = float(train["y"].tail(24).max()) * 1.20
    return float(min(max(yhat, lo), hi))


if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, "salvage")
    import btlib
    df = btlib.load_panel()
    res = btlib.walk_forward(df, forecast_one, min_train=36)
    print(btlib.metrics(res))
