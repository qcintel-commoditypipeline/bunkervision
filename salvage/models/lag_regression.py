"""
Candidate level-model: lag_regression
Family: regression on lags + month dummies.

OLS  y ~ y_lag1 + y_lag12 + C(month_num)

Rationale
---------
- y_lag12 carries the seasonal level (same month last year) and the upward
  trend (12-month-ago actual already includes most of the secular growth).
- y_lag1 adds short-run momentum / level correction.
- Month dummies absorb the residual seasonal shape that a single lag misses.

Leakage discipline
-------------------
- Fit ONLY on `train` rows (months strictly before the test month). The harness
  guarantees this by passing an expanding window.
- The lag columns (y_lag1, y_lag12) are prior actuals and are legitimate to read
  from test_row per the spec. We use ONLY SAFE_TEST_COLS from test_row.
- No same-month other-port / sg_* columns are touched.
- Refit every fold (statsmodels OLS, tiny parameter count).

Robustness
----------
- Drop rows with missing lags before fitting (early panel rows have NaN lags).
- Month dummies are aligned by reindexing to the full 1..12 set so a fold whose
  train set is missing a calendar month does not blow up the design matrix; the
  missing dummy simply gets a 0 coefficient effect (handled via fillna(0)).
- Fall back to seasonal mean of the same calendar month if anything is degenerate
  (too few rows, missing lag values on the test row, singular fit).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm


def _seasonal_fallback(train: pd.DataFrame, test_row) -> float:
    m = int(test_row["month_num"])
    same = train.loc[train["month_num"] == m, "y"]
    if len(same):
        return float(same.tail(5).mean())
    return float(train["y"].tail(12).mean())


def _design(df: pd.DataFrame) -> pd.DataFrame:
    """Build the regressor matrix: lags + month dummies (months 2..12, base=Jan)."""
    X = pd.DataFrame(index=df.index)
    X["y_lag1"] = df["y_lag1"].astype(float)
    X["y_lag12"] = df["y_lag12"].astype(float)
    mn = df["month_num"].astype(int)
    for mo in range(2, 13):  # drop month 1 as the base level -> 11 dummies
        X[f"m_{mo}"] = (mn == mo).astype(float)
    return X


def forecast_one(train: pd.DataFrame, test_row) -> float:
    # Need both lags present to fit/predict the lag-regression cleanly.
    fit_df = train.dropna(subset=["y", "y_lag1", "y_lag12"]).copy()

    # Guard: need enough data for ~13 params without absurd overfit.
    if len(fit_df) < 20:
        return _seasonal_fallback(train, test_row)

    # Test-row lags must be available (they are prior actuals -> legitimate).
    xl1 = test_row.get("y_lag1", np.nan)
    xl12 = test_row.get("y_lag12", np.nan)
    if pd.isna(xl1) or pd.isna(xl12):
        return _seasonal_fallback(train, test_row)

    X = _design(fit_df)
    y = fit_df["y"].astype(float).values
    Xc = sm.add_constant(X, has_constant="add")

    try:
        model = sm.OLS(y, Xc.values).fit()
    except Exception:
        return _seasonal_fallback(train, test_row)

    # Build the single test design row with the SAME columns as training.
    test_X = pd.DataFrame(index=[0])
    test_X["y_lag1"] = float(xl1)
    test_X["y_lag12"] = float(xl12)
    tmn = int(test_row["month_num"])
    for mo in range(2, 13):
        test_X[f"m_{mo}"] = 1.0 if tmn == mo else 0.0
    test_X = test_X[X.columns]  # enforce column order
    test_Xc = sm.add_constant(test_X, has_constant="add")
    # add_constant on a 1-row frame won't add a const if it thinks one exists;
    # force it explicitly to match the fitted param vector length.
    if test_Xc.shape[1] != Xc.shape[1]:
        test_Xc.insert(0, "const", 1.0)

    pred = float(model.predict(test_Xc.values)[0])

    # Sanity clamp: predictions outside a generous band of recent levels are
    # almost certainly an extrapolation artefact -> fall back.
    recent = train["y"].tail(24)
    lo, hi = recent.min() * 0.6, recent.max() * 1.4
    if not np.isfinite(pred) or pred < lo or pred > hi:
        return _seasonal_fallback(train, test_row)
    return pred


if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, "salvage")
    import btlib
    df = btlib.load_panel()
    res = btlib.walk_forward(df, forecast_one, min_train=36)
    print(btlib.metrics(res))
