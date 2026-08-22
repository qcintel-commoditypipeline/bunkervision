"""
Candidate level-model: log_theta_lag_hybrid

Family: Theta / damped-ETS in LOG space (backbone)
        + AR(1)/lag regression on the backbone's OOS log-residuals (correction).

Idea
----
This is the `decomp_regression_stack` recipe (backbone + AR(1)-on-residual stack)
but built on the LOG-Theta/damped-ETS backbone (`log_theta_ets`) instead of an
STL backbone, and operating in LOG space throughout.

Why a residual-correction hybrid (not a blend):
  * `log_theta_ets` is the single strongest classical backbone on this panel
    (MAE% 4.07, bias -0.42). Its one-step error still carries short-horizon,
    autocorrelated structure -- when the damped-ETS log fit over- or under-shoots
    it tends to keep doing so for a month -- that an AR(1) on the residual can
    recover. The existing `theta_lag_ensemble` is a *blend* of two independently
    biased outputs; here we instead *correct* the backbone's own error, which
    targets trend/seasonal level and short-horizon dynamics separately.
  * Working on the LOG residual keeps the correction proportional (matching the
    MAE% metric) and consistent with the backbone's multiplicative structure.

Recipe (mirrors decomp_regression_stack, in log space):
  1. Backbone L_t  = one-step log_theta_ets log-forecast for the test month
     (damped additive ETS on log y, minority Theta member, lognormal half-var
     bias correction). We keep the backbone's level forecast and recover its log.
  2. In-sample, reconstruct the SAME one-step backbone log-forecast Lhat_k for
     every train month k (using only y[:k]) and form the LOG residual
     e_k = log(y_k) - Lhat_k.
  3. Fit a tiny no-intercept AR(1) on the residual: e_t ~ e_lag1, with the slope
     RIDGE-SHRUNK toward 0 (no-intercept preserves the backbone's near-zero log
     bias; an intercept would bake the in-sample residual mean into every fcast).
  4. forecast(log) = L_test + slope * e_lag1 ;  exponentiate to level space.

Empirical finding (validated on this panel)
-------------------------------------------
Unlike the STL backbone in decomp_regression_stack (whose residuals retain AR
structure the correction recovers), the log-Theta/damped-ETS backbone already
WHITENS its own one-step log residuals: their lag-1 autocorrelation is ~ -0.1
(and no other lag carries usable persistence). So an *unshrunk* AR(1) correction
strictly injects variance, and OOS MAE% improves monotonically as the shrinkage
LAMBDA grows -- converging to the bare backbone (MAE% 4.07). We therefore ship a
heavily ridge-shrunk slope (LAMBDA=32): the residual-correction mechanism is kept
live and faithful but is regularised hard toward the backbone, which is the
correct bias-variance choice for a 76-point series. The honest takeaway is that,
for *this* backbone, the lag-residual correction adds no signal -- the value is
all in the log-Theta/ETS backbone itself.

Robustness / overfit control (76-point series):
  * Single free parameter in the correction (the ridge-shrunk AR(1) slope).
  * Std-based clamp on the correction; seasonal fallback when history is short
    or the backbone/residual fit degenerates; final sanity band vs recent level.
  * The in-sample reconstruction uses a fast damped-ETS-only backbone (no Theta
    inside the loop) so the per-fold refit stays tractable; the test-month
    backbone uses the full log_theta_ets blend.

LEAKAGE
-------
- Every fit (ETS, residual reconstruction, OLS) uses train rows only (months
  strictly before the test month; expanding window, refit each fold).
- From test_row only calendar position is used; the AR uses e_lag1 built from
  prior train actuals. Never test_row['y']; never same-month port / sg_* cols.
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


# ---- backbone pieces (shared with log_theta_ets) --------------------------

def _seasonal_fallback(train: pd.DataFrame, test_row, k: int = 3) -> float:
    """Trailing-k-year same-month mean, lightly trend-adjusted (log nudge)."""
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
                seasonal="add",
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


def _theta_log_pred(ly_vals: np.ndarray, months: np.ndarray) -> float | None:
    """One-step Theta forecast on log(y); returns the log-scale point forecast."""
    if ThetaModel is None:
        return None
    try:
        idx = pd.DatetimeIndex(pd.to_datetime(months)).to_period("M").to_timestamp()
        ly = pd.Series(np.asarray(ly_vals, dtype=float), index=idx).asfreq("MS")
    except Exception:
        ly = pd.Series(np.asarray(ly_vals, dtype=float))
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tm = ThetaModel(ly, period=12, deseasonalize=True,
                            use_test=True, method="additive")
            res = tm.fit()
            lpred = float(res.forecast(1).iloc[0])
        if np.isfinite(lpred):
            return lpred
    except Exception:
        return None
    return None


def _backbone_log(y_vals: np.ndarray, months: np.ndarray, full: bool) -> float | None:
    """One-step log_theta_ets backbone, returned in LOG space.

    full=True : full blend (damped-ETS dominant + minority Theta) -- used for the
                test-month forecast.
    full=False: damped-ETS only -- used inside the in-sample reconstruction loop
                so the per-fold refit stays tractable; structurally the same
                backbone (ETS carries weight 0.85), so the residual it produces is
                a faithful proxy of the blend's one-step error.
    """
    n = len(y_vals)
    if n < 30 or (y_vals <= 0).any():
        return None
    ly_vals = np.log(y_vals)

    ep = _ets_log_pred(ly_vals)
    if ep is None and not full:
        return None

    W_ETS = 0.85
    if full:
        tp = _theta_log_pred(ly_vals, months)
        if ep is None and tp is None:
            return None
        if ep is None:
            mu, sigma2 = tp, 0.0
        elif tp is None:
            mu, sigma2 = ep[0], ep[1]
        else:
            mu = W_ETS * ep[0] + (1.0 - W_ETS) * tp
            sigma2 = ep[1]
    else:
        mu, sigma2 = ep[0], ep[1]

    # Lognormal half-variance bias correction in log space: add 0.5*sigma^2 so
    # the exponentiated level is (approximately) mean-unbiased. Clip sigma at 0.10.
    sigma2 = min(max(float(sigma2), 0.0), 0.10 ** 2)
    return float(mu + 0.5 * sigma2)


def forecast_one(train: pd.DataFrame, test_row) -> float:
    """One-step log-Theta/ETS backbone + AR(1)-on-log-residual correction."""
    y_vals = np.asarray(train["y"], dtype=float)
    months = np.asarray(pd.to_datetime(train["month"]).values)
    n = len(y_vals)

    if n < 30 or (y_vals <= 0).any():
        return _seasonal_fallback(train, test_row)

    # --- 1) backbone log-forecast for the test month (full blend) ----------
    L_test = _backbone_log(y_vals, months, full=True)
    if L_test is None or not np.isfinite(L_test):
        return _seasonal_fallback(train, test_row)

    ly_vals = np.log(y_vals)

    # --- 2) in-sample one-step backbone log-residuals ----------------------
    # Reconstruct Lhat_k from y[:k] (ETS-only fast backbone) and form the log
    # residual e_k = log(y_k) - Lhat_k. Start once >=30 history exists.
    start = 30
    if n - start < 8:
        return float(np.exp(L_test))

    e = np.full(n, np.nan)
    for k in range(start, n):
        lk = _backbone_log(y_vals[:k], months[:k], full=False)
        if lk is not None and np.isfinite(lk):
            e[k] = ly_vals[k] - lk

    # --- 3) AR(1)-on-residual, NO intercept --------------------------------
    rows_e, rows_x = [], []
    for k in range(start + 1, n):
        if np.isfinite(e[k]) and np.isfinite(e[k - 1]):
            rows_e.append(e[k])
            rows_x.append([e[k - 1]])

    if len(rows_e) < 8:
        return float(np.exp(L_test))

    Xtr = np.asarray(rows_x, dtype=float).ravel()
    etr = np.asarray(rows_e, dtype=float)

    # No-intercept AR(1) slope = sum(x*e)/sum(x^2). We SHRINK it heavily toward 0:
    # the backbone's log residual is near-white on this panel (lag-1 autocorr
    # ~ -0.1), so an unshrunk slope just injects variance. A ridge prior
    #   slope_hat = sum(x*e) / (sum(x^2) + LAMBDA * sum(x^2)/len)  -> shrink factor
    # collapses the correction toward 0 (i.e. toward the pure backbone) unless the
    # data shows strong, stable persistence -- a single, conservative free knob.
    sxx = float(np.dot(Xtr, Xtr))
    if sxx <= 0:
        return float(np.exp(L_test))
    sxe = float(np.dot(Xtr, etr))
    LAMBDA = 32.0  # ridge strength; large -> strong shrink to backbone
    slope = sxe / (sxx + LAMBDA * (sxx / max(len(Xtr), 1)))

    # --- 4) predict log-residual for the test month ------------------------
    e_lag1 = e[n - 1]
    if not np.isfinite(e_lag1):
        return float(np.exp(L_test))
    e_hat = float(slope * e_lag1)

    # Std-based safety clamp (guards a pathological fold).
    sd = float(np.std(etr))
    if sd > 0:
        cap = 2.0 * sd
        e_hat = float(np.clip(e_hat, -cap, cap))

    pred = float(np.exp(L_test + e_hat))

    # Final sanity band vs recent level.
    recent = float(np.mean(y_vals[-12:]))
    lo, hi = 0.5 * recent, 1.8 * recent
    if not np.isfinite(pred) or pred < lo or pred > hi:
        return float(np.exp(L_test))
    return float(pred)


if __name__ == "__main__":
    import sys, os

    sys.path.insert(0, "salvage")
    import btlib

    df = btlib.load_panel()
    res = btlib.walk_forward(df, forecast_one, min_train=36)
    print(btlib.metrics(res))
