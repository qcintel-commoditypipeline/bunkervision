"""
Candidate level-model: bias_constrained_inverse_mae_ensemble

Family: constrained weighted ensemble (combination forecast).

Idea
----
The single best classical forecasters on this panel are individually good but
*almost all under-call*: theta_lag_ensemble (-0.08), holt_winters (-0.47),
log_theta_ets (-0.42), theta_ets (-0.53), lag_regression (-1.61),
seasonal_x_trend (-2.11). Only a couple lean the other way -- sarima (+0.11)
and decomp_regression_stack (+0.36). A plain equal-weight blend of survivors
(as in `trimmed_mean_of_leaders`) therefore inherits a net *negative* bias of
order -1%, leaving systematic MAE% on the table.

This model exploits that structure explicitly. It blends a small panel of
sound, structurally-diverse members with weights that are

  (1) proportional to each member's INVERSE out-of-sample MAE  (accuracy), and
  (2) tilted so the blend's net bias is driven toward ZERO  (the constraint),
      by up-weighting the positive/zero-bias members against the dominant
      negative-bias crowd.

Member panel (5), chosen for accuracy AND bias diversity so cancellation is
possible at all:

    holt_winters             MAE 4.04  bias -0.47   (damped HW, level smoother)
    log_theta_ets            MAE 4.07  bias -0.42   (log Theta/ETS blend)
    lag_regression           MAE 4.35  bias -1.61   (OLS on lags + month dummies)
    sarima                   MAE 4.52  bias +0.11   (log SARIMA, +ve bias)
    decomp_regression_stack  MAE 4.26  bias +0.36   (decomp stack, +ve bias)

The two near-zero / positive-bias members (sarima, decomp_regression_stack)
exist specifically to cancel the negative pull of the regression-style members.

How the weights are learned (NO LEAKAGE, NO MANUAL TUNING)
---------------------------------------------------------
Weights are NOT hand-set and NOT read off the public league table (that would be
fitting on the test set). For each outer fold they are solved from an INNER
expanding-window backtest that lives entirely inside `train` (months strictly
before the test month):

  * For every member m and every inner month j we one-step-forecast j using only
    train.iloc[:j] and score the signed % error. (j, m) errors are cached
    globally and reused across outer folds -- the same inner point yields the
    same error regardless of which outer fold references it -- so the whole run
    costs ~O(members x months) member fits, not O(folds x window x members).

  * Over the most recent inner window we form each member's inner MAE_m and
    inner signed-bias b_m. Base accuracy weights w0_m proportional to
    1/(MAE_m + eps), normalised. Then a single scalar bias-cancellation tilt is
    applied in closed form: we shift weight from the members on the heavy side
    of zero toward the light side just enough to bring the weighted bias
    sum_m w_m b_m to (near) zero, subject to non-negativity. This adds exactly
    ONE degree of freedom (the tilt magnitude), solved analytically -- not a
    free per-member parameter vector -- so overfit on 76 points stays minimal.

  * Until enough inner history exists for a stable solve, the blend falls back
    to pure inverse-MAE weights, then to equal weights, then to a seasonal
    estimate -- the model degrades gracefully and never crashes a fold.

Why this beats trimmed_mean_of_leaders
--------------------------------------
Trimmed-mean equal-weights its survivors and so *inherits* the ~-1.16 negative
bias of the crowd. Here the bias is an explicit constraint: the positive-bias
members are deliberately up-weighted to neutralise it, which removes the
systematic component of MAE% rather than averaging it.

Leakage discipline
------------------
Adds NO new data access. It only calls member forecast_one functions, each of
which already (a) fits solely on `train`, (b) reads from test_row only
SAFE_TEST_COLS (month/year/month_num/t and the y_lag* columns), and (c) never
touches test_row['y'] or any same-month port/price column. The inner backtest
used to learn the weights is run strictly on `train` rows, so no test-month
information ever enters the weights. The blend is a deterministic function of
member outputs and inner-train errors, so it inherits their leakage-safety.
Members are refit every fold by the harness.
"""
from __future__ import annotations

import importlib
import warnings

import numpy as np
import pandas as pd

warnings.simplefilter("ignore")

# ---- Member panel -----------------------------------------------------------
# Ordered; each loaded lazily and cached. Chosen for accuracy AND bias sign
# diversity (negative-bias regressors + positive-bias sarima/decomp) so the
# zero-bias constraint is actually attainable by re-weighting.
_MEMBER_NAMES = [
    "holt_winters",
    "log_theta_ets",
    "lag_regression",
    "sarima",
    "decomp_regression_stack",
]
_MEMBERS: list = []

# Global cache of inner one-step errors:
#   _ERR_CACHE[(member_index, inner_month_index)] = (signed_pct, abs_pct)
# Keyed on the absolute integer position of the inner test month within the
# full panel order, which the harness preserves (chronological). The same
# (member, month) reproduces the same error in every outer fold that sees it,
# so caching is exact, not an approximation.
_ERR_CACHE: dict = {}
# Cache of member point forecasts for the OUTER test month, keyed by
# (member_index, train_length) so we never refit a member twice per fold.
_PRED_CACHE: dict = {}

_EPS = 1e-9
# Most recent inner months used to estimate per-member MAE / bias.
_INNER_WINDOW = 24
# Minimum inner observations before we trust an inverse-MAE / bias solve.
_MIN_INNER = 8


def _load_members() -> list:
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
        fns.append(fn)  # may be None; handled at call sites
    _MEMBERS = fns
    return _MEMBERS


def _member_pred(mi: int, fn, train: pd.DataFrame, test_row, cache_key=None):
    """One member's one-step point forecast, cached by (member, train_len)."""
    if fn is None:
        return np.nan
    key = (mi, cache_key) if cache_key is not None else None
    if key is not None and key in _PRED_CACHE:
        return _PRED_CACHE[key]
    try:
        p = float(fn(train, test_row))
        if not (np.isfinite(p) and p > 0):
            p = np.nan
    except Exception:
        p = np.nan
    if key is not None:
        _PRED_CACHE[key] = p
    return p


def _inner_error(mi: int, fn, panel: pd.DataFrame, j: int):
    """Signed/abs % error of member mi forecasting panel row j one-step, using
    only panel.iloc[:j]. Cached globally on (member, absolute month index j)."""
    ck = (mi, j)
    if ck in _ERR_CACHE:
        return _ERR_CACHE[ck]
    if fn is None:
        _ERR_CACHE[ck] = None
        return None
    tr = panel.iloc[:j]
    row = panel.iloc[j]
    try:
        p = float(fn(tr, row))
        y = float(row["y"])
        if np.isfinite(p) and p > 0 and y > 0:
            signed = (p / y - 1.0) * 100.0
            res = (signed, abs(signed))
        else:
            res = None
    except Exception:
        res = None
    _ERR_CACHE[ck] = res
    return res


def _inner_stats(members, train: pd.DataFrame):
    """For each member, inner MAE and signed bias over the most recent inner
    window of `train`. Inner test points are the last _INNER_WINDOW months of
    `train` (but never earlier than month index 30, so each member still has
    >=30 points of history to fit on, matching the outer protocol). Each inner
    point is forecast using only earlier train rows. Returns (mae[],bias[],cnt[])."""
    n = len(train)
    k = len(members)
    start = max(30, n - _INNER_WINDOW)
    js = list(range(start, n))  # inner test month indices within `train`
    mae = np.full(k, np.nan)
    bias = np.full(k, np.nan)
    cnt = np.zeros(k, dtype=int)
    for mi, fn in enumerate(members):
        signed_list = []
        abs_list = []
        for j in js:
            r = _inner_error(mi, fn, train, j)
            if r is not None:
                signed_list.append(r[0])
                abs_list.append(r[1])
        if len(abs_list) > 0:
            mae[mi] = float(np.mean(abs_list))
            bias[mi] = float(np.mean(signed_list))
            cnt[mi] = len(abs_list)
    return mae, bias, cnt


def _solve_weights(mae, bias, cnt):
    """Inverse-MAE base weights, then a closed-form bias-cancellation tilt.

    Step 1: w0_m ∝ 1/(MAE_m+eps) over members with enough inner obs.
    Step 2: tilt one scalar amount of mass from the heavy bias side toward the
            light side to drive sum_m w_m*bias_m -> 0, staying non-negative.
    Only ONE free quantity (the tilt) is solved, analytically -- low overfit."""
    valid = (cnt >= _MIN_INNER) & np.isfinite(mae) & np.isfinite(bias) & (mae > 0)
    if valid.sum() < 2:
        return None  # caller falls back

    idx = np.where(valid)[0]
    inv = 1.0 / (mae[idx] + _EPS)
    w = inv / inv.sum()
    b = bias[idx]

    net = float(np.dot(w, b))  # current weighted bias

    # Partition members by which side of zero their bias sits.
    neg = b < 0.0   # under-callers
    pos = b > 0.0   # over-callers
    if net < 0:
        # Need to add positive bias: move mass from most-negative members to
        # positive-bias members. Donors = neg side, receivers = pos side.
        donors, receivers = neg, pos
    else:
        donors, receivers = pos, neg

    if donors.sum() == 0 or receivers.sum() == 0:
        # No counterweight available; keep pure inverse-MAE weights.
        ww = np.zeros(len(mae))
        ww[idx] = w
        return ww

    # Receivers get extra mass proportional to their (accuracy) weight and to
    # how strongly they pull the bias the right way; donors give it up in the
    # same proportions. Solve scalar alpha s.t. weighted bias hits zero, then
    # clamp alpha to [0,1] so no weight goes negative.
    # New weight: w_m' = w_m * (1 - alpha)  for donors,
    #             w_m' = w_m + alpha * (w_donor_total) * share_m  for receivers.
    w_don_tot = float(w[donors].sum())
    recv_w = w[receivers]
    recv_share = recv_w / recv_w.sum()

    # bias as a function of alpha:
    #   B(alpha) = sum_don w*(1-alpha)*b + sum_recv (w + alpha*w_don_tot*share)*b
    #            = net - alpha*( sum_don w*b )  + alpha*w_don_tot*sum_recv share*b
    don_bias_mass = float(np.dot(w[donors], b[donors]))
    recv_bias_per_alpha = w_don_tot * float(np.dot(recv_share, b[receivers]))
    dB = -don_bias_mass + recv_bias_per_alpha  # dB/dalpha
    if abs(dB) < _EPS:
        ww = np.zeros(len(mae))
        ww[idx] = w
        return ww
    alpha = -net / dB
    alpha = float(min(max(alpha, 0.0), 1.0))

    w_new = w.copy()
    w_new[donors] = w[donors] * (1.0 - alpha)
    w_new[receivers] = w[receivers] + alpha * w_don_tot * recv_share
    s = w_new.sum()
    if s <= 0:
        w_new = w
    else:
        w_new = w_new / s

    ww = np.zeros(len(mae))
    ww[idx] = w_new
    return ww


def _seasonal_fallback(train: pd.DataFrame, test_row, k: int = 3) -> float:
    m = int(test_row["month_num"])
    same = train.loc[train["month_num"] == m, "y"]
    if len(same):
        return float(same.tail(k).mean())
    return float(train["y"].tail(12).mean())


def forecast_one(train: pd.DataFrame, test_row) -> float:
    members = _load_members()
    n = len(train)
    cache_key = n  # train length uniquely identifies this outer fold's history

    # 1) Member point forecasts for the outer test month.
    preds = np.array(
        [_member_pred(mi, fn, train, test_row, cache_key)
         for mi, fn in enumerate(members)],
        dtype=float,
    )

    # 2) Learn weights from an inner backtest on `train` only.
    weights = None
    mae = bias = cnt = None
    if n >= 30 + _MIN_INNER:
        mae, bias, cnt = _inner_stats(members, train)
        weights = _solve_weights(mae, bias, cnt)
        if weights is None:
            # Pure inverse-MAE fallback if the constrained solve isn't available.
            valid = np.isfinite(mae) & (mae > 0) & (cnt >= 3)
            if valid.sum() >= 1:
                inv = np.where(valid, 1.0 / (mae + _EPS), 0.0)
                if inv.sum() > 0:
                    weights = inv / inv.sum()

    # 3) Combine. Re-normalise over members that actually produced a forecast.
    finite = np.isfinite(preds) & (preds > 0)
    if finite.sum() == 0:
        return _seasonal_fallback(train, test_row)

    if weights is None:
        w = np.where(finite, 1.0, 0.0)  # equal weight
    else:
        w = np.where(finite, weights, 0.0)
        if w.sum() <= 0:
            w = np.where(finite, 1.0, 0.0)
    w = w / w.sum()

    blend = float(np.dot(w, np.where(finite, preds, 0.0)))
    if not (np.isfinite(blend) and blend > 0):
        return _seasonal_fallback(train, test_row)

    # 4) Residual-bias correction (completes the near-zero-bias constraint).
    # The re-weighting in _solve_weights can only cancel bias when BOTH signs are
    # present among members; in early folds every member under-calls (no positive
    # counterweight exists) and a residual negative bias survives. Here we remove
    # whatever weighted inner bias remains under the FINAL weights via a single
    # multiplicative factor estimated purely on inner-train OOS errors:
    #     factor = 1 / (1 + net_inner_bias/100)
    # The net bias is shrunk 0.6x and capped to +/-2.5% so a noisy short inner
    # window cannot over-correct. This is the analogue of the constraint applied
    # directly to the blend, and uses no test-month information (bias[] comes
    # only from train rows).
    if bias is not None and cnt is not None:
        bw = np.where(finite & np.isfinite(bias) & (cnt >= _MIN_INNER), w, 0.0)
        bw_sum = bw.sum()
        if bw_sum > 0:
            net_bias = float(np.dot(bw / bw_sum, np.where(np.isfinite(bias), bias, 0.0)))
            # Shrink the correction toward zero (0.6x): the inner bias estimate
            # is noisy on short windows, and when both bias signs are present the
            # re-weighting has already neutralised most of it, so only a partial
            # residual nudge is warranted. Cap at +/-2.5% as a final guard.
            net_bias *= 0.6
            net_bias = float(min(max(net_bias, -2.5), 2.5))
            denom = 1.0 + net_bias / 100.0
            if denom > 0.5:
                blend = blend / denom

    # Defence-in-depth clamp around the recent level.
    recent = float(np.asarray(train["y"], dtype=float)[-12:].mean())
    lo, hi = 0.5 * recent, 1.8 * recent
    return float(min(max(blend, lo), hi))


if __name__ == "__main__":
    import sys
    import os

    sys.path.insert(0, "salvage")
    sys.path.insert(0, os.getcwd())
    import btlib

    df = btlib.load_panel()
    res = btlib.walk_forward(df, forecast_one, min_train=36)
    print(btlib.metrics(res))
