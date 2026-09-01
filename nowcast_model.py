"""
Calibrated demand nowcast (salvage prototype) — OFF by default.

Replaces the structurally-broken duration×pump-rate "measurement" with a calibrated
forecast of the OFFICIAL monthly demand series, optionally nudged by an AIS leading
indicator that earns weight only as graded months accrue:

    nowcast_mt(month) = level_model(month) × (1 + α · z_ais)

WHY this exists
---------------
The old engine summed AIS event-duration × a fixed rate. Investigation (2026-06)
showed ~85% of that "tonnage" came from implausible long-dwell events and the
headline only matched MPA via two cancelling errors; there is no independent AIS
stem-size signal (draught is a dead static field). So absolute tonnes cannot be
*measured* from AIS. They can, however, be *nowcast*: the 76-month official series
has strong seasonality + trend and is highly forecastable.

`level_model` — the winner of the 2026-06 out-of-sample bake-off: a log-space blend
of a damped-additive ETS (dominant) and a Theta model, with a lognormal bias
correction. Out-of-sample (expanding-window, 40 folds 2023-2026): MAE 4.07%, bias
-0.42% — vs 8.35% / -8.28% for the seasonal baseline the dashboard uses today.
Ported verbatim from salvage/models/log_theta_ets.py so the live path is exactly
what was validated.

`z_ais` — a standardized deviation of a CLEAN AIS feature (the reliable 30min-6h
stem rate, never the phantom long-dwell tonnage) from its own trailing baseline:
"is this month running hotter or colder than normal?".

`α` — shrinkage coefficient. With only ~2 months of AIS history α ≈ 0 today, so the
nowcast degrades gracefully to the level model. α grows toward NOWCAST_AIS_MAX_ALPHA
only after NOWCAST_AIS_MIN_GRADED_MONTHS of AIS history have been graded against
official figures (out-of-sample evidence the signal helps).

This is a CALIBRATED NOWCAST WITH ERROR BANDS, not an independent measurement.
"""
from __future__ import annotations

import warnings
from datetime import date

import numpy as np
import pandas as pd

import db
from config import (
    NOWCAST_AIS_MAX_ALPHA,
    NOWCAST_AIS_MIN_GRADED_MONTHS,
    NOWCAST_LEVEL_MIN_TRAIN,
)

try:
    from statsmodels.tsa.forecasting.theta import ThetaModel
except Exception:  # pragma: no cover
    ThetaModel = None
try:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
except Exception:  # pragma: no cover
    ExponentialSmoothing = None


def _db_port(port: str) -> str:
    from config import PORTS
    return PORTS.get(port, {}).get("db_port", port)


# ── Official monthly series ─────────────────────────────────────────────────

def official_series(port: str, before_year: int, before_month: int) -> pd.DataFrame:
    """Official monthly TOTAL series for `port`, STRICTLY BEFORE (before_year,
    before_month), chronological. Mirrors accuracy/demand_model lookup:
    port_bunker_sales, with mpa_monthly fallback for Singapore."""
    rows = db.query("""
        SELECT month, volume_mt FROM port_bunker_sales
        WHERE port = ? AND fuel_type = 'TOTAL' AND volume_mt IS NOT NULL
        ORDER BY month
    """, [_db_port(port)])
    if not rows and port == "singapore":
        rows = db.query("""
            SELECT month, volume_mt FROM mpa_monthly
            WHERE fuel_type = 'TOTAL' AND volume_mt IS NOT NULL
            ORDER BY month
        """)
    df = pd.DataFrame(rows, columns=["month", "y"])
    if df.empty:
        return df
    df["month"] = pd.to_datetime(df["month"])
    cutoff = pd.Timestamp(date(before_year, before_month, 1))
    return df[df["month"] < cutoff].reset_index(drop=True)


# ── Level model (ported from salvage/models/log_theta_ets.py) ───────────────

def _seasonal_fallback(df: pd.DataFrame, target_month: int, k: int = 3) -> float:
    same = df[df["month"].dt.month == target_month]["y"]
    if len(same) == 0:
        return float(df["y"].tail(12).mean())
    seas = float(same.tail(k).mean())
    recent = float(df["y"].tail(12).mean())
    hist = float(df["y"].mean())
    if hist > 0:
        seas *= (recent / hist) ** 0.5
    return seas


def _theta_log_pred(ly: pd.Series) -> float | None:
    if ThetaModel is None:
        return None
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tm = ThetaModel(ly, period=12, deseasonalize=True, use_test=True, method="additive")
            lpred = float(tm.fit().forecast(1).iloc[0])
        return lpred if np.isfinite(lpred) else None
    except Exception:
        return None


def _ets_log_pred(ly_vals: np.ndarray) -> tuple[float, float] | None:
    if ExponentialSmoothing is None:
        return None
    s = pd.Series(np.asarray(ly_vals, dtype=float)).reset_index(drop=True)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit = ExponentialSmoothing(
                s, trend="add", damped_trend=True, seasonal="add",
                seasonal_periods=12, initialization_method="estimated",
            ).fit(optimized=True)
            lpred = float(fit.forecast(1).iloc[0])
            resid = np.asarray(fit.resid, dtype=float)
            resid = resid[np.isfinite(resid)]
            rv = float(np.var(resid)) if resid.size > 2 else 0.0
        return (lpred, rv) if np.isfinite(lpred) else None
    except Exception:
        return None


def level_forecast(df: pd.DataFrame, target_month: int) -> float:
    """One-step-ahead log-space damped-ETS/Theta blend. `df` is the official series
    strictly before the target month; `target_month` is the calendar month (1-12)."""
    y_vals = np.asarray(df["y"], dtype=float) if len(df) else np.array([])
    n = len(y_vals)
    if n < NOWCAST_LEVEL_MIN_TRAIN or (y_vals <= 0).any():
        return _seasonal_fallback(df, target_month) if n else 0.0

    ly_vals = np.log(y_vals)
    ly = pd.Series(ly_vals)
    try:
        idx = pd.DatetimeIndex(pd.to_datetime(df["month"]).values).to_period("M").to_timestamp()
        ly = pd.Series(ly_vals, index=idx).asfreq("MS")
    except Exception:
        pass

    W_ETS = 0.85
    tp = _theta_log_pred(ly)
    ep = _ets_log_pred(ly_vals)
    if ep is None and tp is None:
        return _seasonal_fallback(df, target_month)
    if ep is None:
        mu, sigma2 = tp, 0.0
    elif tp is None:
        mu, sigma2 = ep[0], ep[1]
    else:
        mu, sigma2 = W_ETS * ep[0] + (1.0 - W_ETS) * tp, ep[1]

    sigma2 = min(max(float(sigma2), 0.0), 0.10 ** 2)
    pred = float(np.exp(mu + 0.5 * sigma2))
    recent = float(np.mean(y_vals[-12:]))
    return float(min(max(pred, 0.5 * recent), 1.8 * recent))


# ── AIS deviation signal + shrinkage weight ─────────────────────────────────

def _ais_monthly_clean(port: str, before_year: int, before_month: int) -> pd.DataFrame:
    """Monthly CLEAN-stem features (30min-6h band only) for completed months
    strictly before the target. The reliable AIS band — never the phantom tail."""
    rows = db.query("""
        SELECT date_trunc('month', start_ts) AS m,
               SUM(CASE WHEN duration_min>=30 AND duration_min<360 THEN 1 ELSE 0 END) AS clean_stems,
               COUNT(DISTINCT bunker_mmsi) AS active_bunkers
        FROM bunkering_events
        WHERE port = ? AND open = FALSE
        GROUP BY 1 ORDER BY 1
    """, [port])
    df = pd.DataFrame(rows, columns=["month", "clean_stems", "active_bunkers"])
    if df.empty:
        return df
    df["month"] = pd.to_datetime(df["month"])
    cutoff = pd.Timestamp(date(before_year, before_month, 1))
    return df[df["month"] < cutoff].reset_index(drop=True)


def ais_deviation(port: str, year: int, month: int) -> tuple[float, int]:
    """Relative deviation of the clean-stem rate from its trailing baseline.
    Returns (dev, n_history_months). dev = (current - baseline) / baseline,
    clipped to ±3. Returns (0.0, n) when there is no prior month to baseline on.
    The gating on whether this deviation should influence the nowcast is handled
    separately by ais_signal.live_coupling (verdict-based, not count-based)."""
    hist = _ais_monthly_clean(port, year, month)
    n = len(hist)
    if n < 2:
        return 0.0, n
    vals = hist["clean_stems"].astype(float).to_numpy()
    base = float(vals[:-1].mean())
    if base <= 0:
        return 0.0, n
    dev = (float(vals[-1]) - base) / base
    return float(np.clip(dev, -3.0, 3.0)), n


def ais_alpha(n_graded_ais_months: int) -> float:
    """Shrinkage weight for the AIS nudge. Zero until NOWCAST_AIS_MIN_GRADED_MONTHS
    of graded AIS history exist, then ramps linearly toward NOWCAST_AIS_MAX_ALPHA."""
    if n_graded_ais_months < NOWCAST_AIS_MIN_GRADED_MONTHS:
        return 0.0
    ramp = min(1.0, (n_graded_ais_months - NOWCAST_AIS_MIN_GRADED_MONTHS) /
               float(NOWCAST_AIS_MIN_GRADED_MONTHS))
    return float(NOWCAST_AIS_MAX_ALPHA * ramp)


# ── Public entry point ──────────────────────────────────────────────────────

def calibrated_nowcast(port: str, year: int, month: int) -> dict:
    """Full calibrated nowcast for (port, year, month). Pure read; no writes.

    AIS is engaged ONLY when ais_signal.live_coupling confirms a SIGNAL verdict
    (Pearson r > 0, p < 0.10, positive OOS skill). Until then alpha = 0 and the
    nowcast equals the level model. The coupling coefficient (sign + magnitude) is
    fit from the actual residuals by live_coupling, not from elapsed month count.
    """
    import ais_signal as _sig  # lazy import — ais_signal imports nowcast_model at top level

    series = official_series(port, year, month)
    level = level_forecast(series, month)

    coupling = _sig.live_coupling(port)
    alpha = coupling["alpha_hat"]  # 0.0 unless signal_test verdict == "SIGNAL"

    z, n_ais = ais_deviation(port, year, month) if alpha != 0.0 else (0.0, coupling["n_pairs"])
    nowcast = level * (1.0 + alpha * z)
    return {
        "port": port, "year": year, "month": month,
        "level_mt": round(level, 0),
        "ais_z": round(z, 3), "ais_alpha": round(alpha, 4),
        "ais_history_months": n_ais,
        "ais_signal_verdict": coupling["verdict"],
        "nowcast_mt": round(nowcast, 0),
        "ais_adjustment_pct": round(alpha * z * 100.0, 2),
    }


if __name__ == "__main__":
    import sys
    p = sys.argv[1] if len(sys.argv) > 1 else "singapore"
    y = int(sys.argv[2]) if len(sys.argv) > 2 else 2026
    m = int(sys.argv[3]) if len(sys.argv) > 3 else 5
    print(calibrated_nowcast(p, y, m))
