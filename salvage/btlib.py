"""
Shared backtest library for the nowcast salvage bake-off.

Every candidate level-model uses THIS harness so the out-of-sample protocol and
metrics are identical across agents (a fair league table). Strict expanding-window
walk-forward: to predict month t, a model may train ONLY on months < t.

A model is a callable:  forecast_one(train_df, test_row) -> float
  train_df : all panel rows strictly BEFORE the test month (chronological)
  test_row : the single row being predicted (use ONLY its month/calendar/lag fields,
             NEVER its 'y'); same-month other-port columns must be treated as unknown.

Usage in an agent:
    import btlib
    df = btlib.load_panel()
    res = btlib.walk_forward(df, my_forecast_one, min_train=36)
    print(btlib.metrics(res))
"""
from __future__ import annotations

import numpy as np
import pandas as pd

PANEL = "salvage/panel_official.csv"

# Columns a model may read from test_row without leakage (known before month t):
SAFE_TEST_COLS = ["month", "year", "month_num", "t",
                  "y_lag1", "y_lag3", "y_lag6", "y_lag12"]


def load_panel(path: str = PANEL) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["month"]).sort_values("month").reset_index(drop=True)
    return df


def walk_forward(df: pd.DataFrame, forecast_one, min_train: int = 36) -> pd.DataFrame:
    """Expanding-window walk-forward. Returns a frame of (month, y_true, y_pred)."""
    rows = []
    for i in range(min_train, len(df)):
        train = df.iloc[:i].copy()
        test_row = df.iloc[i]
        try:
            pred = float(forecast_one(train, test_row))
        except Exception as e:
            pred = np.nan
        rows.append({"month": test_row["month"], "y_true": float(test_row["y"]),
                     "y_pred": pred})
    return pd.DataFrame(rows)


def metrics(res: pd.DataFrame) -> dict:
    """Compute OOS error metrics over the walk-forward results."""
    r = res.dropna(subset=["y_pred"]).copy()
    if r.empty:
        return {"n": 0}
    r["signed_pct"] = (r["y_pred"] / r["y_true"] - 1.0) * 100.0
    r["abs_pct"] = r["signed_pct"].abs()
    out = {
        "n": int(len(r)),
        "mae_pct": round(float(r["abs_pct"].mean()), 2),
        "median_abs_pct": round(float(r["abs_pct"].median()), 2),
        "bias_pct": round(float(r["signed_pct"].mean()), 2),
        "rmse_pct": round(float(np.sqrt((r["signed_pct"] ** 2).mean())), 2),
        "within_5pct": int((r["abs_pct"] <= 5).sum()),
        "within_10pct": int((r["abs_pct"] <= 10).sum()),
        "worst_abs_pct": round(float(r["abs_pct"].max()), 2),
    }
    # Regime breakdown by OOS year
    r["yr"] = r["month"].dt.year
    out["by_year"] = {int(y): round(float(g["abs_pct"].mean()), 2)
                      for y, g in r.groupby("yr")}
    return out


# ---- Reference models (templates + benchmark) ------------------------------

def seasonal_naive(train: pd.DataFrame, test_row) -> float:
    """Benchmark: average of the same calendar month over all prior years.
    This is the 'kernel' the dashboard already uses (seasonal_avg_monthly_volume)."""
    m = test_row["month_num"]
    same = train[train["month_num"] == m]["y"]
    if len(same) == 0:
        return float(train["y"].iloc[-1])
    return float(same.mean())


def seasonal_naive_recent(train: pd.DataFrame, test_row, k: int = 3) -> float:
    """Average of the same calendar month over the most recent k years."""
    m = test_row["month_num"]
    same = train[train["month_num"] == m]["y"]
    if len(same) == 0:
        return float(train["y"].iloc[-1])
    return float(same.tail(k).mean())


def last12_mean(train: pd.DataFrame, test_row) -> float:
    """Trailing 12-month mean (level only, no seasonality)."""
    return float(train["y"].tail(12).mean())


if __name__ == "__main__":
    df = load_panel()
    print(f"panel: {len(df)} months {df['month'].min().date()} -> {df['month'].max().date()}")
    for name, fn in [("seasonal_naive", seasonal_naive),
                     ("seasonal_naive_k3", seasonal_naive_recent),
                     ("last12_mean", last12_mean)]:
        res = walk_forward(df, fn, min_train=36)
        m = metrics(res)
        print(f"\n{name}: MAE%={m['mae_pct']} bias={m['bias_pct']} "
              f"within5={m['within_5pct']}/{m['n']} within10={m['within_10pct']}/{m['n']} "
              f"worst={m['worst_abs_pct']}")
        print(f"  by_year={m['by_year']}")
