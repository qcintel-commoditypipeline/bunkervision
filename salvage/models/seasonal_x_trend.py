"""
seasonal_x_trend: trailing-12 level x multiplicative seasonal factor.

Idea
----
level   = mean of the last 12 observed y (captures the current trend/regime).
seasonal factor for calendar month m
        = average, over all train months of that calendar month, of
          ( y_i / mean(y over the 12 months ending at i) ).
        i.e. how high/low this calendar month typically sits relative to its own
        local trailing-12 level. This de-trends each ratio so the factor is a
        clean seasonal shape, robust to the upward trend.
forecast = level * factor(m).

Non-leaky:
  - level uses only train y (last 12).
  - seasonal factors are computed ONLY from train (each ratio uses a window
    that ends on a train month, so test-month y is never touched).
  - from test_row we read only month_num.
Low param count: just one ratio per calendar month, shrunk toward 1.0.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _seasonal_factors(y: np.ndarray, month_num: np.ndarray) -> dict:
    """For each i>=11, ratio = y[i] / mean(y[i-11..i]); group by calendar month."""
    n = len(y)
    buckets: dict[int, list[float]] = {}
    for i in range(11, n):
        local_level = y[i - 11 : i + 1].mean()
        if local_level > 0:
            r = y[i] / local_level
            buckets.setdefault(int(month_num[i]), []).append(r)
    # Average ratio per calendar month, with light shrinkage toward 1.0
    # (shrink protects calendar months with few observations / 76-row panel).
    factors = {}
    for m, rs in buckets.items():
        rs = np.asarray(rs, dtype=float)
        k = len(rs)
        raw = rs.mean()
        # shrink weight: more obs -> trust raw more. tau=1 pseudo-obs at 1.0.
        tau = 1.0
        factors[m] = (k * raw + tau * 1.0) / (k + tau)
    return factors


def forecast_one(train: pd.DataFrame, test_row) -> float:
    y = train["y"].to_numpy(dtype=float)
    mn = train["month_num"].to_numpy()

    level = float(y[-12:].mean()) if len(y) >= 12 else float(y.mean())

    m = int(test_row["month_num"])
    factors = _seasonal_factors(y, mn)
    f = factors.get(m, 1.0)

    return level * f


if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, "salvage")
    import btlib

    df = btlib.load_panel()
    res = btlib.walk_forward(df, forecast_one, min_train=36)
    print(btlib.metrics(res))
