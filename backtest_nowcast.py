"""
Reusable out-of-sample backtest for the calibrated nowcast level model.

Standalone (no salvage/ scratch deps): builds the official monthly panel straight
from the DB and runs an expanding-window walk-forward — to predict month t, fit only
on months < t. Prints a league table of the prod level model vs naive benchmarks so
the nowcast's edge is re-verifiable every month as new official figures land.

    python backtest_nowcast.py [--port singapore] [--min-train 36]

Reads only; never writes. Uses nowcast_model.level_forecast (the live code path), so
this validates exactly what production would serve under USE_CALIBRATED_NOWCAST.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

import db
import nowcast_model as nm


def _full_series(port: str) -> pd.DataFrame:
    """Entire official monthly TOTAL series for the port (no cutoff)."""
    # official_series with a far-future cutoff returns everything.
    return nm.official_series(port, 9999, 12)


def _metrics(res: pd.DataFrame) -> dict:
    r = res.dropna(subset=["y_pred"]).copy()
    if r.empty:
        return {"n": 0}
    r["signed"] = (r["y_pred"] / r["y_true"] - 1.0) * 100.0
    r["abs"] = r["signed"].abs()
    r["yr"] = r["month"].dt.year
    return {
        "n": int(len(r)),
        "mae_pct": round(float(r["abs"].mean()), 2),
        "median_abs_pct": round(float(r["abs"].median()), 2),
        "bias_pct": round(float(r["signed"].mean()), 2),
        "within_5pct": int((r["abs"] <= 5).sum()),
        "within_10pct": int((r["abs"] <= 10).sum()),
        "worst_abs_pct": round(float(r["abs"].max()), 2),
        "by_year": {int(y): round(float(g["abs"].mean()), 2) for y, g in r.groupby("yr")},
    }


def _walk(series: pd.DataFrame, predict, min_train: int) -> pd.DataFrame:
    rows = []
    for i in range(min_train, len(series)):
        hist = series.iloc[:i]
        tgt = series.iloc[i]
        tm = int(pd.Timestamp(tgt["month"]).month)
        try:
            pred = float(predict(hist, tm))
        except Exception:
            pred = np.nan
        rows.append({"month": tgt["month"], "y_true": float(tgt["y"]), "y_pred": pred})
    return pd.DataFrame(rows)


# --- models -----------------------------------------------------------------

def _level(hist, tm):
    return nm.level_forecast(hist, tm)

def _seasonal_naive(hist, tm):
    same = hist[hist["month"].dt.month == tm]["y"]
    return float(same.mean()) if len(same) else float(hist["y"].iloc[-1])

def _trailing12(hist, tm):
    return float(hist["y"].tail(12).mean())


def oos_table(port: str = "singapore", min_train: int = 36) -> dict:
    """Structured out-of-sample comparison for the shadow panel / API.

    Read-only. Returns the per-month nowcast vs seasonal-naive vs (where it exists)
    the OLD duration-engine estimate, against the official figure, plus summaries.
    Months are ISO strings so the result is JSON-serializable.
    """
    series = _full_series(port).sort_values("month").reset_index(drop=True)
    out = {"port": port, "min_train": min_train, "rows": [], "summary": {}}
    if len(series) <= min_train:
        return out

    nc = _walk(series, _level, min_train)
    sn = _walk(series, _seasonal_naive, min_train)

    # OLD duration-engine snapshots (only exist for the months the live system ran).
    old = {}
    try:
        for yr, mo, est in db.query(
            "SELECT year, month, estimated_mt FROM monthly_demand_estimates WHERE port = ?",
            [port],
        ):
            old[(int(yr), int(mo))] = float(est) if est is not None else None
    except Exception:
        pass

    def _pct(p, a):
        return None if not a else round((p / a - 1.0) * 100.0, 2)

    for i in range(len(nc)):
        mth = pd.Timestamp(nc.iloc[i]["month"])
        official = float(nc.iloc[i]["y_true"])
        ncp = nc.iloc[i]["y_pred"]
        snp = sn.iloc[i]["y_pred"]
        oldp = old.get((mth.year, mth.month))
        out["rows"].append({
            "month": mth.date().isoformat(),
            "official_mt": official,
            "nowcast_mt": None if pd.isna(ncp) else round(float(ncp), 0),
            "nowcast_err_pct": None if pd.isna(ncp) else _pct(float(ncp), official),
            "seasonal_naive_mt": None if pd.isna(snp) else round(float(snp), 0),
            "seasonal_naive_err_pct": None if pd.isna(snp) else _pct(float(snp), official),
            "old_engine_mt": None if oldp is None else round(oldp, 0),
            "old_engine_err_pct": None if oldp is None else _pct(oldp, official),
        })

    out["summary"] = {
        "nowcast": _metrics(nc),
        "seasonal_naive": _metrics(sn),
        "n_old_engine_months": sum(1 for r in out["rows"] if r["old_engine_mt"] is not None),
    }
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", default="singapore")
    ap.add_argument("--min-train", type=int, default=36)
    args = ap.parse_args()

    series = _full_series(args.port).sort_values("month").reset_index(drop=True)
    if len(series) <= args.min_train:
        print(f"Not enough history for {args.port}: {len(series)} months")
        return

    models = [
        ("nowcast_level (PROD)", _level),
        ("seasonal_naive (current dashboard)", _seasonal_naive),
        ("trailing_12mo_mean", _trailing12),
    ]
    print(f"\nOut-of-sample backtest - {args.port} - {len(series)} months "
          f"({series['month'].min().date()} to {series['month'].max().date()}), "
          f"expanding window, min_train={args.min_train}\n")
    print(f"  {'model':38s} {'MAE%':>6} {'bias%':>7} {'<5%':>6} {'<10%':>6} {'worst%':>7}")
    print("  " + "-" * 76)
    table = []
    for name, fn in models:
        m = _metrics(_walk(series, fn, args.min_train))
        table.append((name, m))
        print(f"  {name:38s} {m['mae_pct']:>6} {m['bias_pct']:>7} "
              f"{m['within_5pct']:>4}/{m['n']} {m['within_10pct']:>4}/{m['n']} {m['worst_abs_pct']:>7}")
    print()
    for name, m in table:
        print(f"  {name:38s} by_year={m['by_year']}")


if __name__ == "__main__":
    main()
