"""
Build the monthly modeling panel for the nowcast salvage effort.

Reads ONLY from the isolated copy salvage/bunkervision_backtest.db (never the live
DB, never sacred tables in write mode) and emits two CSVs the overnight agents share:

  salvage/panel_official.csv  — 76 months of official series + calendar + lags +
                                other-port leading indicators (the LEVEL target+features)
  salvage/panel_ais.csv       — AIS-derived monthly features (Apr/May only)

Leakage note: lags here are simple calendar lags of the *target itself*, which are
legitimately known before month t publishes (they are prior months' official figures).
Other-port columns are emitted RAW (same-month); the backtest harness must LAG them
before use, since other ports publish on their own schedule. The seasonal baseline is
deliberately NOT precomputed — the harness computes it causally per fold.
"""
from __future__ import annotations

import duckdb
import pandas as pd

DB = "salvage/bunkervision_backtest.db"
c = duckdb.connect(DB, read_only=True)

# --- Official Singapore TOTAL: the target -----------------------------------
sg = c.execute("""
    SELECT month, volume_mt AS y
    FROM port_bunker_sales
    WHERE port='singapore' AND fuel_type='TOTAL'
    ORDER BY month
""").df()
sg["month"] = pd.to_datetime(sg["month"])

# --- Singapore fuel-type composition (potential features) -------------------
comp = c.execute("""
    SELECT month, fuel_type, volume_mt
    FROM port_bunker_sales
    WHERE port='singapore' AND fuel_type IN ('VLSFO','HSFO','MGO','LSMGO')
    ORDER BY month
""").df()
comp["month"] = pd.to_datetime(comp["month"])
comp_w = comp.pivot_table(index="month", columns="fuel_type", values="volume_mt").reset_index()
comp_w.columns = ["month"] + [f"sg_{x.lower()}" for x in comp_w.columns[1:]]

# --- Other-port leading indicators (raw, same-month; harness must lag) -------
def port_total(port_db_name: str, col: str) -> pd.DataFrame:
    df = c.execute("""
        SELECT month, volume_mt AS v
        FROM port_bunker_sales
        WHERE port=? AND fuel_type='TOTAL'
        ORDER BY month
    """, [port_db_name]).df()
    df["month"] = pd.to_datetime(df["month"])
    return df.rename(columns={"v": col})

others = sg[["month"]].copy()
for pname, col in [("fujairah", "fujairah_total"),
                   ("rotterdam", "rotterdam_total"),
                   ("panama", "panama_total")]:
    try:
        others = others.merge(port_total(pname, col), on="month", how="left")
    except Exception as e:
        print(f"  (skip {pname}: {e})")

# --- Assemble official panel -------------------------------------------------
panel = sg.merge(comp_w, on="month", how="left").merge(others, on="month", how="left")
panel = panel.sort_values("month").reset_index(drop=True)
panel["year"] = panel["month"].dt.year
panel["month_num"] = panel["month"].dt.month
panel["t"] = range(len(panel))           # trend index
for k in (1, 3, 6, 12):
    panel[f"y_lag{k}"] = panel["y"].shift(k)

panel.to_csv("salvage/panel_official.csv", index=False)
print(f"panel_official.csv: {len(panel)} months, {panel['month'].min().date()} -> {panel['month'].max().date()}")
print("  columns:", list(panel.columns))
print(panel[["month", "y", "y_lag1", "y_lag12"]].tail(4).to_string(index=False))

# --- AIS monthly features (Apr/May 2026 only) -------------------------------
ais = c.execute("""
    WITH ev AS (
        SELECT *,
               date_trunc('month', start_ts) AS m
        FROM bunkering_events
        WHERE port='singapore' AND open=FALSE
    )
    SELECT
        m AS month,
        COUNT(*)                                                   AS ev_all,
        SUM(CASE WHEN duration_min>=30 THEN 1 ELSE 0 END)          AS ev_ge30,
        SUM(CASE WHEN duration_min>=30 AND duration_min<360 THEN 1 ELSE 0 END) AS ev_clean_30m_6h,
        SUM(CASE WHEN duration_min>=30 AND duration_min<360 THEN estimated_mt ELSE 0 END) AS mt_clean_30m_6h,
        SUM(CASE WHEN duration_min>=360 THEN estimated_mt ELSE 0 END) AS mt_long_ge6h,
        SUM(estimated_mt)                                          AS mt_all,
        COUNT(DISTINCT bunker_mmsi)                                AS distinct_bunkers,
        COUNT(DISTINCT recipient_mmsi)                             AS distinct_recipients,
        ROUND(AVG(duration_min),1)                                AS avg_dur,
        ROUND(MEDIAN(estimated_mt),1)                             AS med_mt
    FROM ev
    GROUP BY m ORDER BY m
""").df()
ais.to_csv("salvage/panel_ais.csv", index=False)
print(f"\npanel_ais.csv: {len(ais)} months")
print(ais.to_string(index=False))
c.close()
