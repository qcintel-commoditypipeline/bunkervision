"""
Demand estimation model — port-aware.
Computes seasonal averages and projects current-month demand from detected bunkering events.
For ports without AIS event detection, returns historical/official figures only.
"""
from __future__ import annotations

import calendar
from datetime import date, datetime, timedelta, timezone

import db
from config import DEFAULT_PUMP_RATE_MT_MIN, PORTS


def _db_port(port: str) -> str:
    """Map config key to the port name stored in port_bunker_sales."""
    return PORTS.get(port, {}).get("db_port", port)


def seasonal_avg_monthly_volume(month: int, port: str = "singapore",
                                 fuel_type: str = "TOTAL") -> float | None:
    """Return the historical average volume for calendar month `month` (1-12)."""
    rows = db.query("""
        SELECT AVG(volume_mt)
        FROM port_bunker_sales
        WHERE port = ? AND MONTH(month) = ? AND fuel_type = ?
    """, [_db_port(port), month, fuel_type])
    if rows and rows[0][0] is not None:
        return float(rows[0][0])

    if port == "singapore":
        rows = db.query("""
            SELECT AVG(volume_mt) FROM mpa_monthly
            WHERE MONTH(month) = ? AND fuel_type = ?
        """, [month, fuel_type])
        if rows and rows[0][0] is not None:
            return float(rows[0][0])

    return None


def running_demand_estimate(port: str = "singapore",
                             year: int | None = None,
                             month: int | None = None) -> dict:
    """
    Returns a demand snapshot for the current (or specified) month.
    For ports without a vessel registry, event_count / estimated_mt are 0
    and projected_month_mt is None.
    """
    today = date.today()
    year  = year  or today.year
    month = month or today.month

    days_elapsed  = today.day
    days_in_month = calendar.monthrange(year, month)[1]

    has_registry = PORTS.get(port, {}).get("has_registry", False)

    event_count  = 0
    estimated_mt = 0.0
    open_mt      = 0.0

    if has_registry:
        # Quality-filter / dedup are applied inside aggregate_closed_event_mt and
        # are gated by config flags (legacy aggregate when flags default to off).
        from event_detector import _avg_pump_rate, aggregate_closed_event_mt
        event_count, estimated_mt = aggregate_closed_event_mt(port, year, month)

        open_rows = db.query("""
            SELECT id, start_ts FROM bunkering_events
            WHERE open = TRUE AND port = ?
              AND YEAR(start_ts) = ? AND MONTH(start_ts) = ?
        """, [port, year, month])

        pump_rate = _avg_pump_rate(port)
        now = datetime.now(timezone.utc)
        for _, start_ts in open_rows:
            if start_ts:
                start = start_ts
                if hasattr(start, "replace"):
                    start = start.replace(tzinfo=timezone.utc)
                duration_min = max(1, int((now - start).total_seconds() / 60))
                open_mt += duration_min * pump_rate

    total_so_far = estimated_mt + open_mt
    projected = (total_so_far / days_elapsed * days_in_month) if (has_registry and days_elapsed > 0 and total_so_far > 0) else None

    seasonal = seasonal_avg_monthly_volume(month, port)

    last_rows = db.query("""
        SELECT month, volume_mt FROM port_bunker_sales
        WHERE port = ? AND fuel_type = 'TOTAL'
        ORDER BY month DESC LIMIT 1
    """, [_db_port(port)])
    if not last_rows and port == "singapore":
        last_rows = db.query("""
            SELECT month, volume_mt FROM mpa_monthly
            WHERE fuel_type = 'TOTAL' ORDER BY month DESC LIMIT 1
        """)

    last_official_mt    = float(last_rows[0][1]) if last_rows else None
    last_official_month = last_rows[0][0]         if last_rows else None

    pct = None
    if projected and seasonal and seasonal > 0:
        pct = round((projected / seasonal - 1) * 100, 1)

    return {
        "port": port,
        "has_registry": has_registry,
        "year": year,
        "month": month,
        "event_count": event_count,
        "estimated_mt": round(estimated_mt, 0),
        "open_event_mt": round(open_mt, 0),
        "days_elapsed": days_elapsed,
        "days_in_month": days_in_month,
        "projected_month_mt": round(projected, 0) if projected else None,
        "seasonal_avg_mt": round(seasonal, 0) if seasonal else None,
        "last_official_mt": round(last_official_mt, 0) if last_official_mt else None,
        "last_official_month": str(last_official_month) if last_official_month else None,
        "pct_vs_seasonal": pct,
    }


# ── Month-end snapshot ─────────────────────────────────────────────────────────

def _month_coverage_fraction(port: str, year: int, month: int) -> float:
    """
    Fraction of the calendar month spanned by detected events for this port.

    Returns (last_event_day - first_event_day + 1) / days_in_month, i.e. the
    span of detected activity. A month the system only watched for the back half
    (e.g. the system turned on mid-April 2026) yields a low fraction and is
    treated as partial by the guard.
    """
    days_in_month = calendar.monthrange(year, month)[1]
    rows = db.query("""
        SELECT MIN(start_ts), MAX(start_ts)
        FROM bunkering_events
        WHERE port = ? AND YEAR(start_ts) = ? AND MONTH(start_ts) = ?
    """, [port, year, month])
    if not rows or rows[0][0] is None or rows[0][1] is None:
        return 0.0
    first_day = rows[0][0].day
    last_day  = rows[0][1].day
    return (last_day - first_day + 1) / days_in_month


def save_month_estimate(port: str, year: int, month: int) -> bool:
    """
    Snapshot the total for a completed month into monthly_demand_estimates.
    Only saves if the port has a registry and the month has events.
    Returns True if saved, False if skipped.

    Partial-month guard (PARTIAL_MONTH_GUARD_ENABLED): a month whose detected
    AIS coverage clearly does not span the full calendar month is recorded with
    is_partial=TRUE (when the column exists) rather than graded as complete. The
    system turned on mid-April 2026, so April is a ramp artifact and must not be
    presented as a finished month. The guard defaults OFF so deploying this code
    does not change existing behaviour until the flag is flipped.
    """
    from config import MONTH_COVERAGE_MIN_FRACTION, PARTIAL_MONTH_GUARD_ENABLED

    if not PORTS.get(port, {}).get("has_registry", False):
        return False

    from event_detector import aggregate_closed_event_mt
    event_count, estimated_mt = aggregate_closed_event_mt(port, year, month)

    if event_count == 0:
        return False

    if PARTIAL_MONTH_GUARD_ENABLED:
        coverage = _month_coverage_fraction(port, year, month)
        if coverage < MONTH_COVERAGE_MIN_FRACTION:
            # The monthly_demand_estimates schema has no `is_partial` column
            # (owned by db.py, outside this change set). To avoid grading a
            # ramp month as complete, we SKIP the snapshot entirely and log it.
            # The integrator can add an is_partial column + downstream label
            # later; for now skipping is the safe, non-destructive choice.
            from loguru import logger
            logger.warning(
                f"{port} {year}-{month:02d}: AIS coverage only "
                f"{coverage:.0%} of the month (< "
                f"{MONTH_COVERAGE_MIN_FRACTION:.0%}) — skipping snapshot rather "
                f"than grading a partial month as complete."
            )
            return False

    # NOTE: the previous `estimated_mt / days_in_month * days_in_month` was a
    # no-op (it divided then multiplied by the same number). A completed month's
    # total is simply the sum of its closed events; we do NOT extrapolate.
    projected = estimated_mt
    seasonal  = seasonal_avg_monthly_volume(month, port)
    pct       = round((projected / seasonal - 1) * 100, 1) if seasonal and seasonal > 0 else None

    db.execute("""
        INSERT INTO monthly_demand_estimates
            (port, year, month, estimated_mt, event_count, pct_vs_seasonal, saved_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT (port, year, month) DO UPDATE SET
            estimated_mt    = excluded.estimated_mt,
            event_count     = excluded.event_count,
            pct_vs_seasonal = excluded.pct_vs_seasonal,
            saved_at        = excluded.saved_at
    """, [port, year, month, round(projected, 0), event_count, pct,
          datetime.now(timezone.utc).isoformat()])
    return True


def snapshot_previous_month_if_missing(port: str = "singapore") -> None:
    """
    Called at month rollover (1st of month). Saves last month's estimate
    if it hasn't been saved yet. Safe to call multiple times.
    """
    today = date.today()
    if today.month == 1:
        prev_year, prev_month = today.year - 1, 12
    else:
        prev_year, prev_month = today.year, today.month - 1

    existing = db.query("""
        SELECT 1 FROM monthly_demand_estimates
        WHERE port = ? AND year = ? AND month = ?
    """, [port, prev_year, prev_month])

    if not existing:
        saved = save_month_estimate(port, prev_year, prev_month)
        if saved:
            from loguru import logger
            logger.info(f"Snapshotted {port} {prev_year}-{prev_month:02d} estimate")


# ── Charts ─────────────────────────────────────────────────────────────────────

def _monthly_avg_price(port: str, grade: str = "VLSFO") -> dict[str, float]:
    """Return {YYYY-MM: avg_price} monthly averages from S&B for a given port/grade."""
    import sqlite3
    from pathlib import Path
    PORT_MAP = {
        "singapore":  "Singapore",
        "fujairah":   "Fujairah",
        "rotterdam":  "Rotterdam",
        "antwerp":    "Antwerp",
        "panama":     "Balboa",
        "gibraltar":  "Gibraltar",
        "houston":    "Houston",
        "hong_kong":  "Hong Kong",
        "zhoushan":   "Zhoushan",
    }
    sb_port = PORT_MAP.get(port)
    if not sb_port:
        return {}
    try:
        conn = sqlite3.connect("/root/bunkervision/bunker_prices.db")
        rows = conn.execute("""
            SELECT strftime('%Y-%m', date) AS ym, AVG(price_usd)
            FROM bunker_prices
            WHERE source='shipandbunker' AND port_name=? AND fuel_grade=?
            GROUP BY ym ORDER BY ym
        """, (sb_port, grade)).fetchall()
        conn.close()
        return {r[0]: round(r[1], 2) for r in rows if r[1] is not None}
    except Exception:
        return {}


_MONTH_NAMES = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]


def _window_periods(window: str, frequency: str) -> int | None:
    """Translate a UI window token (1y/3y/5y/all) into a number of trailing
    periods for the given data frequency. Returns None for 'all' (no limit)."""
    if window in (None, "all"):
        return None
    years = {"1y": 1, "3y": 3, "5y": 5}.get(window, 3)
    return years * (4 if frequency == "quarterly" else 12)


def demand_chart_data(port: str = "singapore", window: str = "3y") -> dict:
    """
    Plotly figure: official bars (grey) overlaid with provisional estimates (amber)
    + current-month forecast (green/red) + seasonal avg line.
    """
    import pandas as pd
    import plotly.graph_objects as go

    # Use TOTAL where available per month; fall back to summing grades when TOTAL is absent
    # (handles cases where the source only published grade-level data, e.g. recent MPA months)
    df = db.query_df("""
        SELECT month,
            COALESCE(
                MAX(CASE WHEN fuel_type = 'TOTAL' THEN volume_mt END),
                SUM(CASE WHEN fuel_type != 'TOTAL' THEN volume_mt END)
            ) AS volume_mt
        FROM port_bunker_sales
        WHERE port = ?
        GROUP BY month
        HAVING volume_mt > 0
        ORDER BY month
    """, [_db_port(port)])

    if df.empty and port == "singapore":
        df = db.query_df("""
            SELECT month, SUM(volume_mt) AS volume_mt
            FROM mpa_monthly WHERE fuel_type != 'TOTAL'
            GROUP BY month ORDER BY month
        """)

    port_cfg  = PORTS.get(port, {})
    frequency = port_cfg.get("data_frequency", "monthly")

    # Headline focuses on the recent window so bars stay legible (the full
    # multi-year history crammed onto a phone-width chart is unreadable).
    # Window is user-selectable (1y/3y/all) via the chart toggle.
    _n = _window_periods(window, frequency)
    if _n and not df.empty:
        df = df.tail(_n).reset_index(drop=True)

    _BAR_W_MS = 20 * 24 * 3600 * 1000  # 20-day bar width in ms (date-axis bars need explicit width)

    _LAYOUT = dict(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(22,27,34,0.6)",
        font=dict(family="Inter, system-ui, sans-serif", size=12, color="#c9d1d9"),
        margin=dict(l=64, r=18, t=24, b=54),
        legend=dict(orientation="h", yanchor="bottom", y=1.04, x=0),
        barmode="overlay",
        dragmode=False,
        yaxis=dict(title=None, ticksuffix=" Mt", hoverformat=".2f",
                   automargin=True, gridcolor="rgba(240,246,252,0.05)", zeroline=False),
        xaxis=dict(type="date", tickformat="%b %Y", automargin=True,
                   gridcolor="rgba(0,0,0,0)"),
        xaxis_title="",
    )

    fig = go.Figure(layout=_LAYOUT)

    if not df.empty:
        x_labels = [str(m)[:7] for m in df["month"]]  # YYYY-MM, consistent with provisional labels
        fig.add_trace(go.Bar(
            x=x_labels,
            y=(df["volume_mt"] / 1e6).tolist(),
            name="Official",
            marker_color="#30363d",
            opacity=0.9,
            width=_BAR_W_MS,
        ))

        try:
            if frequency == "quarterly":
                df["qtr"] = df["month"].dt.month.apply(lambda m: (m - 1) // 3)
                avg_by_period = df.groupby("qtr")["volume_mt"].mean()
                avg_vals = [avg_by_period.get((pd.Timestamp(m).month - 1) // 3, None)
                            for m in df["month"]]
            else:
                avg_by_month = df.groupby(df["month"].dt.month)["volume_mt"].mean()
                avg_vals = [avg_by_month.get(m.month, None) for m in df["month"]]
            avg_vals_mt = [v / 1e6 if v is not None else None for v in avg_vals]
        except Exception:
            avg_vals_mt = None

    # Saved month-end estimates (amber) — shown for ALL months with estimates,
    # overlaid on official bars so accuracy vs actual can be compared visually
    official_months: set[str] = set(x_labels) if not df.empty else set()  # YYYY-MM format
    est_rows = db.query("""
        SELECT year, month, estimated_mt, pct_vs_seasonal
        FROM monthly_demand_estimates
        WHERE port = ?
        ORDER BY year, month
    """, [port])
    if est_rows:
        est_x, est_y, est_hover, est_bar_text = [], [], [], []
        for yr, mo, mt, pct in est_rows:
            label = f"{yr}-{mo:02d}"
            est_x.append(label)
            est_y.append(mt)
            pct_str = f"{pct:+.1f}% vs seasonal" if pct is not None else ""
            month_name = f"{_MONTH_NAMES[mo - 1]} {yr}"
            est_hover.append(
                f"<b>{month_name}</b><br>Provisional: {mt/1e6:.2f} Mt"
                + (f"<br>{pct_str}" if pct_str else "")
            )
            # Only show on-bar text for months without official (avoids crowding the overlay)
            est_bar_text.append(
                f"{mt/1e6:.2f} Mt" + (f"<br>{pct_str}" if pct_str else "")
                if label not in official_months else ""
            )
        if est_x:
            fig.add_trace(go.Bar(
                x=est_x,
                y=[v / 1e6 for v in est_y],
                name="Provisional",
                marker_color="#e3b341",
                opacity=0.75,
                width=_BAR_W_MS,
                text=est_bar_text,
                textposition="outside",
                hovertemplate="%{customdata}<extra></extra>",
                customdata=est_hover,
            ))

    # Current-month live forecast (green/red)
    has_registry = port_cfg.get("has_registry", False)
    if has_registry:
        est = running_demand_estimate(port)
        if est.get("projected_month_mt"):
            today     = date.today()
            cur_label = today.strftime("%Y-%m")
            colour    = "#3fb950" if (est.get("pct_vs_seasonal") or 0) >= 0 else "#f85149"
            pct_str   = f"{est['pct_vs_seasonal']:+.1f}% vs seasonal" if est.get("pct_vs_seasonal") is not None else ""
            month_name = today.strftime("%b %Y")
            hover_text = (
                f"<b>{month_name}</b><br>Forecast: {est['projected_month_mt']/1e6:.2f} Mt"
                + (f"<br>{pct_str}" if pct_str else "")
            )
            bar_text = f"{est['projected_month_mt']/1e6:.2f} Mt" + (f"<br>{pct_str}" if pct_str else "")
            fig.add_trace(go.Bar(
                x=[cur_label],
                y=[est["projected_month_mt"] / 1e6],
                name="Forecast",
                marker_color=colour,
                opacity=0.9,
                width=_BAR_W_MS,
                text=[bar_text],
                textposition="outside",
                hovertemplate="%{customdata}<extra></extra>",
                customdata=[hover_text],
            ))

    if not df.empty and avg_vals_mt is not None:
        fig.add_trace(go.Scatter(
            x=x_labels,
            y=avg_vals_mt,
            name="Seasonal Avg",
            mode="lines",
            line=dict(color="#58a6ff", width=1, dash="dot"),
        ))

    return fig.to_dict()


def demand_sparkline_data(port: str = "singapore") -> dict:
    """Minimal, axis-less sparkline of the last ~12 months of total demand plus
    the live current-month forecast — rendered inside the hero card so it earns
    its space."""
    import plotly.graph_objects as go
    from datetime import date

    df = db.query_df("""
        SELECT month,
            COALESCE(
                MAX(CASE WHEN fuel_type = 'TOTAL' THEN volume_mt END),
                SUM(CASE WHEN fuel_type != 'TOTAL' THEN volume_mt END)
            ) AS volume_mt
        FROM port_bunker_sales
        WHERE port = ?
        GROUP BY month HAVING volume_mt > 0
        ORDER BY month
    """, [_db_port(port)])
    if df.empty and port == "singapore":
        df = db.query_df("""
            SELECT month, SUM(volume_mt) AS volume_mt
            FROM mpa_monthly WHERE fuel_type != 'TOTAL'
            GROUP BY month ORDER BY month
        """)

    layout = dict(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=0, t=6, b=0),
        showlegend=False, dragmode=False, hovermode="x",
        xaxis=dict(visible=False), yaxis=dict(visible=False),
    )
    fig = go.Figure(layout=layout)
    if df.empty:
        return fig.to_dict()

    df = df.tail(12).reset_index(drop=True)
    xs = [str(m)[:7] for m in df["month"]]
    ys = [float(v) / 1e6 for v in df["volume_mt"]]

    est = running_demand_estimate(port) if PORTS.get(port, {}).get("has_registry") else {}
    fc  = est.get("projected_month_mt")

    fig.add_trace(go.Scatter(
        x=xs, y=ys, mode="lines", line=dict(color="#58a6ff", width=2),
        fill="tozeroy", fillcolor="rgba(88,166,255,0.10)",
        hovertemplate="%{x}: %{y:.2f} Mt<extra></extra>",
    ))
    if fc:
        colour = "#3fb950" if (est.get("pct_vs_seasonal") or 0) >= 0 else "#f85149"
        fig.add_trace(go.Scatter(
            x=[xs[-1], date.today().strftime("%Y-%m")], y=[ys[-1], fc / 1e6],
            mode="lines+markers", line=dict(color=colour, width=2, dash="dot"),
            marker=dict(size=6, color=colour),
            hovertemplate="%{x} forecast: %{y:.2f} Mt<extra></extra>",
        ))
    vals = ys + ([fc / 1e6] if fc else [])
    fig.update_yaxes(range=[min(vals) * 0.9, max(vals) * 1.05])
    return fig.to_dict()


def fuel_split_chart_data(port: str = "singapore", window: str = "3y") -> dict:
    """
    Plotly stacked bar: official MPA fuel-type breakdown per month.
    Grades: VLSFO / HSFO / LSMGO / MGO / other. Grades that are negligible over
    the visible window (<0.5% share) are dropped to keep the legend readable.
    """
    import plotly.graph_objects as go

    GRADE_COLOURS = {
        "VLSFO":    "#58a6ff",
        "HSFO":     "#f85149",
        "LSMGO":    "#3fb950",
        "MGO":      "#e3b341",
        "LNG":      "#a371f7",
        "Methanol": "#79c0ff",
        "Ammonia":  "#56d364",
    }
    GRADE_ORDER = ["VLSFO", "HSFO", "LSMGO", "MGO", "LNG", "Methanol", "Ammonia"]

    df = db.query_df("""
        SELECT month, fuel_type, volume_mt
        FROM port_bunker_sales
        WHERE port = ? AND fuel_type != 'TOTAL'
        ORDER BY month, fuel_type
    """, [_db_port(port)])

    if df.empty and port == "singapore":
        df = db.query_df("""
            SELECT month, fuel_type, volume_mt
            FROM mpa_monthly
            WHERE fuel_type != 'TOTAL'
            ORDER BY month, fuel_type
        """)

    _LAYOUT = dict(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(22,27,34,0.6)",
        font=dict(family="Inter, system-ui, sans-serif", size=12, color="#c9d1d9"),
        margin=dict(l=64, r=18, t=24, b=54),
        legend=dict(orientation="h", yanchor="bottom", y=1.04, x=0),
        barmode="stack",
        yaxis=dict(title="Volume (mt)", hoverformat=",.0f",
                   automargin=True, gridcolor="rgba(240,246,252,0.05)", zeroline=False),
        xaxis=dict(automargin=True),
        xaxis_title="",
    )

    fig = go.Figure(layout=_LAYOUT)

    if df.empty:
        return fig.to_dict()

    x_all    = sorted(df["month"].astype(str).unique().tolist())
    freq     = PORTS.get(port, {}).get("data_frequency", "monthly")
    _n       = _window_periods(window, freq)
    if _n:
        x_all = x_all[-_n:]                                   # recent window only

    # Significance test over the visible window: drop grades contributing
    # <0.5% so the legend isn't cluttered with flat-zero series (e.g. Ammonia,
    # Methanol, LNG in Singapore).
    win_set   = set(x_all)
    dfw       = df[df["month"].astype(str).isin(win_set)]
    grade_tot = dfw.groupby("fuel_type")["volume_mt"].sum()
    total_all = float(grade_tot.sum()) or 1.0
    keep      = {g for g, v in grade_tot.items() if float(v) / total_all >= 0.005}

    grades   = [g for g in GRADE_ORDER if g in keep]
    others   = [g for g in df["fuel_type"].unique() if g not in GRADE_ORDER and g in keep]

    for grade in grades + others:
        sub = df[df["fuel_type"] == grade]
        grade_map = dict(zip(sub["month"].astype(str), sub["volume_mt"]))
        fig.add_trace(go.Bar(
            x=x_all,
            y=[grade_map.get(x, 0) for x in x_all],
            name=grade,
            marker_color=GRADE_COLOURS.get(grade, "#8b949e"),
        ))

    return fig.to_dict()
