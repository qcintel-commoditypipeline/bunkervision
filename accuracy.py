"""
Accuracy grading for BunkerVision monthly demand estimates.

Compares the model's saved monthly estimates (monthly_demand_estimates) against
official port-authority figures (port_bunker_sales TOTAL rows, with an mpa_monthly
fallback for Singapore) and persists per-month grades into the additive
`estimate_grades` table.

This module ONLY uses the shared `db` module; it never opens its own DB connection.
"""
from __future__ import annotations

import statistics
from datetime import datetime, timezone

import db
from config import PORTS


_MONTH_NAMES = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

# Heuristic threshold: a port's *earliest* saved estimate is treated as a
# partial first month (and excluded from accuracy stats) when its abs error
# exceeds this, because partial-month registries under-count the opening month.
_PARTIAL_MONTH_ABS_PCT_THRESHOLD = 35.0


def _db_port(port: str) -> str:
    """Map a config key to the port name stored in port_bunker_sales.

    Mirrors demand_model._db_port() exactly so display-name ports
    (Istanbul, Malta Offshore, Limassol, Tallinn) resolve correctly.
    """
    return PORTS.get(port, {}).get("db_port", port)


def _official_total(port: str, year: int, month: int) -> float | None:
    """Return the official TOTAL volume_mt for (year, month), or None if not published.

    Looks up port_bunker_sales using the same port mapping as the demand model,
    falling back to mpa_monthly for Singapore when port_bunker_sales lacks the row.
    """
    rows = db.query("""
        SELECT volume_mt
        FROM port_bunker_sales
        WHERE port = ? AND fuel_type = 'TOTAL'
          AND YEAR(month) = ? AND MONTH(month) = ?
        LIMIT 1
    """, [_db_port(port), year, month])
    if rows and rows[0][0] is not None:
        return float(rows[0][0])

    if port == "singapore":
        rows = db.query("""
            SELECT volume_mt
            FROM mpa_monthly
            WHERE fuel_type = 'TOTAL'
              AND YEAR(month) = ? AND MONTH(month) = ?
            LIMIT 1
        """, [year, month])
        if rows and rows[0][0] is not None:
            return float(rows[0][0])

    return None


def _ensure_table() -> None:
    """Create the additive estimate_grades table if it does not exist."""
    db.execute("""
        CREATE TABLE IF NOT EXISTS estimate_grades (
            port           TEXT,
            year           INTEGER,
            month          INTEGER,
            estimated_mt   REAL,
            official_mt    REAL,
            signed_pct_err REAL,
            abs_pct_err    REAL,
            graded_at      TIMESTAMP,
            PRIMARY KEY (port, year, month)
        )
    """)


def grade_saved_estimates(port: str | None = None) -> int:
    """Grade saved monthly estimates against official figures.

    For each row in monthly_demand_estimates (optionally filtered by `port`),
    look up the official TOTAL volume for that (year, month). If the official
    month has not been published yet, the row is left ungraded (skipped).
    Otherwise compute signed % error (est/official - 1) * 100 and abs % error,
    and upsert into estimate_grades.

    Returns the number of rows graded (inserted or updated).
    """
    _ensure_table()

    if port:
        est_rows = db.query("""
            SELECT port, year, month, estimated_mt
            FROM monthly_demand_estimates
            WHERE port = ?
            ORDER BY port, year, month
        """, [port])
    else:
        est_rows = db.query("""
            SELECT port, year, month, estimated_mt
            FROM monthly_demand_estimates
            ORDER BY port, year, month
        """)

    graded = 0
    now = datetime.now(timezone.utc).isoformat()

    for p, year, month, estimated_mt in est_rows:
        official = _official_total(p, int(year), int(month))
        if official is None or official == 0:
            # Official month not published (or zero / unusable) -> leave ungraded.
            continue

        est = float(estimated_mt) if estimated_mt is not None else 0.0
        signed = (est / official - 1.0) * 100.0
        abs_err = abs(signed)

        db.execute("""
            INSERT INTO estimate_grades
                (port, year, month, estimated_mt, official_mt,
                 signed_pct_err, abs_pct_err, graded_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (port, year, month) DO UPDATE SET
                estimated_mt   = excluded.estimated_mt,
                official_mt    = excluded.official_mt,
                signed_pct_err = excluded.signed_pct_err,
                abs_pct_err    = excluded.abs_pct_err,
                graded_at      = excluded.graded_at
        """, [p, int(year), int(month), est, official, signed, abs_err, now])
        graded += 1

    return graded


def grade_on_refresh(port: str | None = None) -> int:
    """Thin wrapper around grade_saved_estimates, intended to run after an
    official-data refresh.

    INTEGRATOR NOTE: the app should call this immediately after
    `refresh_port_stats` (i.e. after new official figures land), so that newly
    published months get graded against the existing saved estimates.
    """
    return grade_saved_estimates(port)


def accuracy_report(port: str = "singapore") -> dict:
    """Build the dashboard accuracy report for a port.

    Returns a dict with chronological per-month rows and an aggregate summary.
    The summary is computed ONLY over rows with status == "graded"
    (pending_official and partial_month rows are excluded).
    """
    # Make sure grades are current for this port before reporting.
    grade_saved_estimates(port)

    est_rows = db.query("""
        SELECT year, month, estimated_mt
        FROM monthly_demand_estimates
        WHERE port = ?
        ORDER BY year, month
    """, [port])

    # Earliest saved month for the port, used by the partial-month heuristic.
    earliest_key = None
    if est_rows:
        first = est_rows[0]
        earliest_key = (int(first[0]), int(first[1]))

    rows: list[dict] = []
    graded_signed: list[float] = []
    graded_abs: list[float] = []

    for year, month, estimated_mt in est_rows:
        year, month = int(year), int(month)
        est = float(estimated_mt) if estimated_mt is not None else None
        official = _official_total(port, year, month)

        label = f"{year}-{month:02d}"

        if official is None or official == 0:
            rows.append({
                "year": year, "month": month, "label": label,
                "estimated_mt": est, "official_mt": None,
                "signed_pct_err": None, "abs_pct_err": None,
                "status": "pending_official",
            })
            continue

        signed = ((est or 0.0) / official - 1.0) * 100.0
        abs_err = abs(signed)

        # Partial-month heuristic: the earliest saved month for a port often
        # only captures a partial AIS registry, which under-counts heavily.
        # If this is that earliest month AND abs error > threshold, flag it
        # as a partial month and EXCLUDE it from the summary stats.
        is_partial = (
            earliest_key is not None
            and (year, month) == earliest_key
            and abs_err > _PARTIAL_MONTH_ABS_PCT_THRESHOLD
        )

        status = "partial_month" if is_partial else "graded"
        rows.append({
            "year": year, "month": month, "label": label,
            "estimated_mt": est, "official_mt": official,
            "signed_pct_err": signed, "abs_pct_err": abs_err,
            "status": status,
        })

        if status == "graded":
            graded_signed.append(signed)
            graded_abs.append(abs_err)

    months_graded = len(graded_abs)
    summary = {
        "months_graded": months_graded,
        "mean_abs_pct_err": (sum(graded_abs) / months_graded) if months_graded else None,
        "median_abs_pct_err": statistics.median(graded_abs) if months_graded else None,
        "within_5pct": sum(1 for a in graded_abs if a <= 5.0),
        "within_10pct": sum(1 for a in graded_abs if a <= 10.0),
        "bias_pct": (sum(graded_signed) / months_graded) if months_graded else None,
    }

    return {"port": port, "rows": rows, "summary": summary}
