"""
Monthly call-vs-actual post-mortem for BunkerVision.

Answers one question for a completed month: **what did we call, and what really
happened?** It pairs the model's saved estimate (what we called) against the
official port-authority TOTAL (what really happened), then decomposes the called
tonnage by event-duration band so a small headline error can't hide a wrong
internal composition (the long-duration "phantom" events that historically
dominated the total — now capped at EVENT_MAX_DURATION_MIN).

Read-only. Uses the shared `db` module and the same engine functions the live
model uses, so the headline call number matches the dashboard exactly.

Usage:
    python monthly_study.py                 # previous complete month, singapore
    python monthly_study.py --year 2026 --month 5
    python monthly_study.py --port singapore --year 2026 --month 5
"""
from __future__ import annotations

import argparse
from datetime import date

import db
from config import DEFAULT_PUMP_RATE_MT_MIN, EVENT_MAX_DURATION_MIN, MIN_BUNKER_EVENT_MIN
from event_detector import aggregate_closed_event_mt
from accuracy import _official_total
from demand_model import seasonal_avg_monthly_volume

_MONTHS = ["", "Jan", "Feb", "Mar", "Apr", "May", "Jun",
           "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def _prev_complete_month() -> tuple[int, int]:
    t = date.today()
    return (t.year - 1, 12) if t.month == 1 else (t.year, t.month - 1)


def _saved_call(port: str, year: int, month: int):
    """The estimate we snapshotted for the month (what we called), or None."""
    rows = db.query("""
        SELECT estimated_mt, event_count, pct_vs_seasonal, saved_at
        FROM monthly_demand_estimates
        WHERE port = ? AND year = ? AND month = ?
    """, [port, year, month])
    return rows[0] if rows else None


def _duration_bands(port: str, year: int, month: int):
    """Decompose closed-event tonnage (>= MIN_BUNKER_EVENT_MIN) into duration
    bands. Diagnostic: this sums stored estimated_mt without re-applying dedup,
    so the band total may sit slightly above the deduped engine figure. The
    point is the SHARE by band, especially the capped >16h tail."""
    rows = db.query("""
        SELECT
            CASE
                WHEN duration_min <  360  THEN '0.5-6h'
                WHEN duration_min <  1440 THEN '6-24h'
                ELSE '>24h'
            END AS band,
            COUNT(*)                AS n,
            COALESCE(SUM(estimated_mt), 0) AS mt,
            COALESCE(SUM(CASE WHEN duration_min > ? THEN 1 ELSE 0 END), 0) AS n_capped
        FROM bunkering_events
        WHERE open = FALSE AND port = ?
          AND YEAR(start_ts) = ? AND MONTH(start_ts) = ?
          AND duration_min >= ?
        GROUP BY 1
    """, [EVENT_MAX_DURATION_MIN, port, year, month, MIN_BUNKER_EVENT_MIN])
    order = {"0.5-6h": 0, "6-24h": 1, ">24h": 2}
    return sorted(rows, key=lambda r: order.get(r[0], 9))


def _cap_reapplied_mt(port: str, year: int, month: int) -> float:
    """Recompute called tonnage as if the duration cap were applied to EVERY
    closed event (>= MIN_BUNKER_EVENT_MIN), i.e. min(duration, cap) * pump_rate.

    The live engine caps tonnage at event-CLOSE time, so a month whose events
    closed before the cap deployed keeps its uncapped stored estimated_mt. This
    shows what the call would be if the cap were applied retroactively — the gap
    vs the stored call is the residual long-duration 'phantom' overhang."""
    rows = db.query("""
        SELECT COALESCE(SUM(LEAST(duration_min, ?) * ?), 0)
        FROM bunkering_events
        WHERE open = FALSE AND port = ?
          AND YEAR(start_ts) = ? AND MONTH(start_ts) = ?
          AND duration_min >= ?
    """, [EVENT_MAX_DURATION_MIN, DEFAULT_PUMP_RATE_MT_MIN,
          port, year, month, MIN_BUNKER_EVENT_MIN])
    return float(rows[0][0]) if rows else 0.0


def _pct(num, den):
    return None if not den else round((num / den - 1.0) * 100.0, 1)


def run(port: str, year: int, month: int) -> None:
    label = f"{_MONTHS[month]} {year}"
    W = 64
    print("=" * W)
    print(f"  CALL vs ACTUAL  -  {port.title()}  -  {label}")
    print("=" * W)

    # --- What we called -----------------------------------------------------
    count_all, called_mt = aggregate_closed_event_mt(port, year, month)
    saved = _saved_call(port, year, month)
    seasonal = seasonal_avg_monthly_volume(month, port)

    print("\nWHAT WE CALLED")
    if saved is not None:
        s_mt, s_evts, s_pct, s_at = saved
        print(f"  Snapshotted estimate : {s_mt:>12,.0f} mt   ({s_evts:,} events)")
        print(f"  vs seasonal avg      : {('%+.1f%%' % s_pct) if s_pct is not None else 'n/a':>12}")
        print(f"  snapshot saved_at    : {s_at}")
        if abs((s_mt or 0) - called_mt) > 1:
            print(f"  [note] live recompute now = {called_mt:,.0f} mt "
                  f"(events still closing / back-filling)")
    else:
        print(f"  (no saved snapshot — live recompute) : {called_mt:>12,.0f} mt "
              f"({count_all:,} events)")
        if seasonal:
            print(f"  vs seasonal avg      : {_pct(called_mt, seasonal):>+11}%")

    # --- What really happened ----------------------------------------------
    official = _official_total(port, year, month)
    print("\nWHAT REALLY HAPPENED")
    if official:
        print(f"  Official TOTAL       : {official:>12,.0f} mt")
        if seasonal:
            print(f"  vs seasonal avg      : {_pct(official, seasonal):>+11}%")
    else:
        print("  Official TOTAL       : NOT PUBLISHED YET")
        print("  -> the 'really happened' half lands when the source updates;")
        print("     re-run this study then (the dashboard auto-grades it).")

    # --- The verdict --------------------------------------------------------
    call_mt = (saved[0] if saved else called_mt) or 0.0
    if official:
        signed = (call_mt / official - 1.0) * 100.0
        print("\nVERDICT")
        print(f"  Called {call_mt:,.0f} vs actual {official:,.0f}")
        print(f"  Signed error         : {signed:>+11.1f}%   (abs {abs(signed):.1f}%)")
        if seasonal:
            print(f"  Seasonal would call  : {seasonal:>12,.0f} mt  "
                  f"-> seasonal error {_pct(seasonal, official):+.1f}% "
                  f"(did the model beat the naive seasonal call?)")

    # --- Composition: where did the called tonnage come from? --------------
    print("\nCALLED-TONNAGE COMPOSITION (closed events >= "
          f"{MIN_BUNKER_EVENT_MIN} min)")
    bands = _duration_bands(port, year, month)
    band_total = sum(float(r[2]) for r in bands) or 1.0
    print(f"  {'band':<8} {'events':>8} {'mt':>14} {'share':>8}  capped(>16h)")
    for band, n, mt, n_capped in bands:
        print(f"  {band:<8} {n:>8,} {float(mt):>14,.0f} "
              f"{float(mt)/band_total*100:>7.1f}%  {n_capped:>6,}")
    print(f"  pump rate (frozen)   : {DEFAULT_PUMP_RATE_MT_MIN} mt/min   "
          f"tonnage cap: {EVENT_MAX_DURATION_MIN} min ({EVENT_MAX_DURATION_MIN/60:.0f}h)")

    # Cap-reapplied diagnostic: the cap bites at event-close time, so a month
    # whose events closed before the cap deployed is not retroactively capped.
    capped = _cap_reapplied_mt(port, year, month)
    if band_total > 1 and abs(capped - band_total) / band_total > 0.005:
        print(f"  if cap re-applied to ALL events: {capped:>12,.0f} mt  "
              f"({_pct(capped, band_total):+.1f}% vs stored) "
              f"<- residual phantom overhang")
        if official:
            print(f"     -> cap-reapplied vs actual: {_pct(capped, official):+.1f}%")
    print("=" * W)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", default="singapore")
    ap.add_argument("--year", type=int)
    ap.add_argument("--month", type=int)
    args = ap.parse_args()

    if args.year and args.month:
        year, month = args.year, args.month
    else:
        year, month = _prev_complete_month()

    run(args.port, year, month)


if __name__ == "__main__":
    main()
