"""
Background thread that polls ais_positions every DETECT_POLL_SECS seconds,
detects bunkering events (stop + proximity), and writes to bunkering_events.
"""
from __future__ import annotations

import math
import threading
import time
from datetime import datetime, timezone

from loguru import logger

import db
from config import (
    DEFAULT_PUMP_RATE_MT_MIN,
    DETECT_POLL_SECS,
    EVENT_MAX_DURATION_MIN,
    PROX_RADIUS_M,
    PUMP_RATE_CALIB_SHRINKAGE,
    PUMP_RATE_CALIBRATION_ENABLED,
    PUMP_RATE_MIN_CALIB_MONTHS,
    STOP_MIN_DURATION,
    STOP_SOG_KT,
    port_for_coord,
)


def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6_371_000
    p = math.pi / 180
    a = (
        math.sin((lat2 - lat1) * p / 2) ** 2
        + math.cos(lat1 * p) * math.cos(lat2 * p) * math.sin((lon2 - lon1) * p / 2) ** 2
    )
    return 2 * R * math.asin(math.sqrt(a))


def _avg_pump_rate(port: str | None = None) -> float:
    """
    Return the pump rate (mt/min) used to convert event duration into tonnage.

    HONEST NOTE ON METHODOLOGY
    --------------------------
    This is a CALIBRATED PROXY, not a physical measurement. We do not observe
    actual mass-flow; we infer tonnage as `duration_min * pump_rate`. The only
    legitimate way to set `pump_rate` is to compare a COMPLETE month of detected
    aggregate signal against that month's published official total and scale to
    match — i.e. `pump_rate = prior * (official / detected_at_prior)`, shrunk
    toward the prior to avoid over-fitting a single noisy month.

    The previous implementation was CIRCULAR: it averaged
    `estimated_mt / duration_min` over closed events, but `estimated_mt` is
    itself defined as `duration_min * pump_rate`, so the ratio is the constant
    `pump_rate` for every row — the "average" just returned the prior. The MPA
    fallback divided by `mpa_monthly.ships_count`, which the loader always
    inserts as NULL, so it never fired either. Net effect: a frozen 3.5 mt/min.

    New design:
      * When PUMP_RATE_CALIBRATION_ENABLED is True AND we have at least
        PUMP_RATE_MIN_CALIB_MONTHS complete, officially-published months for the
        port, fit a shrinkage-regularised factor (see _fit_pump_rate).
      * Otherwise fall back to DEFAULT_PUMP_RATE_MT_MIN, which is a DOCUMENTED
        PRIOR, not a calibration.

    As of 2026-05 we have only Apr+May 2026 of AIS history, only April has a
    published official number, and April's AIS coverage is partial — so the fit
    is NOT yet possible and we return the prior.
    """
    if PUMP_RATE_CALIBRATION_ENABLED:
        fitted = _fit_pump_rate(port)
        if fitted is not None:
            return fitted
    return DEFAULT_PUMP_RATE_MT_MIN


def capped_tonnage_mt(duration_min: float, pump_rate: float) -> float:
    """
    Convert an event duration into the tonnage proxy, with the duration term
    physically capped at EVENT_MAX_DURATION_MIN:

        estimated_mt = min(duration_min, EVENT_MAX_DURATION_MIN) * pump_rate

    Time alongside beyond the cap (default 16h) is AIS gaps or a barge parked
    next to the same ship, not continuous pumping — an 18-day "event" must not
    book a ~90k mt stem. Events longer than the cap still close as before with
    their TRUE duration recorded; only the tonnage term is capped.
    """
    return round(min(duration_min, EVENT_MAX_DURATION_MIN) * pump_rate, 1)


def _fit_pump_rate(port: str | None) -> float | None:
    """
    Mechanism for a per-port calibration factor (returns None until fittable).

    Ratio method: for each COMPLETE month with a published official total,
    compute (official_mt / detected_mt_at_prior). The detected signal at the
    prior is `sum(duration_min) * DEFAULT_PUMP_RATE_MT_MIN`, so the calibrated
    rate is `DEFAULT_PUMP_RATE_MT_MIN * mean(official / detected_at_prior)`,
    then shrunk toward the prior by PUMP_RATE_CALIB_SHRINKAGE.

    Returns None (caller uses the prior) unless there are at least
    PUMP_RATE_MIN_CALIB_MONTHS complete official months — which is NOT the case
    today. This is deliberately conservative: a single partial month (e.g. the
    ramp-up April 2026) must never drive the live number.

    NOTE: "complete month" detection here only counts months that have BOTH a
    published official total and full AIS coverage. Coverage completeness is the
    responsibility of the partial-month guard in demand_model; this function
    simply requires enough official months to exist and returns None otherwise.
    """
    if port is None:
        return None

    # Official monthly totals available for this port (Singapore falls back to
    # mpa_monthly). These define which months *could* anchor a calibration.
    rows = db.query("""
        SELECT YEAR(month), MONTH(month), volume_mt
        FROM port_bunker_sales
        WHERE port = ? AND fuel_type = 'TOTAL' AND volume_mt > 0
    """, [port])
    if not rows and port == "singapore":
        rows = db.query("""
            SELECT YEAR(month), MONTH(month), volume_mt
            FROM mpa_monthly
            WHERE fuel_type = 'TOTAL' AND volume_mt > 0
        """)

    ratios: list[float] = []
    for yr, mo, official in rows:
        det = db.query("""
            SELECT COALESCE(SUM(duration_min), 0)
            FROM bunkering_events
            WHERE open = FALSE AND port = ?
              AND YEAR(start_ts) = ? AND MONTH(start_ts) = ?
        """, [port, yr, mo])
        total_min = float(det[0][0]) if det and det[0][0] else 0.0
        detected_at_prior = total_min * DEFAULT_PUMP_RATE_MT_MIN
        if detected_at_prior > 0 and official and official > 0:
            ratios.append(float(official) / detected_at_prior)

    # Require enough independent official months before trusting a fit.
    if len(ratios) < PUMP_RATE_MIN_CALIB_MONTHS:
        return None

    raw_ratio = sum(ratios) / len(ratios)
    # Shrink the ratio toward 1.0 (i.e. toward the prior) to regularise.
    w = PUMP_RATE_CALIB_SHRINKAGE
    shrunk_ratio = w * 1.0 + (1.0 - w) * raw_ratio
    return DEFAULT_PUMP_RATE_MT_MIN * shrunk_ratio


def aggregate_closed_event_mt(port: str, year: int, month: int) -> tuple[int, float]:
    """
    Return (event_count, volume_mt) of CLOSED events for a port-month, applying
    event-quality filtering and pair-dedup when EVENT_QUALITY_FILTER_ENABLED.

    Behaviour is gated so the legacy path (flag False) returns exactly the old
    aggregate: COUNT(*) and SUM(estimated_mt) over all closed events.

    When the flag is True:
      * `event_count` counts ALL closed events (junk events are still logged /
        counted — only their VOLUME is excluded), matching the task spec.
      * `volume_mt` sums only events with duration >= MIN_BUNKER_EVENT_MIN,
        after collapsing same bunker<->recipient re-openings whose inter-event
        gap is < DEDUP_GAP_MIN minutes into one logical event.
    """
    from config import (
        DEDUP_GAP_MIN,
        EVENT_QUALITY_FILTER_ENABLED,
        MIN_BUNKER_EVENT_MIN,
    )

    base = db.query("""
        SELECT COUNT(*), COALESCE(SUM(estimated_mt), 0)
        FROM bunkering_events
        WHERE open = FALSE AND port = ?
          AND YEAR(start_ts) = ? AND MONTH(start_ts) = ?
    """, [port, year, month])
    event_count = int(base[0][0]) if base else 0
    legacy_mt = float(base[0][1]) if base else 0.0

    if not EVENT_QUALITY_FILTER_ENABLED:
        return event_count, legacy_mt

    # Quality-filtered, deduped volume. Count still reflects ALL events.
    rows = db.query("""
        SELECT bunker_mmsi, recipient_mmsi, start_ts, end_ts, duration_min, estimated_mt
        FROM bunkering_events
        WHERE open = FALSE AND port = ?
          AND YEAR(start_ts) = ? AND MONTH(start_ts) = ?
          AND duration_min >= ?
        ORDER BY bunker_mmsi, recipient_mmsi, start_ts
    """, [port, year, month, MIN_BUNKER_EVENT_MIN])

    volume_mt = 0.0
    prev = None  # (bunker, recipient, end_ts, mt)
    gap_secs = DEDUP_GAP_MIN * 60
    for b, rc, st, en, _dur, mt in rows:
        mt = float(mt or 0.0)
        if (prev is not None and prev[0] == b and prev[1] == rc
                and st is not None and prev[2] is not None
                and (st - prev[2]).total_seconds() < gap_secs):
            # Re-opening of the same pair: fold tonnage into the prior event.
            volume_mt += mt
            prev = (b, rc, en or prev[2], prev[3] + mt)
        else:
            volume_mt += mt
            prev = (b, rc, en, mt)

    return event_count, volume_mt


def _get_stopped_bunkers(window_minutes: int = 30) -> list[dict]:
    """Return bunker vessels that have been at SOG < threshold for >= STOP_MIN_DURATION."""
    rows = db.query(f"""
        SELECT
            p.imo, p.mmsi, p.name,
            AVG(p.lat) AS lat, AVG(p.lon) AS lon,
            MIN(p.ts) AS first_ts,
            MAX(p.ts) AS last_ts
        FROM ais_positions p
        WHERE p.ts >= NOW() - INTERVAL '{window_minutes} minutes'
          AND p.sog < {STOP_SOG_KT}
          AND (
              p.imo  IN (SELECT imo  FROM bunker_vessels WHERE imo  NOT LIKE 'NAME:%')
           OR p.mmsi IN (SELECT mmsi FROM bunker_vessels WHERE mmsi IS NOT NULL)
           OR UPPER(p.name) IN (SELECT UPPER(name) FROM bunker_vessels WHERE name IS NOT NULL)
          )
        GROUP BY p.imo, p.mmsi, p.name
        HAVING DATEDIFF('minute', MIN(p.ts), MAX(p.ts)) >= {STOP_MIN_DURATION}
    """)
    return [
        {"imo": r[0], "mmsi": r[1], "name": r[2],
         "lat": r[3], "lon": r[4],
         "first_ts": r[5], "last_ts": r[6]}
        for r in rows
    ]


def _get_nearby_vessels(lat: float, lon: float, radius_m: float = PROX_RADIUS_M) -> list[dict]:
    """
    Find non-bunker vessels near a point using vessel_positions (current snapshot).
    Filters by bounding box then exact Haversine distance.
    """
    deg_margin = radius_m / 111_000
    rows = db.query(f"""
        SELECT vp.mmsi, vp.imo, vp.name, vp.lat, vp.lon, vp.sog
        FROM vessel_positions vp
        LEFT JOIN bunker_vessels b
          ON b.imo = vp.imo
          OR b.mmsi = vp.mmsi
          OR UPPER(b.name) = UPPER(vp.name)
        WHERE b.imo IS NULL
          AND vp.lat BETWEEN {lat - deg_margin} AND {lat + deg_margin}
          AND vp.lon BETWEEN {lon - deg_margin} AND {lon + deg_margin}
          AND vp.sog < 1.5
          AND vp.ts >= NOW() - INTERVAL '30 minutes'
    """)
    results = []
    for r in rows:
        dist = _haversine_m(lat, lon, r[3], r[4])
        if dist <= radius_m:
            results.append({
                "mmsi": r[0], "imo": r[1], "name": r[2],
                "lat": r[3], "lon": r[4], "sog": r[5], "dist_m": dist,
            })
    return results


def _open_event_exists(bunker_mmsi: str) -> bool:
    rows = db.query(
        "SELECT 1 FROM bunkering_events WHERE bunker_mmsi = ? AND open = TRUE LIMIT 1",
        [bunker_mmsi],
    )
    return bool(rows)


def _draught_vol_estimate(mmsi: str, start_ts: datetime, end_ts: datetime, dwt: float | None) -> tuple[float | None, float | None, float | None]:
    """
    Look up bracketing draught readings for a bunker vessel around an event window.
    Returns (draught_start, draught_end, vol_est_mt) or (None, None, None).

    Volume approximation:
      WPA (waterplane area) ≈ DWT / (full_load_draught * fuel_density)
      full_load_draught ≈ DWT / 1200  (rough rule for SG bunker barges)
      vol_mt = Δdraught * WPA * fuel_density (0.9 t/m³)
    """
    # Widen the window slightly to catch readings just before/after
    window_before = start_ts.replace(tzinfo=timezone.utc) if start_ts.tzinfo else start_ts
    window_after  = end_ts.replace(tzinfo=timezone.utc)   if end_ts.tzinfo   else end_ts

    rows_before = db.query("""
        SELECT draught FROM draught_history
        WHERE mmsi = ? AND ts <= ? AND ts >= ? - INTERVAL '2 hours'
        ORDER BY ts DESC LIMIT 1
    """, [mmsi, window_before, window_before])

    rows_after = db.query("""
        SELECT draught FROM draught_history
        WHERE mmsi = ? AND ts >= ? AND ts <= ? + INTERVAL '2 hours'
        ORDER BY ts ASC LIMIT 1
    """, [mmsi, window_after, window_after])

    d_start = float(rows_before[0][0]) if rows_before and rows_before[0][0] else None
    d_end   = float(rows_after[0][0])  if rows_after  and rows_after[0][0]  else None

    if d_start is None or d_end is None or d_start <= d_end:
        return d_start, d_end, None  # no decrease = no meaningful pump-out reading

    delta = d_start - d_end
    vol_est = None
    if dwt and dwt > 0:
        full_draught = max(1.5, dwt / 1200.0)
        wpa = dwt / (full_draught * 0.9)
        vol_est = round(delta * wpa * 0.9, 0)

    return d_start, d_end, vol_est


def _close_stale_events() -> None:
    """Close any open events where the bunker vessel has been moving for 5+ minutes."""
    open_events = db.query("""
        SELECT e.id, e.bunker_mmsi, e.start_ts, bv.capacity_mt, e.port
        FROM bunkering_events e
        LEFT JOIN bunker_vessels bv ON bv.mmsi = e.bunker_mmsi OR bv.imo = e.bunker_imo
        WHERE e.open = TRUE
    """)

    for ev_id, bunker_mmsi, start_ts, dwt, port in open_events:
        pump_rate = _avg_pump_rate(port)
        rows = db.query("""
            SELECT MAX(sog), MAX(ts) FROM ais_positions
            WHERE mmsi = ? AND ts >= NOW() - INTERVAL '10 minutes'
        """, [bunker_mmsi])
        if not rows or rows[0][0] is None:
            continue
        max_sog, last_ts = rows[0]
        if max_sog is not None and max_sog > 1.0:
            now = datetime.now(timezone.utc)
            start_aware = start_ts.replace(tzinfo=timezone.utc) if start_ts.tzinfo is None else start_ts
            duration_min = max(1, int((now - start_aware).total_seconds() / 60))
            # Tonnage uses the duration term capped at EVENT_MAX_DURATION_MIN;
            # the event itself still closes with its true duration_min.
            estimated_mt = capped_tonnage_mt(duration_min, pump_rate)

            d_start, d_end, draught_vol = _draught_vol_estimate(bunker_mmsi, start_ts, now, dwt)
            delta_m = round(d_start - d_end, 2) if d_start and d_end else None

            db.execute("""
                UPDATE bunkering_events
                SET open = FALSE, end_ts = ?, duration_min = ?, estimated_mt = ?,
                    draught_start = ?, draught_end = ?, draught_delta_m = ?, draught_vol_est = ?
                WHERE id = ?
            """, [now, duration_min, estimated_mt, d_start, d_end, delta_m, draught_vol, ev_id])

            draught_note = f" | draught {d_start}→{d_end}m Δ{delta_m}m ~{draught_vol} mt" if delta_m else ""
            logger.info(f"Closed event {ev_id}: {duration_min} min, ~{estimated_mt} mt (duration est){draught_note}")


def _detect_cycle() -> None:
    stopped = _get_stopped_bunkers()
    if not stopped:
        return

    for bunker in stopped:
        if _open_event_exists(bunker["mmsi"]):
            continue
        nearby = _get_nearby_vessels(bunker["lat"], bunker["lon"])
        if not nearby:
            continue
        # Take the closest non-bunker vessel
        recipient = min(nearby, key=lambda v: v["dist_m"])
        ts = datetime.now(timezone.utc)
        port = port_for_coord(bunker["lat"], bunker["lon"]) or "singapore"
        db.execute("""
            INSERT INTO bunkering_events
              (id, bunker_imo, bunker_name, bunker_mmsi,
               recipient_mmsi, recipient_name,
               start_ts, lat, lon, open, port)
            VALUES (nextval('event_id_seq'), ?, ?, ?, ?, ?, ?, ?, ?, TRUE, ?)
        """, [
            bunker["imo"], bunker["name"], bunker["mmsi"],
            recipient["mmsi"], recipient["name"],
            ts, bunker["lat"], bunker["lon"], port,
        ])
        logger.info(
            f"New bunkering event: {bunker['name']} ({bunker['imo']}) "
            f"alongside {recipient['name']} ({recipient['mmsi']}) "
            f"dist={recipient['dist_m']:.0f}m"
        )


def _purge_old_positions() -> None:
    from config import AIS_KEEP_HOURS
    db.execute(f"""
        DELETE FROM ais_positions
        WHERE ts < NOW() - INTERVAL '{AIS_KEEP_HOURS} hours'
    """)


def _run_detector() -> None:
    logger.info("Event detector started")
    while True:
        try:
            _close_stale_events()
            _detect_cycle()
            _purge_old_positions()
        except Exception as e:
            logger.error(f"Detector error: {e}")
        time.sleep(DETECT_POLL_SECS)


def start_detector_thread() -> threading.Thread:
    t = threading.Thread(target=_run_detector, daemon=True, name="event-detector")
    t.start()
    return t
