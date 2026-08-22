from __future__ import annotations
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).parent
DB_PATH  = BASE_DIR / "bunkervision.db"

# aisstream.io credentials
AISSTREAM_API_KEY = os.getenv("AISSTREAM_API_KEY", "")
AISSTREAM_WS_URL  = "wss://stream.aisstream.io/v0/stream"

# Singapore bounding box: [minLat, minLon], [maxLat, maxLon]
SG_BBOX = [[1.05, 103.50], [1.55, 104.25]]

# Map centre for dashboard
MAP_CENTRE = {"lat": 1.26, "lon": 103.82, "zoom": 11}

# Optional Mapbox token — falls back to free CARTO dark tiles if absent
MAPBOX_TOKEN = os.getenv("MAPBOX_TOKEN", "")

FLASK_SECRET = os.getenv("SECRET_KEY", "bunkervision-dev-secret")
DEBUG        = os.getenv("FLASK_DEBUG", "0") == "1"
HOST         = os.getenv("HOST", "127.0.0.1")
PORT         = int(os.getenv("PORT", "5100"))

# Shared admin token guarding /admin and every mutating API endpoint (checked
# via the X-Admin-Token header or a ?token= query param). If UNSET, protected
# endpoints return 503 — they fail CLOSED rather than running unauthenticated.
ADMIN_TOKEN = os.getenv("ADMIN_TOKEN", "")

# SQLite DB written by the independent price scrapers (integr8 / shipandbunker /
# quantum). Configurable so dev machines aren't hardwired to the VPS path; the
# default preserves production behaviour unchanged.
BUNKER_PRICES_DB = os.getenv("BUNKER_PRICES_DB", "/root/bunkervision/bunker_prices.db")

# Event detection thresholds
STOP_SOG_KT          = 0.5    # knots — vessel considered stopped below this
STOP_MIN_DURATION    = 10     # minutes stopped before we open a candidate event
PROX_RADIUS_M        = 200    # metres — alongside threshold
DETECT_POLL_SECS     = 60     # how often event_detector runs
AIS_KEEP_HOURS       = 48     # hours of raw positions to retain in DB

# Bunkering pump-rate PRIOR (mt/min).
#
# IMPORTANT: this is a *documented prior*, NOT a physical measurement and NOT a
# fitted calibration. Every event's tonnage is currently estimated as
# `duration_min * DEFAULT_PUMP_RATE_MT_MIN`. A real calibration factor can only
# be fit once we have several COMPLETE months whose official totals are
# published (see PUMP_RATE_CALIBRATION_ENABLED below). Until then we fall back
# to this prior and label the output a "calibrated proxy", not a measurement.
DEFAULT_PUMP_RATE_MT_MIN = 3.5

# ── Methodology flags ───────────────────────────────────────────────────────────
# MOST flags below default to legacy behaviour so that merely deploying this
# code does not move the live number; flip a flag (and only then) to opt into
# the improved methodology. Deliberate EXCEPTIONS that are active by default:
#   * EVENT_QUALITY_FILTER_ENABLED = True  (enabled 2026-05-31; total ~-1.6%)
#   * EVENT_MAX_DURATION_MIN — physically caps the tonnage duration term at
#     16h, which LOWERS tonnage for any event longer than the cap (multi-day
#     "events" no longer book physically impossible stems).

# Event-quality filtering: when True, events shorter than MIN_BUNKER_EVENT_MIN
# are EXCLUDED from VOLUME aggregation (they are still detected/logged/counted
# separately). Real Singapore bunkering runs ~3-12h; sub-30-min "events" are
# almost entirely AIS noise / brief drifts and are ~89% of events but only ~2%
# of modelled volume. Default False = legacy behaviour (count everything).
# Enabled 2026-05-31: makes event count + avg stem realistic; total moves ~-1.6%.
EVENT_QUALITY_FILTER_ENABLED = True

# Minimum plausible bunkering duration (minutes) for an event to contribute to
# VOLUME aggregation when EVENT_QUALITY_FILTER_ENABLED is True.
MIN_BUNKER_EVENT_MIN = 30

# Physical cap (minutes) on the DURATION TERM used to convert an event into
# tonnage: estimated_mt = min(duration_min, EVENT_MAX_DURATION_MIN) * pump_rate.
# Real Singapore bunkering runs ~3-12h; time alongside beyond 16h is AIS gaps
# or a barge parked next to the same ship, not continuous pumping — without a
# cap an 18-day "event" books a physically impossible ~90k mt stem. Events
# LONGER than the cap still close exactly as before (true duration_min is
# stored and event detection is unchanged); only the tonnage term is capped.
# NOTE: unlike most flags this is ACTIVE IMMEDIATELY and changes tonnage for
# events longer than the cap.
EVENT_MAX_DURATION_MIN = 960   # 16 hours

# Collapse obvious re-openings / double-counts of the SAME bunker<->recipient
# pair whose gap (next.start - prev.end) is below DEDUP_GAP_MIN minutes into a
# single logical event for volume aggregation. Only active when the quality
# filter is enabled. Default value chosen conservatively; flag still gated by
# EVENT_QUALITY_FILTER_ENABLED.
DEDUP_GAP_MIN = 60

# Per-port pump-rate calibration. When True AND a port has enough COMPLETE
# months with published official totals, fit a shrinkage-regularised factor
# (detected_aggregate / official, shrunk toward 1.0). When False (default) or
# when insufficient complete months exist, fall back to the DEFAULT prior.
# As of 2026-05 we have only Apr+May 2026 of AIS history and only April has a
# published official number, and April's AIS coverage is partial — so NO fit is
# yet possible. This flag is the mechanism for later, not a live calibration.
PUMP_RATE_CALIBRATION_ENABLED = False

# Minimum number of COMPLETE, officially-published months required before a
# calibration factor is fit. Below this we use the documented prior.
PUMP_RATE_MIN_CALIB_MONTHS = 3

# Shrinkage weight toward 1.0 for the calibration ratio (0 = trust raw ratio,
# 1 = ignore data and keep prior). Regularises a noisy small-sample fit.
PUMP_RATE_CALIB_SHRINKAGE = 0.5

# Partial-month guard: a month whose detected AIS event coverage clearly does
# not span the full calendar month is flagged `partial` / skipped rather than
# graded as a complete month. Default False = legacy (grade as complete).
PARTIAL_MONTH_GUARD_ENABLED = False

# A month is considered fully covered only if detected events span at least this
# fraction of the calendar month (first event near day 1, last event near
# month-end). Used only when PARTIAL_MONTH_GUARD_ENABLED is True.
MONTH_COVERAGE_MIN_FRACTION = 0.9

# ── Calibrated nowcast (salvage prototype, 2026-06) ─────────────────────────
# A demand nowcast that replaces the broken duration×rate "measurement" with a
# calibrated forecast of the OFFICIAL monthly series (76 months of MPA history,
# strong seasonality+trend), optionally nudged by an AIS leading-indicator that
# earns weight only as graded months accrue. See nowcast_model.py + docs/.
#
#   nowcast = level_model(month) × (1 + α · z_ais)
#
# The level model (log-space damped-ETS/Theta blend) was selected out-of-sample
# in the 2026-06 bake-off: MAE 4.07% vs 8.35% for the old seasonal baseline,
# near-zero bias. The AIS term is a small, shrinkage-regularised deviation nudge:
# with only ~2 months of AIS history α≈0 today, so the nowcast == the level model.
# OFF by default — flipping this changes /api/demand; the shadow backtest panel
# (/api/accuracy-backtest) compares it to the live model without touching either.
USE_CALIBRATED_NOWCAST = True

# AIS deviation earns weight only after this many GRADED months of AIS history
# (out-of-sample evidence that the signal helps). Below this, α = 0.
NOWCAST_AIS_MIN_GRADED_MONTHS = 6

# Hard cap on the AIS nudge magnitude (fraction of level). Even fully "earned",
# the AIS term may move the nowcast at most ±10% — it informs, never dominates.
NOWCAST_AIS_MAX_ALPHA = 0.10

# Minimum months of official history before the level model will fit; below this
# it falls back to a trend-adjusted seasonal mean.
NOWCAST_LEVEL_MIN_TRAIN = 30

# MPA URLs
MPA_BUNKER_STATS_URL    = "https://www.mpa.gov.sg/maritime-singapore/what-we-do/develop-singapore-as-an-international-maritime-centre/port-statistics/bunkering"
MPA_VESSEL_REGISTRY_URL = "https://www.mpa.gov.sg/port-marine-ops/port-operations/marine-services/bunkering/list-of-licensed-bunker-suppliers"

# ── Multi-port definitions ─────────────────────────────────────────────────────
# bbox format: [[minLat, minLon], [maxLat, maxLon]]  (matches aisstream.io convention)
PORTS: dict[str, dict] = {
    "singapore": {
        "name": "Singapore",
        "bbox": [[1.05, 103.50], [1.55, 104.25]],
        "centre": {"lat": 1.26, "lon": 103.82, "zoom": 11},
        "has_registry": True,
        "data_source": "mysql_mpa",
        "data_frequency": "monthly",
    },
    "fujairah": {
        "name": "Fujairah",
        "bbox": [[24.90, 56.15], [25.35, 56.65]],
        "centre": {"lat": 25.12, "lon": 56.35, "zoom": 11},
        "has_registry": True,
        "data_source": "mysql_fujairah",
        "data_frequency": "monthly",
        # AIS coverage intermittent due to regional conflict — estimates will be low during outages
        "ais_coverage_note": "patchy",
    },
    "rotterdam": {
        "name": "Rotterdam",
        "bbox": [[51.78, 3.85], [52.05, 4.80]],
        "centre": {"lat": 51.92, "lon": 4.28, "zoom": 11},
        "has_registry": False,
        "data_source": "excel",
        "data_frequency": "quarterly",
    },
    "antwerp": {
        "name": "Antwerp",
        "bbox": [[51.13, 4.18], [51.45, 4.65]],
        "centre": {"lat": 51.27, "lon": 4.40, "zoom": 11},
        "has_registry": False,
        "data_source": "excel",
        "data_frequency": "quarterly",
    },
    "panama": {
        "name": "Panama",
        "bbox": [[8.65, -80.10], [9.55, -79.40]],
        "centre": {"lat": 9.08, "lon": -79.68, "zoom": 10},
        "has_registry": False,
        "data_source": "excel",
        "data_frequency": "monthly",
    },
    "khor_fakkan": {
        "name": "Khor Fakkan",
        "bbox": [[25.32, 56.33], [25.43, 56.43]],
        "centre": {"lat": 25.37, "lon": 56.38, "zoom": 12},
        "has_registry": True,
        "data_source": None,
        "data_frequency": None,
        # AIS coverage intermittent due to regional conflict
        "ais_coverage_note": "patchy",
    },
    "gibraltar": {
        "name": "Gibraltar",
        "bbox": [[35.88, -5.60], [36.22, -5.10]],
        "centre": {"lat": 36.10, "lon": -5.35, "zoom": 12},
        "has_registry": False,
        "data_source": None,
        "data_frequency": None,
    },
    "houston": {
        "name": "Houston",
        "bbox": [[29.30, -95.30], [29.80, -94.50]],
        "centre": {"lat": 29.55, "lon": -94.90, "zoom": 10},
        "has_registry": False,
        "data_source": None,
        "data_frequency": None,
    },
    "hong_kong": {
        "name": "Hong Kong",
        "bbox": [[22.15, 113.90], [22.55, 114.50]],
        "centre": {"lat": 22.30, "lon": 114.15, "zoom": 11},
        "has_registry": False,
        "data_source": None,
        "data_frequency": None,
    },
    "las_palmas": {
        "name": "Las Palmas",
        "bbox": [[27.88, -15.65], [28.18, -15.30]],
        "centre": {"lat": 28.05, "lon": -15.47, "zoom": 12},
        "has_registry": False,
        "data_source": None,
        "data_frequency": None,
    },
    "zhoushan": {
        "name": "Zhoushan",
        "bbox": [[29.70, 121.80], [30.30, 122.60]],
        "centre": {"lat": 29.98, "lon": 122.12, "zoom": 11},
        "has_registry": False,
        "data_source": None,
        "data_frequency": None,
    },
    "malta_offshore": {
        "name": "Malta Offshore",
        "db_port": "Malta Offshore",
        "bbox": [[35.70, 14.10], [36.10, 14.70]],
        "centre": {"lat": 35.90, "lon": 14.40, "zoom": 10},
        "has_registry": False,
        "data_source": "eurostat",
        "data_frequency": "monthly",
    },
    "limassol": {
        "name": "Limassol",
        "db_port": "Limassol",
        "bbox": [[34.55, 33.00], [34.75, 33.25]],
        "centre": {"lat": 34.65, "lon": 33.10, "zoom": 11},
        "has_registry": False,
        "data_source": "eurostat",
        "data_frequency": "monthly",
    },
    "istanbul": {
        "name": "Istanbul",
        "db_port": "Istanbul",
        "bbox": [[40.85, 28.50], [41.30, 29.30]],
        "centre": {"lat": 41.05, "lon": 28.90, "zoom": 10},
        "has_registry": False,
        "data_source": "epdk",
        "data_frequency": "monthly",
    },
    "tallinn": {
        "name": "Tallinn",
        "db_port": "Tallinn",
        "bbox": [[59.35, 24.50], [59.55, 24.85]],
        "centre": {"lat": 59.44, "lon": 24.70, "zoom": 11},
        "has_registry": False,
        "data_source": "eurostat",
        "data_frequency": "monthly",
    },
    "algoa_bay": {
        "name": "Algoa Bay",
        "bbox": [[-34.10, 25.45], [-33.50, 26.55]],
        "centre": {"lat": -33.82, "lon": 25.98, "zoom": 10},
        "has_registry": False,
        "data_source": None,
        "data_frequency": None,
    },
}

# All bounding boxes sent as one aisstream.io subscription
ALL_BBOXES = [p["bbox"] for p in PORTS.values()]


def port_for_coord(lat: float, lon: float) -> str | None:
    """Return the port key whose bounding box contains (lat, lon), or None."""
    for key, cfg in PORTS.items():
        bb = cfg["bbox"]
        if bb[0][0] <= lat <= bb[1][0] and bb[0][1] <= lon <= bb[1][1]:
            return key
    return None
