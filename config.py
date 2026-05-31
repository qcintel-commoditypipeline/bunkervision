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

# Event detection thresholds
STOP_SOG_KT          = 0.5    # knots — vessel considered stopped below this
STOP_MIN_DURATION    = 10     # minutes stopped before we open a candidate event
PROX_RADIUS_M        = 200    # metres — alongside threshold
DETECT_POLL_SECS     = 60     # how often event_detector runs
AIS_KEEP_HOURS       = 48     # hours of raw positions to retain in DB

# Bunkering pump-rate prior (mt/min) used before enough event data to calibrate
DEFAULT_PUMP_RATE_MT_MIN = 3.5

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
