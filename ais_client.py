"""
WebSocket client for aisstream.io.
Subscribes to the Singapore bounding box and:
  - Writes positions for known bunker vessels to ais_positions
  - Maintains vessel_cache for all vessels seen (for proximity lookup)
  - Pushes position updates to a Queue consumed by the SSE endpoint
"""
from __future__ import annotations

import json
import queue
import threading
import time
from datetime import datetime, timezone

import websocket
from loguru import logger
from tenacity import retry, stop_never, wait_exponential

import db
from config import AISSTREAM_API_KEY, AISSTREAM_WS_URL, ALL_BBOXES, PORTS, port_for_coord

# Shared queue — Flask SSE endpoint drains this
position_queue: queue.Queue[dict] = queue.Queue(maxsize=500)

# In-memory last-known positions for all vessels in the box
# mmsi → {mmsi, imo, name, lat, lon, sog, cog, heading, nav_status, draught, state, ts}
_live: dict[str, dict] = {}
_live_lock = threading.Lock()

# In-memory set of known bunker vessel names (uppercase) — populated at startup
_bunker_names: set[str] = set()
_bunker_names_lock = threading.Lock()


_port_for_coord = port_for_coord


def refresh_bunker_names() -> None:
    """Reload bunker vessel name set from DB into memory."""
    rows = db.query("SELECT UPPER(name) FROM bunker_vessels WHERE name IS NOT NULL")
    with _bunker_names_lock:
        _bunker_names.clear()
        _bunker_names.update(r[0] for r in rows)
    logger.info(f"Bunker name set refreshed: {len(_bunker_names)} vessels")


def is_bunker_vessel(mmsi: str, imo: str, name: str) -> bool:
    rows = db.query(
        "SELECT 1 FROM bunker_vessels WHERE imo = ? OR mmsi = ? LIMIT 1",
        [imo, mmsi],
    )
    if rows:
        return True
    if name:
        with _bunker_names_lock:
            return name.upper() in _bunker_names
    return False


def get_live_positions() -> list[dict]:
    with _live_lock:
        return list(_live.values())


# ── Message handlers ───────────────────────────────────────────────────────────

def _handle_position(msg: dict) -> None:
    meta  = msg.get("MetaData", {})
    pos   = msg.get("Message", {}).get("PositionReport", {})
    if not pos:
        return

    mmsi    = str(meta.get("MMSI", ""))
    name    = meta.get("ShipName", "").strip()
    lat     = meta.get("latitude",  pos.get("Latitude",  0.0))
    lon     = meta.get("longitude", pos.get("Longitude", 0.0))
    sog     = pos.get("Sog", 0.0)
    cog     = pos.get("Cog", 0.0)
    heading = pos.get("TrueHeading", 511)
    nav_st  = pos.get("NavigationalStatus", 0)
    ts      = datetime.now(timezone.utc)

    # Resolve IMO + draught from in-memory cache (no DB hit on hot path)
    with _live_lock:
        cached = _live.get(mmsi, {})
    imo     = cached.get("imo", "")
    draught = cached.get("draught")

    port = _port_for_coord(lat, lon) or cached.get("port", "")

    entry = {
        "mmsi": mmsi, "imo": imo, "name": name or cached.get("name", ""),
        "lat": lat, "lon": lon,
        "sog": sog, "cog": cog, "heading": heading,
        "nav_status": nav_st, "draught": draught,
        "state": "underway", "port": port, "ts": ts.isoformat(),
    }

    # ── Update live map immediately — this must happen before any DB op ──────
    with _live_lock:
        _live[mmsi] = entry

    # Push to SSE queue immediately (non-blocking — drop if full)
    try:
        position_queue.put_nowait(entry)
    except queue.Full:
        pass

    # ── DB writes are best-effort; errors must not propagate ─────────────────

    # Current position for ALL vessels — used by proximity detection
    try:
        db.execute(
            """INSERT INTO vessel_positions (mmsi, imo, name, lat, lon, sog, ts)
               VALUES (?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT (mmsi) DO UPDATE SET
                   imo  = CASE WHEN excluded.imo  != '' THEN excluded.imo  ELSE vessel_positions.imo  END,
                   name = CASE WHEN excluded.name != '' THEN excluded.name ELSE vessel_positions.name END,
                   lat  = excluded.lat, lon = excluded.lon,
                   sog  = excluded.sog, ts  = excluded.ts""",
            [mmsi, imo, entry["name"], lat, lon, sog, ts],
        )
    except Exception as e:
        logger.debug(f"vessel_positions write skipped: {e}")

    # Full position history — bunker vessels only (used for stop duration & draught)
    try:
        if is_bunker_vessel(mmsi, imo, entry["name"]):
            db.execute(
                """INSERT INTO ais_positions
                   (id, imo, mmsi, name, lat, lon, sog, cog, heading, nav_status, ts)
                   VALUES (nextval('event_id_seq'), ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                [imo, mmsi, entry["name"], lat, lon, sog, cog, heading, nav_st, ts],
            )
    except Exception as e:
        logger.debug(f"ais_positions write skipped: {e}")

    try:
        db.execute(
            """INSERT INTO vessel_cache (mmsi, imo, name, last_seen)
               VALUES (?, ?, ?, ?)
               ON CONFLICT (mmsi) DO UPDATE SET
                   name      = CASE WHEN excluded.name != '' THEN excluded.name ELSE vessel_cache.name END,
                   last_seen = excluded.last_seen""",
            [mmsi, imo, entry["name"], ts],
        )
    except Exception as e:
        logger.debug(f"vessel_cache write skipped: {e}")


def _handle_static(msg: dict) -> None:
    meta   = msg.get("MetaData", {})
    static = msg.get("Message", {}).get("ShipStaticData", {})
    if not static:
        return

    mmsi      = str(meta.get("MMSI", ""))
    imo_raw   = static.get("ImoNumber", 0)
    imo       = str(imo_raw) if imo_raw else ""
    name      = static.get("Name", "").strip()
    type_code = static.get("Type", 0)
    draught   = static.get("MaximumStaticDraught")  # metres, 0.1m resolution
    ts        = datetime.now(timezone.utc)

    if not mmsi:
        return

    # Update in-memory entry with richer data
    with _live_lock:
        if mmsi in _live:
            if imo:   _live[mmsi]["imo"]     = imo
            if name:  _live[mmsi]["name"]    = name
            if draught is not None:
                _live[mmsi]["draught"] = draught

    try:
        db.execute(
            """INSERT INTO vessel_cache (mmsi, imo, name, type_code, draught, last_seen)
               VALUES (?, ?, ?, ?, ?, ?)
               ON CONFLICT (mmsi) DO UPDATE SET
                   imo       = CASE WHEN excluded.imo != '' THEN excluded.imo ELSE vessel_cache.imo END,
                   name      = CASE WHEN excluded.name != '' THEN excluded.name ELSE vessel_cache.name END,
                   type_code = excluded.type_code,
                   draught   = COALESCE(excluded.draught, vessel_cache.draught),
                   last_seen = excluded.last_seen""",
            [mmsi, imo, name, type_code, draught, ts],
        )
    except Exception as e:
        logger.debug(f"vessel_cache static write skipped: {e}")

    # Log draught history for bunker vessels (used for independent volume cross-check)
    if draught and draught > 0 and is_bunker_vessel(mmsi, imo, name):
        try:
            db.execute(
                "INSERT INTO draught_history (id, mmsi, draught, ts) "
                "VALUES (nextval('event_id_seq'), ?, ?, ?)",
                [mmsi, draught, ts],
            )
            logger.debug(f"Draught logged: {name} ({mmsi}) = {draught}m")
        except Exception as e:
            logger.debug(f"draught_history write skipped: {e}")


# ── WebSocket callbacks ────────────────────────────────────────────────────────

def _on_open(ws: websocket.WebSocketApp) -> None:
    logger.info(f"aisstream.io connected — subscribing to {len(ALL_BBOXES)} port bboxes")
    payload = {
        "APIKey": AISSTREAM_API_KEY,
        "BoundingBoxes": ALL_BBOXES,
        "FilterMessageTypes": ["PositionReport", "ShipStaticData"],
    }
    ws.send(json.dumps(payload))


def _on_message(ws: websocket.WebSocketApp, raw: str) -> None:
    try:
        msg = json.loads(raw)
    except json.JSONDecodeError:
        return
    try:
        mtype = msg.get("MessageType", "")
        if mtype == "PositionReport":
            _handle_position(msg)
        elif mtype == "ShipStaticData":
            _handle_static(msg)
    except Exception as e:
        logger.debug(f"Message handler error: {e}")


def _on_error(ws: websocket.WebSocketApp, err: Exception) -> None:
    logger.warning(f"aisstream.io error: {err}")


def _on_close(ws: websocket.WebSocketApp, code: int, msg: str) -> None:
    logger.info(f"aisstream.io closed: {code} {msg}")


# ── Background thread with reconnect ──────────────────────────────────────────

def _run_client() -> None:
    if not AISSTREAM_API_KEY:
        logger.warning("AISSTREAM_API_KEY not set — AIS client disabled")
        return

    while True:
        try:
            ws = websocket.WebSocketApp(
                AISSTREAM_WS_URL,
                on_open=_on_open,
                on_message=_on_message,
                on_error=_on_error,
                on_close=_on_close,
            )
            ws.run_forever(ping_interval=30, ping_timeout=10)
        except Exception as e:
            logger.error(f"AIS client crashed: {e}")
        logger.info("Reconnecting in 15s…")
        time.sleep(15)


def start_ais_thread() -> threading.Thread:
    t = threading.Thread(target=_run_client, daemon=True, name="ais-client")
    t.start()
    return t


if __name__ == "__main__":
    import sys
    if not AISSTREAM_API_KEY:
        print("Set AISSTREAM_API_KEY in .env first")
        sys.exit(1)
    logger.info("Starting AIS client test — press Ctrl+C to stop")
    _run_client()
