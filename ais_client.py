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
import liveness
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

# ── Ingest observability (for /healthz, see liveness.py) ──────────────────────
# Cumulative counters since process start + trailing-window buckets. `received`
# counts EVERY typed AIS frame the socket delivers (PositionReport + ShipStaticData)
# — it is the feed-liveness signal. `write_errors` counts DB writes that failed;
# those used to vanish into logger.debug("… write skipped") (2026-08-19: a feed
# silent for days and a DB that cannot be written both looked like a quiet port).
_stats = {"received": 0, "write_errors": 0, "connects": 0,
          "last_msg_epoch": None,     # time.time() of the last AIS message
          "started_epoch": None}      # time.time() the client thread started
_stats_lock = threading.Lock()
_window = liveness.WindowCounter()
_client_thread: threading.Thread | None = None
_last_write_warn = {"t": 0.0}


def _note_message() -> None:
    now = time.time()
    with _stats_lock:
        _stats["received"] += 1
        _stats["last_msg_epoch"] = now
    _window.add("received", now=now)


def _note_write_error(what: str, err: Exception) -> None:
    """Count a failed DB write (cumulative + window) and say so in the journal at
    WARNING — rate-limited to one line per minute so a dead DB does not flood, but
    never silent: a non-zero window count flips /healthz to degraded."""
    now = time.time()
    with _stats_lock:
        _stats["write_errors"] += 1
        n = _stats["write_errors"]
        warn = now - _last_write_warn["t"] >= 60
        if warn:
            _last_write_warn["t"] = now
    _window.add("write_errors", now=now)
    if warn:
        logger.warning(f"DB write failed ({what}; {n} total this process): {err}")
    else:
        logger.debug(f"{what} write skipped: {err}")


def _last_msg_iso() -> str | None:
    e = _stats.get("last_msg_epoch")
    return None if e is None else datetime.fromtimestamp(e, tz=timezone.utc).isoformat()


def ingest_enabled() -> bool:
    """True when the live ingest is CONFIGURED to run (key present). Configuration,
    not liveness — see ingest_snapshot()."""
    return bool(AISSTREAM_API_KEY)


def ingest_stats() -> dict:
    """Cumulative counters for /healthz?verbose=1 (JSON-safe)."""
    with _stats_lock:
        d = dict(_stats)
    d["last_msg_ts"] = _last_msg_iso()
    d["started_at"] = (None if d["started_epoch"] is None else
                       datetime.fromtimestamp(d["started_epoch"], tz=timezone.utc).isoformat())
    d["enabled"] = ingest_enabled()
    d["thread_alive"] = None if _client_thread is None else _client_thread.is_alive()
    with _live_lock:
        d["live_vessels"] = len(_live)
    d["queue_size"] = position_queue.qsize()
    return d


def ingest_snapshot(now: float | None = None, window_minutes: int | None = None) -> dict:
    """What the /healthz gate needs (see liveness.evaluate): configuration, thread
    liveness, last-message time and the trailing-window counts. `now` is injectable
    so tests can age the counters without sleeping."""
    now = time.time() if now is None else now
    w = liveness.stale_minutes() if window_minutes is None else int(window_minutes)
    with _stats_lock:
        last_msg = _stats["last_msg_epoch"]
        started = _stats["started_epoch"]
        received_total = _stats["received"]
        write_errors_total = _stats["write_errors"]
    return {
        "enabled": ingest_enabled(),
        # thread handle missing while enabled == never started (or died before we
        # looked): both are "absent", and absence is degraded, never "ok".
        "thread_alive": bool(_client_thread is not None and _client_thread.is_alive()),
        "flusher_alive": None,          # BunkerVision writes inline; no flusher thread
        "started_epoch": started,
        "last_msg_epoch": last_msg,
        "received_window": _window.total("received", w * 60, now=now),
        "write_errors_window": _window.total("write_errors", w * 60, now=now),
        "received_total": received_total,
        "write_errors_total": write_errors_total,
    }


def _reset_for_tests() -> None:
    """Zero every counter/handle (tests only — never called in the app)."""
    global _client_thread
    with _stats_lock:
        _stats.update({"received": 0, "write_errors": 0, "connects": 0,
                       "last_msg_epoch": None, "started_epoch": None})
        _last_write_warn["t"] = 0.0
    _window.reset()
    _client_thread = None


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
        _note_write_error("vessel_positions", e)

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
        _note_write_error("ais_positions", e)

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
        _note_write_error("vessel_cache", e)


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
        _note_write_error("vessel_cache(static)", e)

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
            _note_write_error("draught_history", e)


# ── WebSocket callbacks ────────────────────────────────────────────────────────

def _on_open(ws: websocket.WebSocketApp) -> None:
    with _stats_lock:
        _stats["connects"] += 1
    logger.info(f"aisstream.io connected (#{_stats['connects']}) — subscribing to "
                f"{len(ALL_BBOXES)} port bboxes")
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
    # aisstream error frames look like {"error": "..."} (bad key, throttled, second
    # client on the key) — surface them; they are exactly what explains a silent feed.
    if isinstance(msg, dict) and "error" in msg and "MessageType" not in msg:
        logger.warning(f"aisstream.io ERROR frame: {str(msg.get('error'))[:160]}")
        return
    try:
        mtype = msg.get("MessageType", "")
        if mtype:
            _note_message()      # feed liveness: any typed AIS frame counts as "received"
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

    with _stats_lock:
        _stats["started_epoch"] = time.time()
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
    global _client_thread
    t = threading.Thread(target=_run_client, daemon=True, name="ais-client")
    t.start()
    _client_thread = t
    return t


if __name__ == "__main__":
    import sys
    if not AISSTREAM_API_KEY:
        print("Set AISSTREAM_API_KEY in .env first")
        sys.exit(1)
    logger.info("Starting AIS client test — press Ctrl+C to stop")
    _run_client()
