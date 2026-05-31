"""
BunkerVision — Singapore Bunker Demand Tracker
Run:  python app.py
"""
from __future__ import annotations

import concurrent.futures
import json
import re
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests as _requests

from flask import Flask, Response, jsonify, render_template, request, stream_with_context

import db
import demand_model
import prices_data
from ais_client import get_live_positions, position_queue, start_ais_thread
from config import DEBUG, FLASK_SECRET, HOST, MAP_CENTRE, MAPBOX_TOKEN, PORT, PORTS
from event_detector import start_detector_thread
from mpa_scraper import load_all_port_stats, seed_vessels_from_json

app = Flask(__name__)
app.secret_key = FLASK_SECRET


# ── Routes ─────────────────────────────────────────────────────────────────────

def _get_port():
    """Return validated port key from ?port= query param (default: singapore)."""
    port = request.args.get("port", "singapore")
    return port if port in PORTS else "singapore"


@app.route("/")
def dashboard():
    port     = _get_port()
    port_cfg = PORTS[port]
    return render_template(
        "dashboard.html",
        map_centre=port_cfg["centre"],
        mapbox_token=MAPBOX_TOKEN,
        ports=PORTS,
        current_port=port,
        port_cfg=port_cfg,
    )


@app.route("/events")
def events_page():
    page = int(request.args.get("page", 1))
    per_page = 50
    offset = (page - 1) * per_page
    rows = db.query("""
        SELECT id, bunker_name, bunker_imo, recipient_name, recipient_mmsi,
               start_ts, end_ts, duration_min, estimated_mt,
               lat, lon, open
        FROM bunkering_events
        ORDER BY start_ts DESC
        LIMIT ? OFFSET ?
    """, [per_page, offset])
    total = db.query("SELECT COUNT(*) FROM bunkering_events")[0][0]
    events = [
        {
            "id": r[0], "bunker_name": r[1], "bunker_imo": r[2],
            "recipient_name": r[3], "recipient_mmsi": r[4],
            "start_ts": str(r[5]), "end_ts": str(r[6]) if r[6] else None,
            "duration_min": r[7], "estimated_mt": r[8],
            "lat": r[9], "lon": r[10], "open": r[11],
        }
        for r in rows
    ]
    return render_template("events.html", events=events, page=page,
                           per_page=per_page, total=total)


@app.route("/vessels")
def vessels_page():
    port = _get_port()
    rows = db.query("""
        SELECT bv.imo, bv.mmsi, bv.name, bv.port, bv.licensee, bv.capacity_mt,
               bv.last_updated,
               COUNT(be.id) AS event_count
        FROM bunker_vessels bv
        LEFT JOIN bunkering_events be ON be.bunker_imo = bv.imo
        GROUP BY bv.imo, bv.mmsi, bv.name, bv.port, bv.licensee, bv.capacity_mt, bv.last_updated
        ORDER BY event_count DESC, bv.name
    """)
    vessels = [
        {"imo": r[0], "mmsi": r[1], "name": r[2], "port": r[3] or "singapore",
         "licensee": r[4], "capacity_mt": r[5], "last_updated": str(r[6]), "event_count": r[7]}
        for r in rows
    ]
    port_counts: dict[str, int] = {}
    for v in vessels:
        p = v["port"]
        port_counts[p] = port_counts.get(p, 0) + 1

    return render_template(
        "vessels.html", vessels=vessels, ports=PORTS,
        current_port=port, port_counts=port_counts,
    )


# ── API ────────────────────────────────────────────────────────────────────────

@app.route("/api/positions")
def api_positions():
    port     = _get_port()
    bbox     = PORTS[port]["bbox"]
    all_pos  = get_live_positions()

    # Filter to selected port's bounding box
    positions = [
        p for p in all_pos
        if (p.get("lat") is not None and p.get("lon") is not None
            and bbox[0][0] <= p["lat"] <= bbox[1][0]
            and bbox[0][1] <= p["lon"] <= bbox[1][1])
    ]

    bunker_rows = db.query("SELECT imo, mmsi FROM bunker_vessels")
    bunker_iset = {r[0] for r in bunker_rows if r[0]}
    bunker_mset = {r[1] for r in bunker_rows if r[1]}

    open_events = db.query(
        "SELECT bunker_mmsi, recipient_mmsi FROM bunkering_events WHERE open = TRUE AND port = ?",
        [port],
    )
    alongside_mmsis: set[str] = set()
    for bm, rm in open_events:
        alongside_mmsis.add(bm)
        alongside_mmsis.add(rm)

    result = []
    for p in positions:
        is_bunker = p.get("imo") in bunker_iset or p.get("mmsi") in bunker_mset
        if p["mmsi"] in alongside_mmsis:
            state = "alongside"
        elif is_bunker and p.get("sog", 1) < 0.5:
            state = "stopped"
        elif is_bunker:
            state = "underway"
        else:
            state = "other"
        p["is_bunker"] = is_bunker
        p["state"] = state
        result.append(p)

    return jsonify(result)


@app.route("/api/events")
def api_events():
    port  = _get_port()
    limit = int(request.args.get("limit", 20))
    rows = db.query("""
        SELECT id, bunker_name, bunker_imo, recipient_name, recipient_mmsi,
               start_ts, end_ts, duration_min, estimated_mt, lat, lon, open,
               draught_start, draught_end, draught_delta_m, draught_vol_est
        FROM bunkering_events
        WHERE port = ?
        ORDER BY start_ts DESC LIMIT ?
    """, [port, limit])
    return jsonify([
        {
            "id": r[0], "bunker_name": r[1], "bunker_imo": r[2],
            "recipient_name": r[3], "recipient_mmsi": r[4],
            "start_ts": str(r[5]), "end_ts": str(r[6]) if r[6] else None,
            "duration_min": r[7], "estimated_mt": r[8],
            "lat": r[9], "lon": r[10], "open": r[11],
            "draught_start": r[12], "draught_end": r[13],
            "draught_delta_m": r[14], "draught_vol_est": r[15],
        }
        for r in rows
    ])


@app.route("/api/demand")
def api_demand():
    return jsonify(demand_model.running_demand_estimate(port=_get_port()))


@app.route("/api/demand_chart")
def api_demand_chart():
    return jsonify(demand_model.demand_chart_data(port=_get_port()))


@app.route("/api/fuel_split_chart")
def api_fuel_split_chart():
    return jsonify(demand_model.fuel_split_chart_data(port=_get_port()))


@app.route("/api/snapshot_estimate", methods=["POST"])
def api_snapshot_estimate():
    """Manually trigger a month-end estimate snapshot."""
    port  = _get_port()
    year  = int(request.json.get("year",  date.today().year))
    month = int(request.json.get("month", date.today().month))
    saved = demand_model.save_month_estimate(port, year, month)
    return jsonify({"saved": saved, "port": port, "year": year, "month": month})


@app.route("/api/debug")
def api_debug():
    import threading
    from ais_client import _live, position_queue
    from config import AISSTREAM_API_KEY
    threads = {t.name: t.is_alive() for t in threading.enumerate()}
    return jsonify({
        "api_key_set": bool(AISSTREAM_API_KEY),
        "api_key_prefix": AISSTREAM_API_KEY[:6] + "…" if AISSTREAM_API_KEY else "",
        "live_vessel_count": len(_live),
        "queue_size": position_queue.qsize(),
        "threads": threads,
    })


@app.route("/api/stats")
def api_stats():
    port = _get_port()
    bbox = PORTS[port]["bbox"]

    vessel_count  = db.query("SELECT COUNT(*) FROM bunker_vessels WHERE port = ?", [port])[0][0]
    active_events = db.query(
        "SELECT COUNT(*) FROM bunkering_events WHERE open = TRUE AND port = ?", [port]
    )[0][0]
    total_events  = db.query(
        "SELECT COUNT(*) FROM bunkering_events WHERE port = ?", [port]
    )[0][0]

    # Live positions in this port's bbox
    all_pos    = get_live_positions()
    live_count = sum(
        1 for p in all_pos
        if (p.get("lat") is not None
            and bbox[0][0] <= p["lat"] <= bbox[1][0]
            and bbox[0][1] <= p.get("lon", 0) <= bbox[1][1])
    )

    return jsonify({
        "port": port,
        "vessel_count": vessel_count,
        "active_events": active_events,
        "total_events": total_events,
        "live_positions": live_count,
        "updated": datetime.now(timezone.utc).isoformat(),
    })


@app.route("/api/registry/candidates")
def api_registry_candidates():
    port = _get_port()
    bbox = PORTS[port]["bbox"]
    min_lat, min_lon = bbox[0]
    max_lat, max_lon = bbox[1]

    rows = db.query("""
        SELECT vp.mmsi, vp.imo, vp.name, vp.lat, vp.lon, vp.sog, vp.ts,
               vc.type_code, vc.draught
        FROM vessel_positions vp
        LEFT JOIN vessel_cache vc ON vc.mmsi = vp.mmsi
        WHERE vp.lat BETWEEN ? AND ?
          AND vp.lon BETWEEN ? AND ?
          AND vp.ts >= NOW() - INTERVAL '4 hours'
          AND vp.sog < 1.5
          AND NOT EXISTS (
              SELECT 1 FROM bunker_vessels bv
              WHERE bv.mmsi = vp.mmsi
                 OR (vp.imo != '' AND vp.imo IS NOT NULL AND bv.imo = vp.imo)
                 OR UPPER(bv.name) = UPPER(vp.name)
          )
          AND (
              -- Small tanker by AIS type in bunker barge draught range
              (vc.type_code >= 80 AND vc.type_code <= 89 AND (vc.draught IS NULL OR vc.draught <= 9))
              -- Unknown type but draught in barge range
              OR (vc.draught >= 3 AND vc.draught <= 9)
          )
        ORDER BY
            CASE WHEN vc.type_code >= 80 AND vc.type_code <= 89 THEN 0 ELSE 1 END,
            vp.sog ASC,
            vc.draught DESC NULLS LAST
        LIMIT 300
    """, [min_lat, max_lat, min_lon, max_lon])

    candidates = []
    for r in rows:
        type_code = r[7]
        draught   = r[8]
        if type_code and 80 <= type_code <= 89 and (draught is None or draught <= 9):
            confidence = "high"
        elif draught and draught >= 3:
            confidence = "medium"
        else:
            confidence = "low"
        candidates.append({
            "mmsi": r[0], "imo": r[1] or "", "name": r[2] or "",
            "lat": r[3], "lon": r[4], "sog": r[5],
            "last_seen": str(r[6]),
            "type_code": type_code,
            "draught": draught,
            "confidence": confidence,
        })
    return jsonify(candidates)


@app.route("/api/registry/add", methods=["POST"])
def api_registry_add():
    data = request.get_json() or {}
    mmsi = (data.get("mmsi") or "").strip()
    imo  = (data.get("imo")  or "").strip()
    name = (data.get("name") or "").strip()
    port = data.get("port", "singapore")
    licensee    = data.get("licensee") or f"AIS-{port.upper()}"
    capacity_mt = data.get("capacity_mt")

    if not (mmsi or imo or name):
        return jsonify({"error": "mmsi, imo, or name required"}), 400
    if port not in PORTS:
        return jsonify({"error": "invalid port"}), 400

    imo_key = imo if imo else (f"NAME:{name}" if name else f"MMSI:{mmsi}")
    today = datetime.now(timezone.utc).date()

    db.execute("""
        INSERT INTO bunker_vessels (imo, mmsi, name, port, licensee, capacity_mt, last_updated)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT (imo) DO UPDATE SET
            mmsi         = COALESCE(excluded.mmsi, bunker_vessels.mmsi),
            name         = COALESCE(excluded.name, bunker_vessels.name),
            port         = excluded.port,
            licensee     = excluded.licensee,
            capacity_mt  = COALESCE(excluded.capacity_mt, bunker_vessels.capacity_mt),
            last_updated = excluded.last_updated
    """, [imo_key, mmsi or None, name or None, port, licensee, capacity_mt, today])

    try:
        from ais_client import refresh_bunker_names
        refresh_bunker_names()
    except Exception:
        pass

    return jsonify({"added": imo_key, "port": port})


_VF_HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64; rv:126.0) Gecko/20100101 Firefox/126.0",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
}
_vf_last_request = 0.0
_VF_MIN_GAP = 4.0   # seconds between VF requests — stay well under rate limit


def _fetch_vf_type(mmsi: str) -> str:
    """Fetch ship type from VesselFinder for a given MMSI. Returns raw type string."""
    global _vf_last_request
    gap = time.monotonic() - _vf_last_request
    if gap < _VF_MIN_GAP:
        time.sleep(_VF_MIN_GAP - gap)
    _vf_last_request = time.monotonic()
    try:
        r = _requests.get(
            f"https://www.vesselfinder.com/vessels/details/{mmsi}",
            headers=_VF_HEADERS,
            timeout=12,
        )
        if r.status_code == 429 or r.status_code == 403:
            return "Blocked"
        m = re.search(r'<h2 class="vst">([^,<]+)', r.text)
        if m:
            return m.group(1).strip()
        m = re.search(r'is an? ([A-Za-z/() ]+?) built in', r.text)
        if m:
            return m.group(1).strip()
        return "Unknown"
    except Exception:
        return "Error"


def _classify_vf_type(vtype: str) -> dict:
    """Map a VesselFinder ship type string to a confidence level and label."""
    vl = vtype.lower()
    if "bunker" in vl:
        return {"type": vtype, "label": "Bunker vessel", "level": "confirmed"}
    if vtype == "Tanker":
        # Plain "Tanker" with no further qualifier = inland waterway tanker
        return {"type": vtype, "label": "Inland tanker", "level": "likely"}
    if "oil products" in vl or ("tanker" in vl and "oil" in vl and "chemical" not in vl):
        return {"type": vtype, "label": "Oil products tanker", "level": "possible"}
    if "chemical" in vl or "product" in vl:
        return {"type": vtype, "label": vtype, "level": "cargo"}
    if "tanker" in vl or "petroleum" in vl:
        return {"type": vtype, "label": vtype, "level": "possible"}
    if vtype in ("Unknown", "Error", "Not found"):
        return {"type": vtype, "label": "—", "level": "unknown"}
    return {"type": vtype, "label": vtype, "level": "not_tanker"}


@app.route("/api/registry/vf_check", methods=["POST"])
def api_registry_vf_check():
    """Batch-verify candidate MMSIs against VesselFinder. Sequential with polite rate-limiting."""
    data  = request.get_json() or {}
    mmsis = [str(m) for m in data.get("mmsis", [])][:30]  # cap at 30 per call

    results = {}
    for mmsi in mmsis:
        vtype = _fetch_vf_type(mmsi)
        results[mmsi] = _classify_vf_type(vtype)

    return jsonify(results)


@app.route("/api/registry/eq_check", methods=["POST"])
def api_registry_eq_check():
    """Batch-verify candidate IMOs against Equasis. Sequential, rate-limited."""
    from equasis import lookup_imo, classify_ship_type
    data = request.get_json() or {}
    imos = [str(i) for i in data.get("imos", []) if str(i).isdigit()][:50]

    results = {}
    for imo in imos:
        info = lookup_imo(imo)
        if info:
            ship_type = info.get("ship_type", "")
            results[imo] = {**classify_ship_type(ship_type), "ship_type": ship_type,
                            "flag": info.get("flag", ""), "gt": info.get("gt")}
        else:
            results[imo] = {"level": "unknown", "label": "—", "ship_type": ""}

    return jsonify(results)


@app.route("/api/registry/eq_search")
def api_registry_eq_search():
    """
    Search Equasis by vessel name term.
    Returns vessels with Equasis ship type — useful for finding 'Bunkering Tanker' vessels.
    Query: ?q=bunker&filter=bunkering (filter: 'bunkering' shows only Bunkering Tanker type)
    """
    from equasis import search as eq_search, classify_ship_type
    q       = request.args.get("q", "").strip()
    filt    = request.args.get("filter", "")  # 'bunkering' to only return bunkering tankers

    if not q or len(q) < 3:
        return jsonify({"error": "q must be at least 3 chars"}), 400

    rows = eq_search(q)
    if filt == "bunkering":
        rows = [r for r in rows if "bunker" in r.get("ship_type", "").lower()]

    out = []
    for r in rows:
        ship_type = r.get("ship_type", "")
        cl = classify_ship_type(ship_type)
        # Check if this vessel is already in our registry
        imo = r["imo"]
        in_registry = bool(db.query(
            "SELECT 1 FROM bunker_vessels WHERE imo = ? OR mmsi IN (SELECT mmsi FROM vessel_cache WHERE imo = ?) LIMIT 1",
            [imo, imo]
        ))
        # Check if visible in any port AIS area
        in_ais = bool(db.query(
            "SELECT 1 FROM vessel_cache WHERE imo = ? AND last_seen >= ? LIMIT 1",
            [imo, datetime.now(timezone.utc) - timedelta(days=7)]
        ))
        out.append({
            **r,
            "confidence": cl["level"],
            "type_label": cl["label"],
            "in_registry": in_registry,
            "in_ais": in_ais,
        })
    return jsonify(out)


@app.route("/api/registry/eq_tanker_scan", methods=["POST"])
def api_registry_eq_tanker_scan():
    """
    AIS tanker scan → Equasis type lookup.
    Finds all tankers (type_code 80-89) seen in a port bbox in the last N days,
    looks each up on Equasis by IMO, and auto-adds confirmed 'Bunkering Tanker' vessels.
    Returns all results for review.
    """
    from equasis import lookup_imo, classify_ship_type
    data  = request.get_json() or {}
    port  = data.get("port", "")
    days  = min(int(data.get("days", 7)), 30)

    if port not in PORTS:
        return jsonify({"error": "invalid port"}), 400

    cfg = PORTS[port]
    bb  = cfg["bbox"]
    min_lat, min_lon = bb[0]
    max_lat, max_lon = bb[1]

    # Compute cutoff in Python to avoid DuckDB NOW()/pytz issues
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)

    # Tankers in bbox, not already in registry, with an IMO
    rows = db.query("""
        SELECT vp.mmsi, vc.imo, COALESCE(vc.name, vp.name) AS name,
               vc.type_code, vp.lat, vp.lon, vp.sog
        FROM vessel_positions vp
        JOIN vessel_cache vc ON vc.mmsi = vp.mmsi
        WHERE vc.type_code BETWEEN 80 AND 89
          AND vp.lat BETWEEN ? AND ?
          AND vp.lon BETWEEN ? AND ?
          AND vp.ts >= ?
          AND vc.imo IS NOT NULL AND vc.imo != ''
          AND NOT EXISTS (
              SELECT 1 FROM bunker_vessels bv
              WHERE bv.mmsi = vp.mmsi OR bv.imo = vc.imo
          )
        ORDER BY vp.sog ASC
        LIMIT 10
    """, [min_lat, max_lat, min_lon, max_lon, cutoff])

    today   = datetime.now(timezone.utc).date()
    results = []
    n_added = 0

    for (mmsi, imo, name, type_code, lat, lon, sog) in rows:
        info = lookup_imo(str(imo))
        if not info:
            results.append({
                "mmsi": mmsi, "imo": imo, "name": name or mmsi,
                "ship_type": "", "flag": "", "gt": None,
                "confidence": "unknown", "type_label": "—",
                "sog": sog, "auto_added": False,
            })
            continue

        ship_type = info.get("ship_type", "")
        cl        = classify_ship_type(ship_type)
        disp_name = info.get("name") or name or mmsi
        auto_added = False

        if cl["level"] == "confirmed":
            db.execute("""
                INSERT INTO bunker_vessels (imo, mmsi, name, port, licensee, last_updated)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT (imo) DO UPDATE SET
                    mmsi         = excluded.mmsi,
                    name         = COALESCE(excluded.name, bunker_vessels.name),
                    port         = excluded.port,
                    last_updated = excluded.last_updated
            """, [str(imo), mmsi, disp_name, port, "Equasis-AUTO", today])
            n_added   += 1
            auto_added = True

        results.append({
            "mmsi": mmsi, "imo": imo, "name": disp_name,
            "ship_type": ship_type, "flag": info.get("flag", ""),
            "gt": info.get("gt"), "confidence": cl["level"],
            "type_label": cl["label"], "sog": sog,
            "auto_added": auto_added,
        })

    if n_added:
        try:
            from ais_client import refresh_bunker_names
            refresh_bunker_names()
        except Exception:
            pass

    return jsonify({"scanned": len(rows), "auto_added": n_added, "results": results})


_OPERATORS_FILE = Path(__file__).parent / "data" / "operators.json"


@app.route("/api/registry/operators", methods=["GET"])
def api_registry_operators_get():
    try:
        data = json.loads(_OPERATORS_FILE.read_text())
        return jsonify({"operators": data.get("operators", [])})
    except Exception:
        return jsonify({"operators": []})


@app.route("/api/registry/operators", methods=["POST"])
def api_registry_operators_save():
    data = request.get_json() or {}
    ops = [str(o).strip() for o in data.get("operators", []) if str(o).strip()]
    _OPERATORS_FILE.write_text(json.dumps({"operators": ops}, indent=2))
    return jsonify({"ok": True, "operators": ops})


@app.route("/api/registry/operators/scan", methods=["POST"])
def api_registry_operators_scan():
    """
    Search Equasis for a list of operator names, auto-assign confirmed Bunkering
    Tankers to whichever port bboxes they've been seen in via AIS.
    """
    from equasis import search as eq_search, classify_ship_type

    data  = request.get_json() or {}
    terms = data.get("terms", [])
    if not terms:
        return jsonify({"error": "no operators supplied"}), 400

    today   = datetime.now(timezone.utc).date()
    results = []
    n_added = 0

    def _ports_for_imo(imo: str) -> list:
        positions = db.query(
            "SELECT lat, lon FROM vessel_positions vp "
            "JOIN vessel_cache vc ON vc.mmsi = vp.mmsi WHERE vc.imo = ? LIMIT 500",
            [str(imo)]
        )
        matched = []
        for port_key, cfg in PORTS.items():
            bb = cfg["bbox"]
            for lat, lon in positions:
                if bb[0][0] <= lat <= bb[1][0] and bb[0][1] <= lon <= bb[1][1]:
                    matched.append(port_key)
                    break
        return matched

    for term in terms:
        hits = eq_search(term)
        for r in hits:
            imo       = r.get("imo", "")
            name      = r.get("name", "")
            ship_type = r.get("ship_type", "")
            cl        = classify_ship_type(ship_type)

            if not imo or cl["level"] != "confirmed":
                continue

            already = db.query(
                "SELECT 1 FROM bunker_vessels WHERE imo = ? LIMIT 1", [str(imo)]
            )
            if already:
                continue

            ports_seen = _ports_for_imo(imo)
            if not ports_seen:
                results.append({"operator": term, "name": name, "imo": imo,
                                 "ship_type": ship_type, "ports": [],
                                 "added": False, "note": "no AIS sightings in any port bbox"})
                continue

            for port_key in ports_seen:
                db.execute("""
                    INSERT INTO bunker_vessels (imo, mmsi, name, port, licensee, last_updated)
                    VALUES (?, NULL, ?, ?, ?, ?)
                    ON CONFLICT (imo) DO UPDATE SET
                        name         = COALESCE(excluded.name, bunker_vessels.name),
                        port         = excluded.port,
                        last_updated = excluded.last_updated
                """, [str(imo), name, port_key, "Equasis-AUTO", today])

            n_added += 1
            results.append({"operator": term, "name": name, "imo": imo,
                             "ship_type": ship_type, "ports": ports_seen, "added": True})

    if n_added:
        try:
            from ais_client import refresh_bunker_names
            refresh_bunker_names()
        except Exception:
            pass

    return jsonify({"searched": len(terms), "added": n_added, "results": results})


@app.route("/api/registry/flag_scan", methods=["POST"])
def api_registry_flag_scan():
    """
    Scan all Equasis pages for a flag state and collect Bunkering Tanker vessels.
    Body: {flag: "Panama" or "0485", max_pages: 50, dry_run: false}
    """
    from equasis import find_bunkering_tankers_by_flag, classify_ship_type, FLAG_CODES

    data      = request.get_json() or {}
    flag_input = data.get("flag", "").strip()
    max_pages = min(int(data.get("max_pages", 50)), 200)
    dry_run   = bool(data.get("dry_run", False))

    if not flag_input:
        return jsonify({"error": "flag required — use country name (e.g. 'Panama') or Equasis code (e.g. '0485')"}), 400

    # Accept country name or numeric code
    flag_code = FLAG_CODES.get(flag_input, flag_input)

    vessels = find_bunkering_tankers_by_flag(flag_code, max_pages=max_pages)

    def _ports_for_imo(imo: str) -> list:
        positions = db.query(
            "SELECT lat, lon FROM vessel_positions vp "
            "JOIN vessel_cache vc ON vc.mmsi = vp.mmsi WHERE vc.imo = ? LIMIT 500",
            [str(imo)]
        )
        matched = []
        for port_key, cfg in PORTS.items():
            bb = cfg["bbox"]
            for lat, lon in positions:
                if bb[0][0] <= lat <= bb[1][0] and bb[0][1] <= lon <= bb[1][1]:
                    matched.append(port_key)
                    break
        return matched

    today   = datetime.now(timezone.utc).date()
    results = []
    n_added = 0

    for v in vessels:
        imo  = v.get("imo", "")
        name = v.get("name", "")
        if not imo:
            continue

        already = db.query("SELECT 1 FROM bunker_vessels WHERE imo = ? LIMIT 1", [str(imo)])
        if already:
            results.append({**v, "added": False, "ports": [], "note": "already in registry"})
            continue

        ports_seen = _ports_for_imo(imo)
        if not ports_seen:
            results.append({**v, "added": False, "ports": [], "note": "no AIS sightings in any port bbox"})
            continue

        if not dry_run:
            for port_key in ports_seen:
                db.execute("""
                    INSERT INTO bunker_vessels (imo, mmsi, name, port, licensee, last_updated)
                    VALUES (?, NULL, ?, ?, ?, ?)
                    ON CONFLICT (imo) DO UPDATE SET
                        name         = COALESCE(excluded.name, bunker_vessels.name),
                        port         = excluded.port,
                        last_updated = excluded.last_updated
                """, [str(imo), name, ports_seen[0], "Equasis-Flag", today])
            n_added += 1

        results.append({**v, "added": not dry_run and bool(ports_seen), "ports": ports_seen})

    if n_added:
        try:
            from ais_client import refresh_bunker_names
            refresh_bunker_names()
        except Exception:
            pass

    return jsonify({
        "flag": flag_code,
        "bunkering_tankers_found": len(vessels),
        "added": n_added,
        "dry_run": dry_run,
        "results": results,
    })


_global_scan_state: dict = {"status": "idle", "found": 0, "page": 0, "log": []}
_global_scan_lock = __import__("threading").Lock()


@app.route("/api/registry/global_bunkering_scan", methods=["GET"])
def api_global_bunkering_scan_status():
    return jsonify(_global_scan_state)


@app.route("/api/registry/global_bunkering_scan", methods=["POST"])
def api_global_bunkering_scan():
    """
    Scan Equasis 'Other Tankers' category globally to find all Bunkering Tankers worldwide.
    Runs in background. No body required; optional {dry_run: true, cat_codes: ["8","6"]}.
    GET the same endpoint to check progress.
    """
    import threading
    from equasis import find_all_bunkering_tankers_global

    data      = request.get_json() or {}
    dry_run   = bool(data.get("dry_run", False))
    cat_codes = data.get("cat_codes", ["8"])

    with _global_scan_lock:
        if _global_scan_state.get("status") == "running":
            return jsonify({"error": "scan already running", "state": _global_scan_state}), 409
        _global_scan_state.update({"status": "running", "found": 0, "page": 0, "log": [], "dry_run": dry_run})

    def _run():
        today = datetime.now(timezone.utc).date()
        n_added = 0

        def _progress(page, results, total_found):
            with _global_scan_lock:
                _global_scan_state["page"]  = page
                _global_scan_state["found"] = total_found

        try:
            vessels = find_all_bunkering_tankers_global(cat_codes=cat_codes, progress_cb=_progress)

            for v in vessels:
                imo  = v.get("imo", "")
                name = v.get("name", "")
                if not imo:
                    continue

                # Always store in the discovery table regardless of AIS sightings
                db.execute("""
                    INSERT INTO eq_bunkering_found
                        (imo, name, ship_type, flag, gt, year, found_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT (imo) DO UPDATE SET
                        name      = COALESCE(excluded.name, eq_bunkering_found.name),
                        ship_type = excluded.ship_type,
                        flag      = excluded.flag,
                        gt        = COALESCE(excluded.gt, eq_bunkering_found.gt),
                        found_at  = excluded.found_at
                """, [imo, name, v.get("ship_type",""), v.get("flag",""),
                      v.get("gt"), v.get("year",""), today.isoformat()])

                if dry_run:
                    continue

                # Auto-add to registry if AIS sightings exist in any port bbox
                already = db.query("SELECT 1 FROM bunker_vessels WHERE imo=? LIMIT 1", [imo])
                if already:
                    continue

                positions = db.query(
                    "SELECT lat, lon FROM vessel_positions vp "
                    "JOIN vessel_cache vc ON vc.mmsi=vp.mmsi WHERE vc.imo=? LIMIT 500", [imo]
                )
                for port_key, cfg in PORTS.items():
                    bb = cfg["bbox"]
                    for lat, lon in positions:
                        if bb[0][0] <= lat <= bb[1][0] and bb[0][1] <= lon <= bb[1][1]:
                            db.execute("""
                                INSERT INTO bunker_vessels (imo, mmsi, name, port, licensee, last_updated)
                                VALUES (?, NULL, ?, ?, ?, ?)
                                ON CONFLICT (imo) DO UPDATE SET
                                    name=COALESCE(excluded.name,bunker_vessels.name),
                                    port=excluded.port, last_updated=excluded.last_updated
                            """, [imo, name, port_key, "Equasis-Global", today])
                            n_added += 1
                            break

            with _global_scan_lock:
                _global_scan_state.update({
                    "status": "done",
                    "found": len(vessels),
                    "added_to_registry": n_added,
                    "completed_at": datetime.now(timezone.utc).isoformat(),
                })
        except Exception as exc:
            logger.error(f"Global bunkering scan failed: {exc}")
            with _global_scan_lock:
                _global_scan_state.update({"status": "error", "error": str(exc)})

    threading.Thread(target=_run, daemon=True).start()
    return jsonify({"status": "started", "dry_run": dry_run, "cat_codes": cat_codes})


@app.route("/api/registry/eq_bunkering_found", methods=["GET"])
def api_eq_bunkering_found():
    """Return all vessels discovered by the global Equasis scan."""
    rows = db.query("""
        SELECT imo, name, ship_type, flag, gt, year, added_to_registry, found_at
        FROM eq_bunkering_found ORDER BY flag, name
    """)
    return jsonify([
        {"imo": r[0], "name": r[1], "ship_type": r[2], "flag": r[3],
         "gt": r[4], "year": r[5], "added_to_registry": bool(r[6]), "found_at": str(r[7])}
        for r in rows
    ])


@app.route("/api/registry/apply_eq_found", methods=["POST"])
def api_apply_eq_found():
    """
    Process vessels already in eq_bunkering_found against AIS position history.
    Adds any with port bbox sightings to bunker_vessels registry.
    No Equasis requests needed — works from stored discovery data.
    """
    today   = datetime.now(timezone.utc).date()
    vessels = db.query("SELECT imo, name FROM eq_bunkering_found")
    n_checked = 0
    n_added   = 0
    n_already = 0

    for imo, name in vessels:
        n_checked += 1
        already = db.query("SELECT 1 FROM bunker_vessels WHERE imo=? LIMIT 1", [imo])
        if already:
            n_already += 1
            continue

        positions = db.query(
            "SELECT lat, lon FROM vessel_positions vp "
            "JOIN vessel_cache vc ON vc.mmsi=vp.mmsi WHERE vc.imo=? LIMIT 500", [imo]
        )
        for port_key, cfg in PORTS.items():
            bb = cfg["bbox"]
            for lat, lon in positions:
                if bb[0][0] <= lat <= bb[1][0] and bb[0][1] <= lon <= bb[1][1]:
                    db.execute("""
                        INSERT INTO bunker_vessels (imo, mmsi, name, port, licensee, last_updated)
                        VALUES (?, NULL, ?, ?, ?, ?)
                        ON CONFLICT (imo) DO UPDATE SET
                            name=COALESCE(excluded.name, bunker_vessels.name),
                            port=excluded.port, last_updated=excluded.last_updated
                    """, [imo, name, port_key, "Equasis-Global", today])
                    db.execute(
                        "UPDATE eq_bunkering_found SET added_to_registry=TRUE WHERE imo=?", [imo]
                    )
                    n_added += 1
                    break

    if n_added:
        try:
            from ais_client import refresh_bunker_names
            refresh_bunker_names()
        except Exception:
            pass

    return jsonify({
        "checked": n_checked,
        "already_in_registry": n_already,
        "added": n_added,
        "no_ais_sightings": n_checked - n_already - n_added,
    })


@app.route("/api/registry/auto_detect", methods=["POST"])
def api_registry_auto_detect():
    """
    Auto-populate registry with high-confidence bunker vessel candidates.
    Uses port-specific heuristics that don't require VF/Equasis.
    """
    data = request.get_json() or {}
    port = data.get("port", "")
    if port not in PORTS:
        return jsonify({"error": "invalid port"}), 400

    bbox = PORTS[port]["bbox"]
    min_lat, min_lon = bbox[0]
    max_lat, max_lon = bbox[1]
    today = datetime.now(timezone.utc).date()
    added = []

    # ── Dutch inland tankers (Rotterdam, Antwerp) ─────────────────────────────
    # MMSI prefix 244-246 = Netherlands inland waterway. Type 80-89 + no IMO =
    # inland oil barge. In Rotterdam/Antwerp these are virtually all bunker barges.
    if port in ("rotterdam", "antwerp"):
        rows = db.query("""
            SELECT vp.mmsi, vp.name, vp.lat, vp.lon,
                   vc.type_code, vc.draught, vc.imo
            FROM vessel_positions vp
            LEFT JOIN vessel_cache vc ON vc.mmsi = vp.mmsi
            WHERE vp.lat BETWEEN ? AND ?
              AND vp.lon BETWEEN ? AND ?
              AND vp.ts >= NOW() - INTERVAL '24 hours'
              AND vp.sog < 1.5
              AND (vp.mmsi LIKE '244%' OR vp.mmsi LIKE '245%' OR vp.mmsi LIKE '246%')
              AND vc.type_code >= 80 AND vc.type_code <= 89
              AND (vc.draught IS NULL OR vc.draught BETWEEN 1.5 AND 7)
              AND NOT EXISTS (
                  SELECT 1 FROM bunker_vessels bv WHERE bv.mmsi = vp.mmsi
              )
        """, [min_lat, max_lat, min_lon, max_lon])

        for mmsi, name, lat, lon, type_code, draught, imo in rows:
            imo_key = imo if imo else f"MMSI:{mmsi}"
            db.execute("""
                INSERT INTO bunker_vessels (imo, mmsi, name, port, licensee, last_updated)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT (imo) DO UPDATE SET
                    mmsi         = excluded.mmsi,
                    name         = COALESCE(excluded.name, bunker_vessels.name),
                    port         = excluded.port,
                    last_updated = excluded.last_updated
            """, [imo_key, mmsi, name or "", port, "AIS-AUTO", today])
            added.append({"mmsi": mmsi, "name": name or mmsi, "reason": "Dutch inland tanker"})

    try:
        from ais_client import refresh_bunker_names
        refresh_bunker_names()
    except Exception:
        pass

    return jsonify({"added": len(added), "vessels": added})


@app.route("/api/registry/remove", methods=["POST"])
def api_registry_remove():
    data = request.get_json() or {}
    imo = (data.get("imo") or "").strip()
    if not imo:
        return jsonify({"error": "imo required"}), 400

    db.execute("DELETE FROM bunker_vessels WHERE imo = ?", [imo])

    try:
        from ais_client import refresh_bunker_names
        refresh_bunker_names()
    except Exception:
        pass

    return jsonify({"removed": imo})


@app.route("/api/refresh_registry", methods=["POST"])
def api_refresh_registry():
    n = seed_vessels_from_json()
    return jsonify({"upserted": n})


@app.route("/api/refresh_port_stats", methods=["POST"])
def api_refresh_port_stats():
    from mpa_scraper import load_all_port_stats
    load_all_port_stats()
    return jsonify({"status": "ok"})


# ── Pricing ────────────────────────────────────────────────────────────────────

@app.route("/prices")
def prices_page():
    hub   = request.args.get("hub",   "singapore")
    grade = request.args.get("grade", "VLSFO")
    days  = int(request.args.get("days", 90))
    if hub   not in prices_data.HUBS:      hub   = "singapore"
    if grade not in prices_data.GRADES:    grade = "VLSFO"
    if days  not in prices_data.LOOKBACKS: days  = 90

    all_ports  = prices_data.get_all_ports_latest()
    latest_all = prices_data.get_all_latest()

    # Latest sales volumes from DuckDB (port key → {fuel_type → volume_mt})
    sales_summary: dict[str, dict] = {}
    try:
        rows = db.query("""
            SELECT port, fuel_type, volume_mt, ships_count, month
            FROM port_bunker_sales
            WHERE (port, month) IN (
                SELECT port, MAX(month) FROM port_bunker_sales GROUP BY port
            )
            ORDER BY port, fuel_type
        """)
        for r in rows:
            sales_summary.setdefault(r[0].lower().replace(" ", "_"), {})[r[1]] = {
                "volume_mt": r[2], "ships_count": r[3], "month": str(r[4])
            }
    except Exception:
        pass

    return render_template(
        "prices.html",
        hub=hub,
        grade=grade,
        days=days,
        grades=prices_data.GRADES,
        lookbacks=prices_data.LOOKBACKS,
        latest_all=latest_all,
        all_ports=all_ports,
        sales_summary=sales_summary,
        current_port="singapore",   # navbar port selector — prices hub is separate
        ports=PORTS,
    )


@app.route("/api/prices/chart")
def api_prices_chart():
    hub   = request.args.get("hub",   "singapore")
    grade = request.args.get("grade", "VLSFO")
    days  = int(request.args.get("days", 90))
    if hub   not in prices_data.HUBS:      hub   = "singapore"
    if grade not in prices_data.GRADES:    grade = "VLSFO"
    if days  not in prices_data.LOOKBACKS: days  = 90
    fig = prices_data.build_plotly_figure(hub, grade, days)
    return jsonify(fig)


@app.route("/api/prices/latest")
def api_prices_latest():
    hub   = request.args.get("hub",   "singapore")
    grade = request.args.get("grade", "VLSFO")
    if hub   not in prices_data.HUBS:   hub   = "singapore"
    if grade not in prices_data.GRADES: grade = "VLSFO"
    return jsonify(prices_data.get_latest(hub, grade))


# ── SSE stream ─────────────────────────────────────────────────────────────────

@app.route("/stream")
def stream():
    def generate():
        while True:
            try:
                item = position_queue.get(timeout=25)
                yield f"data: {json.dumps(item)}\n\n"
            except Exception:
                # Heartbeat to keep connection alive
                yield "data: {\"heartbeat\":true}\n\n"
    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


# ── Startup ────────────────────────────────────────────────────────────────────

def _initial_load():
    """Seed vessels and load all port stats at startup (runs in background thread)."""
    time.sleep(2)
    from loguru import logger
    try:
        seed_vessels_from_json()
    except Exception as e:
        logger.error(f"Vessel seed failed: {e}")
    try:
        load_all_port_stats()
    except Exception as e:
        logger.warning(f"Port stats load failed (non-fatal): {e}")


if __name__ == "__main__":
    # Initialise DB schema
    db.get_conn()

    # Background threads
    threading.Thread(target=_initial_load, daemon=True).start()
    start_ais_thread()
    start_detector_thread()

    # Scheduled weekly registry refresh
    from apscheduler.schedulers.background import BackgroundScheduler
    sched = BackgroundScheduler()
    sched.add_job(seed_vessels_from_json, "interval", weeks=1)
    sched.add_job(load_all_port_stats,    "interval", days=1)
    # Snapshot previous month's estimate on the 1st of each month at 06:00 UTC
    sched.add_job(
        lambda: [demand_model.snapshot_previous_month_if_missing(p) for p in PORTS],
        "cron", day=1, hour=6, minute=0,
    )
    sched.start()

    # Also run on startup in case the server was down over a month boundary
    for _p in PORTS:
        try:
            demand_model.snapshot_previous_month_if_missing(_p)
        except Exception:
            pass

    app.run(host=HOST, port=PORT, debug=DEBUG, use_reloader=False)
