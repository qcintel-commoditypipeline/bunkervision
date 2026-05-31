"""
prices_data.py — Multi-source bunker & cargo price queries.

Reads from the SQLite DB at /root/bunkervision/bunker_prices.db,
which is written by three independent scrapers:
  - integr8_scraper.py   → bunker_prices table, source='integr8'
  - sb_scraper.py        → bunker_prices table, source='shipandbunker'
  - quantum_scraper.py   → quantum_prices table (QCIntel assessments)
"""
from __future__ import annotations

import sqlite3
import threading
from datetime import date, timedelta
from pathlib import Path

PRICE_DB = Path("/root/bunkervision/bunker_prices.db")

_lock = threading.Lock()


def _conn() -> sqlite3.Connection:
    c = sqlite3.connect(PRICE_DB, check_same_thread=False)
    c.row_factory = sqlite3.Row
    return c


# ── Port / region catalogues ──────────────────────────────────────────────────

_SB_PORTS = [
    # Asia-Pacific
    "Singapore", "Hong Kong", "Zhoushan", "Kaohsiung", "Busan", "Osaka",
    "Manila", "Ho Chi Minh City", "Haiphong", "Colombo", "Kochi", "Chennai",
    "Karachi", "Sydney", "Melbourne", "Brisbane", "Fremantle", "Auckland",
    # Middle East / E. Africa
    "Fujairah", "Jeddah", "Djibouti",
    # Europe
    "Rotterdam", "Antwerp", "Hamburg", "Gothenburg", "Skaw", "Bergen", "Gdansk",
    "Tallinn", "Riga", "Gibraltar", "Las Palmas", "Lisbon", "Malta Offshore",
    "Limassol", "Piraeus", "Istanbul", "Augusta",
    # Americas
    "New York", "Houston", "Los Angeles", "Seattle", "Vancouver", "Balboa",
    "Cartagena", "Guayaquil", "Port of Spain", "Santos", "Rio de Janeiro",
    "Buenos Aires", "Montevideo",
    # Africa
    "Durban", "Cape Town", "Tema",
]

_INTEGR8_PORTS = [
    "Singapore", "Rotterdam", "Fujairah", "Gibraltar", "Hong Kong", "Houston",
    "Las Palmas", "Lisbon", "New York", "Malta Offshore", "Skaw", "Zhoushan",
    "Balboa", "Durban", "Los Angeles", "Port Suez",
]

PORT_REGIONS: dict[str, str] = {
    **{p: "Asia-Pacific" for p in [
        "Singapore", "Hong Kong", "Zhoushan", "Kaohsiung", "Busan", "Osaka",
        "Manila", "Ho Chi Minh City", "Haiphong", "Colombo", "Kochi", "Chennai",
        "Karachi", "Sydney", "Melbourne", "Brisbane", "Fremantle", "Auckland",
    ]},
    **{p: "Middle East" for p in ["Fujairah", "Jeddah", "Djibouti", "Port Suez"]},
    **{p: "Europe" for p in [
        "Rotterdam", "Antwerp", "Hamburg", "Gothenburg", "Skaw", "Bergen",
        "Gdansk", "Tallinn", "Riga", "Gibraltar", "Las Palmas", "Lisbon",
        "Malta Offshore", "Limassol", "Piraeus", "Istanbul", "Augusta",
    ]},
    **{p: "Americas" for p in [
        "New York", "Houston", "Los Angeles", "Seattle", "Vancouver", "Balboa",
        "Cartagena", "Guayaquil", "Port of Spain", "Santos", "Rio de Janeiro",
        "Buenos Aires", "Montevideo",
    ]},
    **{p: "Africa" for p in ["Durban", "Cape Town", "Tema"]},
}

# Ports where we have official bunker sales data (excel or MPA)
SALES_PORTS: dict[str, str] = {
    "Singapore":  "MPA monthly",
    "Rotterdam":  "Quarterly",
    "Antwerp":    "Quarterly",
    "Balboa":     "Monthly",   # Panama canal stats labelled as Balboa in S&B
    "Fujairah":   "Platts/Fedcom monthly",
    "Malta Offshore": "Eurostat monthly",
    "Limassol":       "Eurostat monthly",
    "Istanbul":      "EPDK monthly (proxy)",
    "Tallinn":       "Eurostat monthly",
}


def _port_key(name: str) -> str:
    return name.lower().replace(" ", "_")


SB_PORT_MAP:      dict[str, str] = {_port_key(p): p for p in _SB_PORTS}
INTEGR8_PORT_MAP: dict[str, str] = {_port_key(p): p for p in _INTEGR8_PORTS}

HUBS      = sorted(set(SB_PORT_MAP) | set(INTEGR8_PORT_MAP))
GRADES    = ["VLSFO", "HSFO", "LSMGO", "MGO"]
LOOKBACKS = [30, 60, 90, 180, 365, 730, 0]


# ── Hub / grade → QCIntel code mapping (Singapore & Rotterdam only) ───────────

QUANTUM_MAP: dict[tuple[str, str], dict[str, str]] = {
    ("singapore", "HSFO"):  {"cargo": "H3FSIS", "bunker": "B3ESIC"},
    ("singapore", "VLSFO"): {"cargo": "MFFSIS", "bunker": "MFESIC"},
    ("rotterdam", "HSFO"):  {"cargo": "H3FARS", "bunker": "BHEARA"},
    ("rotterdam", "VLSFO"): {"cargo": "MFFARS", "bunker": "LSFNWS"},
}

QUANTUM_LABELS: dict[str, str] = {
    "H3FSIS": "Quantum 380 FOB SG",
    "B3ESIC": "Quantum 380 bunker SG",
    "MFFSIS": "Quantum VLSFO FOB SG",
    "MFESIC": "Quantum VLSFO bunker SG",
    "H3FARS": "Quantum 380 FOB ARA",
    "BHEARA": "Quantum 380 bunker ARA",
    "MFFARS": "Quantum VLSFO FOB ARA",
    "LSFNWS": "Quantum LSFNWS NWE phys",
}


# ── Queries ───────────────────────────────────────────────────────────────────

def _cutoff(days: int) -> str:
    if days == 0:
        return "2000-01-01"
    return (date.today() - timedelta(days=days)).isoformat()


def get_all_ports_latest() -> list[dict]:
    """Return latest S&B prices for every port — used by the overview table."""
    with _lock:
        conn = _conn()

        rows = conn.execute("""
            WITH lat AS (
                SELECT port_name, MAX(date) AS d
                FROM bunker_prices WHERE source='shipandbunker'
                GROUP BY port_name
            )
            SELECT b.port_name, b.country, l.d AS date,
                   MAX(CASE WHEN b.fuel_grade='VLSFO' THEN b.price_usd END) AS vlsfo,
                   MAX(CASE WHEN b.fuel_grade='HSFO'  THEN b.price_usd END) AS hsfo,
                   MAX(CASE WHEN b.fuel_grade='LSMGO' THEN b.price_usd END) AS lsmgo,
                   MAX(CASE WHEN b.fuel_grade='MGO'   THEN b.price_usd END) AS mgo
            FROM bunker_prices b
            JOIN lat l ON b.port_name=l.port_name AND b.date=l.d
            WHERE b.source='shipandbunker'
            GROUP BY b.port_name ORDER BY b.port_name
        """).fetchall()

        wow_rows = conn.execute("""
            WITH lat AS (
                SELECT port_name, MAX(date) AS d
                FROM bunker_prices WHERE source='shipandbunker' AND fuel_grade='VLSFO'
                GROUP BY port_name
            ),
            prev AS (
                SELECT b.port_name, MAX(b.date) AS prev_d
                FROM bunker_prices b JOIN lat l ON b.port_name=l.port_name
                WHERE b.source='shipandbunker' AND b.fuel_grade='VLSFO'
                  AND b.date <= date(l.d, '-7 days')
                GROUP BY b.port_name
            )
            SELECT lat.port_name, b_n.price_usd AS now_p, b_p.price_usd AS prv_p
            FROM lat
            JOIN  bunker_prices b_n ON b_n.port_name=lat.port_name
                  AND b_n.date=lat.d AND b_n.source='shipandbunker' AND b_n.fuel_grade='VLSFO'
            LEFT JOIN prev       ON prev.port_name=lat.port_name
            LEFT JOIN bunker_prices b_p ON b_p.port_name=prev.port_name
                  AND b_p.date=prev.prev_d AND b_p.source='shipandbunker'
                  AND b_p.fuel_grade='VLSFO'
        """).fetchall()
        wow_map: dict[str, float | None] = {}
        for r in wow_rows:
            if r["now_p"] is not None and r["prv_p"] is not None:
                wow_map[r["port_name"]] = round(r["now_p"] - r["prv_p"], 1)

        i8_rows = conn.execute("""
            WITH lat AS (
                SELECT port_name, MAX(date) AS d
                FROM bunker_prices WHERE source='integr8' GROUP BY port_name
            )
            SELECT b.port_name,
                   MAX(CASE WHEN b.fuel_grade='VLSFO' THEN b.price_usd END) AS vlsfo
            FROM bunker_prices b
            JOIN lat l ON b.port_name=l.port_name AND b.date=l.d
            WHERE b.source='integr8' GROUP BY b.port_name
        """).fetchall()
        i8_map = {r["port_name"]: r["vlsfo"] for r in i8_rows}

        conn.close()

    result = []
    for r in rows:
        name  = r["port_name"]
        vlsfo = r["vlsfo"]
        hsfo  = r["hsfo"]
        hi5   = round(vlsfo - hsfo, 1) if vlsfo and hsfo else None
        result.append({
            "port":     name,
            "key":      _port_key(name),
            "region":   PORT_REGIONS.get(name, "Other"),
            "date":     r["date"],
            "vlsfo":    vlsfo,
            "hsfo":     hsfo,
            "lsmgo":    r["lsmgo"],
            "mgo":      r["mgo"],
            "hi5":      hi5,
            "wow":      wow_map.get(name),
            "i8_vlsfo": i8_map.get(name),
            "sales":    SALES_PORTS.get(name),
        })
    # Fill in all catalogue ports not already in result (null prices shown as -)
    result_ports = {r['port'] for r in result}
    all_catalogue = sorted(set(_SB_PORTS) | set(_INTEGR8_PORTS))
    for port in all_catalogue:
        if port not in result_ports:
            result.append({
                'port':     port,
                'key':      _port_key(port),
                'region':   PORT_REGIONS.get(port, 'Other'),
                'date':     None,
                'vlsfo':    None,
                'hsfo':     None,
                'lsmgo':    None,
                'mgo':      None,
                'hi5':      None,
                'wow':      None,
                'i8_vlsfo': None,
                'sales':    SALES_PORTS.get(port),
            })
    result.sort(key=lambda r: r['port'])
    return result


def get_chart_series(hub: str, grade: str, days: int = 90) -> dict:
    """
    Return aligned time-series for all sources as lists.
    Works for any port; quantum series will be empty for non-SG/RTM hubs.
    """
    q_codes  = QUANTUM_MAP.get((hub, grade), {})
    port_sb  = SB_PORT_MAP.get(hub, "")
    port_i8  = INTEGR8_PORT_MAP.get(hub, "")
    cut      = _cutoff(days)

    with _lock:
        conn = _conn()

        def _bp(source: str, port: str) -> dict[str, float]:
            if not port:
                return {}
            rows = conn.execute(
                """SELECT date, price_usd FROM bunker_prices
                   WHERE source=? AND port_name=? AND fuel_grade=? AND date>=?
                   ORDER BY date""",
                (source, port, grade, cut),
            ).fetchall()
            return {r["date"]: r["price_usd"] for r in rows}

        def _qp(code: str) -> dict[str, float]:
            if not code:
                return {}
            rows = conn.execute(
                """SELECT date, value FROM quantum_prices
                   WHERE prime_code=? AND laycan_code='SPO' AND date>=?
                   ORDER BY date""",
                (code, cut),
            ).fetchall()
            return {r["date"]: r["value"] for r in rows}

        i8 = _bp("integr8", port_i8)
        sb = _bp("shipandbunker", port_sb)
        qc = _qp(q_codes.get("cargo", ""))
        qb = _qp(q_codes.get("bunker", ""))

        dates = sorted(set(i8) | set(sb) | set(qc) | set(qb))
        conn.close()

    return {
        "dates":          dates,
        "integr8":        [i8.get(d) for d in dates],
        "sb":             [sb.get(d) for d in dates],
        "quantum_cargo":  [qc.get(d) for d in dates],
        "quantum_bunker": [qb.get(d) for d in dates],
    }


def get_latest(hub: str, grade: str) -> dict:
    """Return the most recent price from each source for a hub/grade."""
    q_codes = QUANTUM_MAP.get((hub, grade), {})
    port_sb = SB_PORT_MAP.get(hub, "")
    port_i8 = INTEGR8_PORT_MAP.get(hub, "")

    with _lock:
        conn = _conn()

        def _bp_latest(source: str, port: str):
            if not port:
                return None, None
            row = conn.execute(
                """SELECT price_usd, date FROM bunker_prices
                   WHERE source=? AND port_name=? AND fuel_grade=?
                   ORDER BY date DESC LIMIT 1""",
                (source, port, grade),
            ).fetchone()
            return (row["price_usd"], row["date"]) if row else (None, None)

        def _qp_latest(code: str):
            if not code:
                return None, None
            row = conn.execute(
                """SELECT value, date FROM quantum_prices
                   WHERE prime_code=? AND laycan_code='SPO'
                   ORDER BY date DESC LIMIT 1""",
                (code,),
            ).fetchone()
            return (row["value"], row["date"]) if row else (None, None)

        i8_val,  i8_date  = _bp_latest("integr8", port_i8)
        sb_val,  sb_date  = _bp_latest("shipandbunker", port_sb)
        qc_val,  qc_date  = _qp_latest(q_codes.get("cargo", ""))
        qb_val,  qb_date  = _qp_latest(q_codes.get("bunker", ""))
        conn.close()

    def _prem(delivered, cargo):
        if delivered is not None and cargo is not None:
            return round(delivered - cargo, 2)
        return None

    return {
        "integr8":         {"value": i8_val, "date": i8_date},
        "sb":              {"value": sb_val, "date": sb_date},
        "quantum_cargo":   {"value": qc_val, "date": qc_date,
                            "code": q_codes.get("cargo", "")},
        "quantum_bunker":  {"value": qb_val, "date": qb_date,
                            "code": q_codes.get("bunker", "")},
        "premium_integr8": _prem(i8_val, qc_val),
        "premium_sb":      _prem(sb_val, qc_val),
        "premium_qbunker": _prem(qb_val, qc_val),
    }


def get_all_latest() -> dict:
    """Return latest prices for all hub/grade combos — for the overview KPI table."""
    result = {}
    # Only compute for the two hubs with full Quantum coverage to keep existing table
    for hub in ["singapore", "rotterdam"]:
        for grade in ["HSFO", "VLSFO"]:
            result[(hub, grade)] = get_latest(hub, grade)
    return result


def build_plotly_figure(hub: str, grade: str, days: int) -> dict:
    """Build a Plotly figure dict for the price comparison chart."""
    series      = get_chart_series(hub, grade, days)
    port_name   = SB_PORT_MAP.get(hub) or INTEGR8_PORT_MAP.get(hub) or hub.replace("_", " ").title()
    grade_label = {"HSFO": "HSFO 380", "VLSFO": "VLSFO 0.5%S",
                   "LSMGO": "LSMGO", "MGO": "MGO"}.get(grade, grade)

    COLORS = {
        "integr8":        "#f0883e",
        "sb":             "#58a6ff",
        "quantum_cargo":  "#8b949e",
        "quantum_bunker": "#3fb950",
    }
    q_codes     = QUANTUM_MAP.get((hub, grade), {})
    cargo_code  = q_codes.get("cargo", "")
    bunker_code = q_codes.get("bunker", "")
    NAMES = {
        "integr8":        "Integr8 (delivered)",
        "sb":             "Ship & Bunker (delivered)",
        "quantum_cargo":  QUANTUM_LABELS.get(cargo_code,  f"Quantum {cargo_code} (FOB)") if cargo_code else "Quantum (FOB)",
        "quantum_bunker": QUANTUM_LABELS.get(bunker_code, f"Quantum {bunker_code}") if bunker_code else "Quantum (bunker)",
    }
    DASH  = {"integr8": "solid", "sb": "solid",
             "quantum_cargo": "dash", "quantum_bunker": "solid"}
    WIDTH = {"integr8": 2, "sb": 2, "quantum_cargo": 1.5, "quantum_bunker": 2}

    traces = []
    for key in ["quantum_cargo", "quantum_bunker", "integr8", "sb"]:
        vals = series[key]
        if not any(v is not None for v in vals):
            continue   # skip entirely empty series
        traces.append({
            "type": "scatter", "mode": "lines",
            "name": NAMES[key],
            "x": series["dates"], "y": vals,
            "line":  {"color": COLORS[key], "width": WIDTH[key], "dash": DASH[key]},
            "hovertemplate": "%{y:.2f} $/mt<extra>" + NAMES[key] + "</extra>",
            "connectgaps": False,
        })

    layout = {
        "title": None,
        "paper_bgcolor": "transparent",
        "plot_bgcolor":  "transparent",
        "font": {"color": "#e6edf3", "size": 11, "family": "Inter, system-ui, sans-serif"},
        "legend": {"orientation": "h", "x": 0, "y": 1.07,
                   "font": {"size": 11}, "bgcolor": "transparent"},
        "xaxis": {"gridcolor": "#30363d", "linecolor": "#30363d",
                  "tickcolor": "#30363d", "zeroline": False},
        "yaxis": {"gridcolor": "#30363d", "linecolor": "#30363d",
                  "tickcolor": "#30363d", "zeroline": False,
                  "title": "USD/MT", "titlefont": {"size": 10, "color": "#8b949e"}},
        "hovermode": "x unified",
        "hoverlabel": {
            "bgcolor":     "#1c2128",
            "bordercolor": "#444c56",
            "font": {"color": "#e6edf3", "size": 11,
                     "family": "Inter, system-ui, sans-serif"},
            "namelength": -1,
        },
        "margin": {"l": 50, "r": 10, "t": 30, "b": 40},
    }

    return {"data": traces, "layout": layout}
