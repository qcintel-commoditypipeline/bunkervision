"""
Data ingestion for all ports:
  1. seed_vessels_from_json()    — Singapore MPA registry (220 vessels from PDF)
  2. load_mysql_bunker_stats()   — Singapore MPA monthly via ODBC → mpa_monthly + port_bunker_sales
  3. load_fujairah_stats()       — Fujairah Platts/Fedcom monthly via ODBC (m³→mt conversion)
  4. load_excel_stats(port)      — Panama/Rotterdam/Antwerp from Excel files in data/

Run standalone:  python mpa_scraper.py
"""
from __future__ import annotations

import json
import os
from datetime import date
from pathlib import Path

from loguru import logger

import db
from config import BASE_DIR

VESSEL_SEED_FILE = BASE_DIR / "data" / "sg_bunker_vessels.json"
DATA_DIR         = BASE_DIR / "data"

def _mysql_connect(dsn: str = "StackheroMySQL"):
    """
    Return a DB-API connection.
    Tries pyodbc DSN first (works on Windows); falls back to direct pymysql
    using MYSQL_HOST/PORT/USER/PASSWORD env vars (used on the server).
    """
    # Direct connection via env vars (server)
    host = os.getenv("MYSQL_HOST")
    if host:
        import pymysql
        ssl_ca = os.getenv("MYSQL_SSL_CA")
        return pymysql.connect(
            host=host,
            port=int(os.getenv("MYSQL_PORT", "3306")),
            user=os.getenv("MYSQL_USER", ""),
            password=os.getenv("MYSQL_PASSWORD", ""),
            database=os.getenv("MYSQL_DATABASE", "qc_intel"),
            ssl={"ca": ssl_ca} if ssl_ca else None,
            connect_timeout=10,
        )
    # ODBC DSN (Windows/local)
    import pyodbc
    return pyodbc.connect(f"DSN={dsn}", timeout=10)


_PBS_UPSERT = """
    INSERT INTO port_bunker_sales (port, month, fuel_type, volume_mt, ships_count)
    VALUES (?, ?, ?, ?, ?)
    ON CONFLICT (port, month, fuel_type) DO UPDATE SET
        volume_mt   = excluded.volume_mt,
        ships_count = COALESCE(excluded.ships_count, port_bunker_sales.ships_count)
"""


# ── Singapore vessel registry ──────────────────────────────────────────────────

def seed_vessels_from_json() -> int:
    """Seed bunker_vessels table from the MPA PDF extract JSON."""
    if not VESSEL_SEED_FILE.exists():
        logger.warning(f"Vessel seed file not found: {VESSEL_SEED_FILE}")
        return 0

    with open(VESSEL_SEED_FILE) as f:
        vessels = json.load(f)

    today = date.today()
    upsert_sql = """
        INSERT INTO bunker_vessels (imo, name, port, licensee, capacity_mt, last_updated)
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT (imo) DO UPDATE SET
            name         = excluded.name,
            port         = excluded.port,
            capacity_mt  = excluded.capacity_mt,
            last_updated = excluded.last_updated
    """
    rows = [
        [f"NAME:{v['name']}", v["name"], "singapore", "MPA-SG", v.get("dwt"), today]
        for v in vessels
    ]
    db.executemany(upsert_sql, rows)
    logger.info(f"Seeded {len(rows)} bunker vessels from JSON")

    try:
        from ais_client import refresh_bunker_names
        refresh_bunker_names()
    except Exception:
        pass

    return len(rows)


# ── Singapore monthly stats ────────────────────────────────────────────────────

def load_mysql_bunker_stats(dsn: str = "StackheroMySQL") -> int:
    """Pull Singapore MPA monthly sales via MySQL. Writes mpa_monthly + port_bunker_sales."""
    GRADE_MAP = {
        "lsfo":     "VLSFO",
        "mfo":      "HSFO",
        "mgo":      "MGO",
        "lsmgo":    "LSMGO",
        "lng":      "LNG",
        "methanol": "Methanol",
        "ammonia":  "Ammonia",
        "total":    "TOTAL",
    }

    logger.info(f"Connecting to MySQL (Singapore)…")
    try:
        conn = _mysql_connect(dsn)
    except Exception as e:
        logger.error(f"MySQL connection failed: {e}")
        return 0

    try:
        cursor = conn.cursor()
        cols = ", ".join(["date"] + list(GRADE_MAP.keys()))
        cursor.execute(
            f"SELECT {cols} FROM qc_intel.energy_oil_singapore_mpa_bunkersales_monthly "
            f"ORDER BY date"
        )
        rows = cursor.fetchall()
    except Exception as e:
        logger.error(f"MySQL query failed: {e}")
        conn.close()
        return 0
    finally:
        conn.close()

    mpa_upsert = """
        INSERT INTO mpa_monthly (month, fuel_type, volume_mt, ships_count)
        VALUES (?, ?, ?, NULL)
        ON CONFLICT (month, fuel_type) DO UPDATE SET volume_mt = excluded.volume_mt
    """

    mpa_records = []
    pbs_records = []
    for row in rows:
        month_dt = row[0]
        if hasattr(month_dt, "date"):
            month_dt = month_dt.date()
        month_dt = month_dt.replace(day=1)

        for i, (col, grade) in enumerate(GRADE_MAP.items(), start=1):
            val = row[i]
            if val is not None:
                try:
                    mt = float(val) * 1000  # source is '000 tonnes
                    mpa_records.append([month_dt, grade, mt])
                    pbs_records.append(["singapore", month_dt, grade, mt, None])
                except (TypeError, ValueError):
                    pass

    if mpa_records:
        db.executemany(mpa_upsert, mpa_records)
        db.executemany(_PBS_UPSERT, pbs_records)
        logger.info(f"Singapore: upserted {len(mpa_records)} monthly stat rows")

    return len(mpa_records)


# ── Fujairah monthly stats ─────────────────────────────────────────────────────

# Density factors (mt per m³ @ 15°C) for Fujairah source data
_FUJAIRAH_DENSITY = {
    "lsfo_180cst": 0.97,
    "lsfo_380cst": 0.98,
    "mfo_380cst":  0.95,
    "mgo":         0.84,
    "lsmgo":       0.85,
    "lubricants":  0.90,
}

def load_fujairah_stats(dsn: str = "StackheroMySQL") -> int:
    """Pull Fujairah Platts/Fedcom monthly data (m³) from MySQL, convert to mt."""
    logger.info(f"Connecting to MySQL (Fujairah)…")
    try:
        conn = _mysql_connect(dsn)
    except Exception as e:
        logger.error(f"MySQL connection failed: {e}")
        return 0

    cols = ["date"] + list(_FUJAIRAH_DENSITY.keys()) + ["total"]
    try:
        cursor = conn.cursor()
        cursor.execute(
            f"SELECT {', '.join(cols)} "
            f"FROM qc_intel.energy_oil_uae_plattsfedcom_fujairahbunker_monthly_raw_long "
            f"ORDER BY date"
        )
        rows = cursor.fetchall()
    except Exception as e:
        logger.error(f"Fujairah MySQL query failed: {e}")
        conn.close()
        return 0
    finally:
        conn.close()

    def _nz(x) -> float:
        return float(x) if x is not None else 0.0

    records = []
    for row in rows:
        month_dt = row[0]
        if hasattr(month_dt, "date"):
            month_dt = month_dt.date()
        month_dt = month_dt.replace(day=1)

        d = dict(zip(cols[1:], row[1:]))

        hsfo_mt  = _nz(d.get("lsfo_180cst")) * 0.97 + _nz(d.get("lsfo_380cst")) * 0.98
        vlsfo_mt = _nz(d.get("mfo_380cst"))  * 0.95
        mgo_mt   = _nz(d.get("mgo"))          * 0.84
        lsmgo_mt = _nz(d.get("lsmgo"))        * 0.85
        lubes_mt = _nz(d.get("lubricants"))   * 0.90
        total_mt = hsfo_mt + vlsfo_mt + mgo_mt + lsmgo_mt + lubes_mt

        grade_vols = [
            ("HSFO",  hsfo_mt),
            ("VLSFO", vlsfo_mt),
            ("MGO",   mgo_mt),
            ("LSMGO", lsmgo_mt),
            ("TOTAL", total_mt),
        ]
        for grade, mt in grade_vols:
            if mt > 0:
                records.append(["fujairah", month_dt, grade, mt, None])

    if records:
        db.executemany(_PBS_UPSERT, records)
        logger.info(f"Fujairah: upserted {len(records)} monthly rows")

    return len(records)


# ── Excel loaders (Panama / Rotterdam / Antwerp) ───────────────────────────────

def _excel_to_records(port: str, df, col_map: dict, date_col: str = "Date") -> list:
    """
    Convert a DataFrame (from read_excel) to port_bunker_sales records.
    col_map: {source_col: fuel_type_label}  (use "TOTAL" for the total column)
    Rows with null dates are skipped.
    """
    import pandas as pd
    records = []
    for _, row in df.iterrows():
        raw_date = row.get(date_col)
        if raw_date is None or (hasattr(raw_date, "__class__") and str(raw_date) == "NaT"):
            continue
        try:
            month_dt = pd.Timestamp(raw_date).date().replace(day=1)
        except Exception:
            continue

        for src_col, fuel_type in col_map.items():
            val = row.get(src_col)
            if val is not None:
                try:
                    mt = float(val)
                    if mt > 0:
                        records.append([port, month_dt, fuel_type, mt, None])
                except (TypeError, ValueError):
                    pass
    return records


def load_panama_stats() -> int:
    """Load Panama monthly bunker sales from data/panama_bunker_sales.xlsx."""
    try:
        import pandas as pd
    except ImportError:
        logger.error("pandas not installed")
        return 0

    path = DATA_DIR / "panama_bunker_sales.xlsx"
    if not path.exists():
        logger.info(f"Panama Excel not found: {path} — skipping")
        return 0

    logger.info(f"Loading Panama data from {path}…")
    try:
        df = pd.read_excel(path, sheet_name="Sheet1", engine="openpyxl")
        df.columns = [str(c).strip() for c in df.columns]
    except Exception as e:
        logger.error(f"Failed to read Panama Excel: {e}")
        return 0

    col_map = {
        "Total":         "TOTAL",
        "Total bio":     "Biofuels",
        "Total VLSFO":   "VLSFO",
        "Total RMG 380": "HSFO",
        "Total MGO":     "MGO",
        "Total LSMGO":   "LSMGO",
    }
    # Tolerate pre-renamed columns too
    for alias, canonical in [("Biofuels", "Biofuels"), ("VLSFO", "VLSFO"),
                               ("HSFO", "HSFO"), ("MGO", "MGO"), ("LSMGO", "LSMGO")]:
        if alias in df.columns and alias not in col_map:
            col_map[alias] = canonical

    records = _excel_to_records("panama", df, col_map)
    if records:
        db.executemany(_PBS_UPSERT, records)
        logger.info(f"Panama: upserted {len(records)} rows")
    return len(records)


def load_rotterdam_stats() -> int:
    """Load Rotterdam quarterly bunker sales from data/rotterdam_bunker_sales.xlsx."""
    try:
        import pandas as pd
    except ImportError:
        logger.error("pandas not installed")
        return 0

    path = DATA_DIR / "rotterdam_bunker_sales.xlsx"
    if not path.exists():
        logger.info(f"Rotterdam Excel not found: {path} — skipping")
        return 0

    logger.info(f"Loading Rotterdam data from {path}…")
    try:
        df = pd.read_excel(path, sheet_name="Tabelle1", engine="openpyxl", header=0)
        df.columns = [str(c).strip() for c in df.columns]
    except Exception as e:
        logger.error(f"Failed to read Rotterdam Excel: {e}")
        return 0

    # Normalise: PQ renames "MGO" → "LSMGO" and "All bunkers" → "Total"
    df = df.rename(columns={"MGO": "LSMGO", "All bunkers": "Total"}, errors="ignore")

    # LNG m³ → mt (0.45 t/m³)
    if "LNG" in df.columns:
        df["LNG"] = pd.to_numeric(df["LNG"], errors="coerce").fillna(0) * 0.45

    # Biofuels = Bio-fuel oil + Bio-distillate bunkers
    bio_cols = [c for c in df.columns if "bio" in c.lower() or "Bio" in c]
    if bio_cols and "Biofuels" not in df.columns:
        df["Biofuels"] = sum(pd.to_numeric(df[c], errors="coerce").fillna(0) for c in bio_cols)

    col_map = {c: c for c in ["ULSFO", "VLSFO", "HSFO", "LSMGO", "MDO",
                               "Methanol", "LNG", "Biofuels", "Total"]
               if c in df.columns}
    col_map["Total"] = "TOTAL"

    records = _excel_to_records("rotterdam", df, col_map)
    if records:
        db.executemany(_PBS_UPSERT, records)
        logger.info(f"Rotterdam: upserted {len(records)} rows")
    return len(records)


def load_antwerp_stats() -> int:
    """Load Antwerp quarterly bunker sales from data/antwerp_bunker_sales.xlsx."""
    try:
        import pandas as pd
    except ImportError:
        logger.error("pandas not installed")
        return 0

    path = DATA_DIR / "antwerp_bunker_sales.xlsx"
    if not path.exists():
        logger.info(f"Antwerp Excel not found: {path} — skipping")
        return 0

    logger.info(f"Loading Antwerp data from {path}…")
    try:
        df = pd.read_excel(path, sheet_name="Sheet1", engine="openpyxl", header=0)
        df.columns = [str(c).strip() for c in df.columns]
    except Exception as e:
        logger.error(f"Failed to read Antwerp Excel: {e}")
        return 0

    # Raw Excel has no header on date/quarter cols — pandas names them Unnamed: N
    unnamed = [c for c in df.columns if str(c).startswith("Unnamed")]
    if len(unnamed) >= 2:
        df = df.rename(columns={unnamed[0]: "Date", unnamed[1]: "Quarter"})
    elif len(unnamed) == 1:
        df = df.rename(columns={unnamed[0]: "Date"})

    df = df.rename(columns={
        "Gasoil":                      "LSMGO",
        "Marine diesel":               "MDO",
        "Unspecified FO":              "FO_Unspecified",
        "Ultra low sulphur fuel oil":  "ULSFO",
        "Ultra low sulphur fuel oil ": "ULSFO",
        "Very low sulphur fuel oil":   "VLSFO",
        "High sulphur fuel oil":       "HSFO",
        "Biofuel":                     "Biofuels",
    }, errors="ignore")

    col_map = {c: c for c in ["ULSFO", "VLSFO", "HSFO", "LSMGO", "MDO",
                               "FO_Unspecified", "LNG", "Methanol", "Biofuels", "Total"]
               if c in df.columns}
    col_map["Total"] = "TOTAL"

    records = _excel_to_records("antwerp", df, col_map)
    if records:
        db.executemany(_PBS_UPSERT, records)
        logger.info(f"Antwerp: upserted {len(records)} rows")
    return len(records)


_EUROSTAT_OIL_URL = (
    "https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data/nrg_cb_oilm"
)

# Eurostat SIEC codes → bunker fuel grade mapping
_EUROSTAT_GRADE_SIECS = {
    "VLSFO": "O46811",   # fuel oil very low sulphur <0.5%
    "HSFO":  "O4682",    # fuel oil high sulphur >=1%
    "MGO":   "O4671",    # gas oil and diesel oil (marine portion ≈ total minus road diesel)
}
_ROAD_DIESEL_SIEC = "O46711"  # subtracted from gas oil to remove road diesel

# Countries where INTMARB ≈ a single port (island/micro-state single-hub markets)
_EUROSTAT_PORTS = {
    "MT": "Malta Offshore",
    "CY": "Limassol",
    "EE": "Tallinn",
}


def load_eurostat_stats() -> int:
    """
    Fetch Eurostat INTMARB (international marine bunkers) monthly energy data for
    single-hub island states where country total == port total.
    Currently: Malta → Malta Offshore, Cyprus → Limassol.
    Returns total rows upserted.
    """
    import requests as _req

    total = 0
    for geo, port_name in _EUROSTAT_PORTS.items():
        try:
            r = _req.get(
                _EUROSTAT_OIL_URL,
                params={
                    "format": "JSON", "lang": "EN",
                    "geo": geo, "nrg_bal": "INTMARB",
                    "sinceTimePeriod": "2020-01",
                },
                timeout=30,
            )
            r.raise_for_status()
            d = r.json()
        except Exception as e:
            logger.warning(f"Eurostat {geo}: fetch failed — {e}")
            continue

        dims = d["dimension"]
        siec_idx  = dims["siec"]["category"]["index"]   # code → flat pos
        time_idx  = dims["time"]["category"]["index"]
        time_rev  = {v: k for k, v in time_idx.items()}  # pos → YYYY-MM

        sz_siec = len(siec_idx)
        sz_time = len(time_idx)
        values  = d.get("value", {})

        def _series(siec_code: str) -> dict[str, float]:
            """Return {YYYY-MM: value_kt} for one SIEC code."""
            if siec_code not in siec_idx:
                return {}
            sp = siec_idx[siec_code]
            out = {}
            for flat, val in values.items():
                i = int(flat)
                tp = i % sz_time; i //= sz_time
                # geo=0, unit=0, nrg_bal=0 — single value each
                i //= 1; i //= 1; i //= 1
                si = i % sz_siec
                if si == sp and val is not None:
                    out[time_rev[tp]] = float(val)
            return out

        road_diesel = _series(_ROAD_DIESEL_SIEC)

        records = []
        for grade, siec_code in _EUROSTAT_GRADE_SIECS.items():
            series = _series(siec_code)
            for month, kt in series.items():
                if grade == "MGO":
                    kt -= road_diesel.get(month, 0.0)
                volume_mt = round(kt * 1_000, 1)
                if volume_mt <= 0:
                    continue
                records.append((port_name, month + "-01", grade, volume_mt, None))

        if records:
            db.executemany(_PBS_UPSERT, records)
            logger.info(f"Eurostat {geo} ({port_name}): upserted {len(records)} rows")
            total += len(records)

    return total



def load_istanbul_epdk_stats() -> int:
    try:
        import pandas as pd
        from datetime import date as date_type
    except ImportError:
        logger.error('pandas not installed')
        return 0

    path = DATA_DIR / 'istanbul_epdk.xlsx'
    if not path.exists():
        logger.info(f'Istanbul EPDK Excel not found: {path} - skipping')
        return 0

    logger.info(f'Loading Istanbul EPDK data from {path}...')
    try:
        df = pd.read_excel(path, sheet_name='T5_6_Bunker_YoY', engine='openpyxl')
    except Exception as e:
        logger.error(f'Failed to read Istanbul EPDK Excel: {e}')
        return 0

    current_year = date_type.today().year
    marine = df[df['Product'].astype(str).str.contains('Denizcilik', na=False)].copy()
    # Cap 2 months before today: EPDK has ~2-month lag; avoids date-parse bugs assigning old files to future months
    cutoff = date_type.today().replace(day=1)
    for _ in range(2):
        m = cutoff.month - 1 or 12
        y = cutoff.year - (1 if cutoff.month == 1 else 0)
        cutoff = cutoff.replace(year=y, month=m)
    marine = marine[(marine['Year'] >= 2016) &
                    ((marine['Year'] < cutoff.year) |
                     ((marine['Year'] == cutoff.year) & (marine['Month'] <= cutoff.month)))].copy()
    marine = marine.dropna(subset=['Year', 'Month', 'Volume_t'])
    marine = marine[marine['Volume_t'] > 0]

    records = []
    seen = set()
    for _, row in marine.iterrows():
        try:
            yr = int(row['Year'])
            mo = int(row['Month'])
            vol = float(row['Volume_t'])
            month_str = f'{yr}-{mo:02d}-01'
            key = (yr, mo)
            if key in seen:
                continue
            seen.add(key)
            records.append(('Istanbul', month_str, 'TOTAL', vol, None))
        except (TypeError, ValueError):
            continue

    if records:
        db.execute("DELETE FROM port_bunker_sales WHERE port='Istanbul'")
        db.executemany(_PBS_UPSERT, records)
        logger.info(f'Istanbul EPDK: upserted {len(records)} rows')
    return len(records)

def load_all_port_stats(dsn: str = "StackheroMySQL") -> None:
    """Load historical data for every port that has a configured source."""
    load_mysql_bunker_stats(dsn)
    load_fujairah_stats(dsn)
    load_panama_stats()
    load_rotterdam_stats()
    load_antwerp_stats()
    load_eurostat_stats()
    load_istanbul_epdk_stats()


if __name__ == "__main__":
    logger.info("Seeding vessels…")
    n = seed_vessels_from_json()
    print(f"  {n} vessels seeded")

    logger.info("Loading all port stats…")
    load_all_port_stats()

    rows = db.query(
        "SELECT port, fuel_type, COUNT(*), MIN(month), MAX(month) "
        "FROM port_bunker_sales GROUP BY port, fuel_type ORDER BY port, fuel_type"
    )
    print("\nport_bunker_sales summary:")
    for r in rows:
        print(f"  {r[0]:12s}  {r[1]:10s}  {r[2]} rows  {r[3]} → {r[4]}")
