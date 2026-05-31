from __future__ import annotations
import threading
import duckdb
from config import DB_PATH

_lock = threading.Lock()
_conn: duckdb.DuckDBPyConnection | None = None


def get_conn() -> duckdb.DuckDBPyConnection:
    global _conn
    if _conn is None:
        _conn = duckdb.connect(str(DB_PATH))
        _init_schema(_conn)
    return _conn


def _init_schema(conn: duckdb.DuckDBPyConnection) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS bunker_vessels (
            imo          TEXT PRIMARY KEY,
            mmsi         TEXT,
            name         TEXT,
            port         TEXT DEFAULT 'singapore',
            licensee     TEXT,
            capacity_mt  REAL,
            last_updated DATE
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS ais_positions (
            id         BIGINT,
            imo        TEXT,
            mmsi       TEXT,
            name       TEXT,
            lat        REAL,
            lon        REAL,
            sog        REAL,
            cog        REAL,
            heading    INTEGER,
            nav_status INTEGER,
            ts         TIMESTAMP
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS vessel_cache (
            mmsi      TEXT PRIMARY KEY,
            imo       TEXT,
            name      TEXT,
            type_code INTEGER,
            draught   REAL,
            last_seen TIMESTAMP
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS vessel_positions (
            mmsi TEXT PRIMARY KEY,
            imo  TEXT,
            name TEXT,
            lat  REAL,
            lon  REAL,
            sog  REAL,
            ts   TIMESTAMP
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS draught_history (
            id      BIGINT,
            mmsi    TEXT,
            draught REAL,
            ts      TIMESTAMP
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS bunkering_events (
            id                BIGINT PRIMARY KEY,
            bunker_imo        TEXT,
            bunker_name       TEXT,
            bunker_mmsi       TEXT,
            recipient_mmsi    TEXT,
            recipient_name    TEXT,
            start_ts          TIMESTAMP,
            end_ts            TIMESTAMP,
            duration_min      INTEGER,
            lat               REAL,
            lon               REAL,
            estimated_mt      REAL,
            draught_start     REAL,
            draught_end       REAL,
            draught_delta_m   REAL,
            draught_vol_est   REAL,
            open              BOOLEAN DEFAULT TRUE
        )
    """)
    conn.execute("""
        CREATE SEQUENCE IF NOT EXISTS event_id_seq START 1
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS port_bunker_sales (
            port        TEXT,
            month       DATE,
            fuel_type   TEXT,
            volume_mt   REAL,
            ships_count INTEGER,
            PRIMARY KEY (port, month, fuel_type)
        )
    """)
    # Migrations for existing tables
    try:
        conn.execute("ALTER TABLE bunkering_events ADD COLUMN port TEXT DEFAULT 'singapore'")
    except Exception:
        pass
    try:
        conn.execute("ALTER TABLE bunker_vessels ADD COLUMN port TEXT DEFAULT 'singapore'")
    except Exception:
        pass
    conn.execute("""
        CREATE TABLE IF NOT EXISTS mpa_monthly (
            month        DATE,
            fuel_type    TEXT,
            volume_mt    REAL,
            ships_count  INTEGER,
            PRIMARY KEY (month, fuel_type)
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS monthly_demand_estimates (
            port            TEXT,
            year            INTEGER,
            month           INTEGER,
            estimated_mt    REAL,
            event_count     INTEGER,
            pct_vs_seasonal REAL,
            saved_at        TIMESTAMP,
            PRIMARY KEY (port, year, month)
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS eq_bunkering_found (
            imo                TEXT PRIMARY KEY,
            name               TEXT,
            ship_type          TEXT,
            flag               TEXT,
            gt                 INTEGER,
            year               TEXT,
            added_to_registry  BOOLEAN DEFAULT FALSE,
            found_at           TEXT
        )
    """)


def query(sql: str, params: list | None = None):
    with _lock:
        c = get_conn()
        if params:
            return c.execute(sql, params).fetchall()
        return c.execute(sql).fetchall()


def query_df(sql: str, params: list | None = None):
    with _lock:
        c = get_conn()
        if params:
            return c.execute(sql, params).df()
        return c.execute(sql).df()


def execute(sql: str, params: list | None = None) -> None:
    with _lock:
        c = get_conn()
        if params:
            c.execute(sql, params)
        else:
            c.execute(sql)


def executemany(sql: str, rows: list[list]) -> None:
    with _lock:
        c = get_conn()
        c.executemany(sql, rows)
