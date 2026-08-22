#!/usr/bin/env python3
"""
Ship & Bunker price scraper.

Fetches daily HSFO (IFO380), VLSFO, and LSMGO price history from the
Ship & Bunker undocumented graph API and stores it in the same SQLite
database as the Integr8 scraper, with source='shipandbunker'.

The S&B API returns ~3 years of daily data per request, so this script
is useful both as a historical backfill and as a daily complement to
the Integr8 feed (wider port coverage, different price assessment).

Run once for backfill, then daily alongside integr8_scraper.py:
    0 9 * * * /usr/bin/python3 /root/bunkervision/sb_scraper.py >> /root/bunkervision/scraper.log 2>&1
"""

import logging
import sqlite3
import sys
from datetime import date, datetime, timezone
from pathlib import Path

import requests

API_URL = "https://shipandbunker.com/a/.json"
DB_PATH = Path(__file__).parent / "bunker_prices.db"

# S&B market codes mapped to canonical port names (matching Integr8)
PORTS = {
    # ── Asia-Pacific ──────────────────────────────────────────────────────────
    "SG SIN": "Singapore",
    "CN HOK": "Hong Kong",
    "CN ZOS": "Zhoushan",
    "TW KHH": "Kaohsiung",
    "KR PUS": "Busan",
    "JP OSA": "Osaka",
    "PH MNL": "Manila",
    "VN SGN": "Ho Chi Minh City",
    "VN HPH": "Haiphong",
    "LK CMB": "Colombo",
    "IN COK": "Kochi",
    "IN MAA": "Chennai",
    "PK KHI": "Karachi",
    "AU SYD": "Sydney",
    # ── Middle East ───────────────────────────────────────────────────────────
    "AE FJR": "Fujairah",
    "SA JED": "Jeddah",
    "DJ JIB": "Djibouti",
    # ── Europe ────────────────────────────────────────────────────────────────
    "NL RTM": "Rotterdam",
    "BE ANT": "Antwerp",
    "DE HAM": "Hamburg",
    "SE GOT": "Gothenburg",
    "DK SKW": "Skaw",
    "NO BGO": "Bergen",
    "PL GDN": "Gdansk",
    "EE TLL": "Tallinn",
    "LV RIX": "Riga",
    "GI GIB": "Gibraltar",
    "ES LPA": "Las Palmas",
    "PT LIS": "Lisbon",
    "MT MLA": "Malta Offshore",
    "GR PIR": "Piraeus",
    "TR IST": "Istanbul",
    "IT AUG": "Augusta",
    # ── Americas ─────────────────────────────────────────────────────────────
    "US NYC": "New York",
    "US HOU": "Houston",
    "US LAX": "Los Angeles",
    "US SEA": "Seattle",
    "CA VAN": "Vancouver",
    "PA BLB": "Balboa",
    "CO CTG": "Cartagena",
    "EC GYE": "Guayaquil",
    "TT POS": "Port of Spain",
    "BR SSZ": "Santos",
    "BR RIO": "Rio de Janeiro",
    "AR BUE": "Buenos Aires",
    "UY MVD": "Montevideo",
    # ── Africa ───────────────────────────────────────────────────────────────
    "ZA DUR": "Durban",
    "ZA CPT": "Cape Town",
    "GH TEM": "Tema",
    # ── Oceania ───────────────────────────────────────────────────────────────
    "AU MEL": "Melbourne",
    "AU BNE": "Brisbane",
    "AU FRE": "Fremantle",
    "NZ AKL": "Auckland",
}

# S&B product codes → canonical fuel grade names
FUEL_MAP = {
    "IFO380": "HSFO",
    "VLSFO":  "VLSFO",
    "LSMGO":  "LSMGO",
    "MGO":    "MGO",
}

HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36",
    "Referer": "https://shipandbunker.com/prices/apac/sea/sg-sin-singapore",
    "Content-Type": "application/x-www-form-urlencoded",
    "X-Requested-With": "XMLHttpRequest",
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger(__name__)


def init_db(conn: sqlite3.Connection) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS bunker_prices (
            date        TEXT    NOT NULL,
            port_id     INTEGER NOT NULL,
            port_name   TEXT    NOT NULL,
            country     TEXT    NOT NULL,
            fuel_grade  TEXT    NOT NULL,
            price_usd   REAL    NOT NULL,
            fetched_at  TEXT    NOT NULL,
            source      TEXT    NOT NULL DEFAULT 'integr8',
            PRIMARY KEY (date, port_id, fuel_grade, source)
        )
    """)
    # Add source column to existing DB if it was created before this scraper
    try:
        conn.execute("ALTER TABLE bunker_prices ADD COLUMN source TEXT NOT NULL DEFAULT 'integr8'")
        log.info("Added 'source' column to existing bunker_prices table")
    except sqlite3.OperationalError:
        pass  # Column already exists

    conn.execute("""
        CREATE VIEW IF NOT EXISTS hi5_spread AS
        SELECT
            date,
            port_name,
            source,
            MAX(CASE WHEN fuel_grade = 'VLSFO' THEN price_usd END) AS vlsfo,
            MAX(CASE WHEN fuel_grade = 'HSFO'  THEN price_usd END) AS hsfo,
            MAX(CASE WHEN fuel_grade = 'LSMGO' THEN price_usd END) AS lsmgo,
            MAX(CASE WHEN fuel_grade = 'VLSFO' THEN price_usd END) -
            MAX(CASE WHEN fuel_grade = 'HSFO'  THEN price_usd END) AS spread
        FROM bunker_prices
        GROUP BY date, port_name, source
    """)
    conn.commit()


BATCH_SIZE = 10  # API silently drops ports beyond this limit


def fetch_port_batch(market_codes: list[str]) -> dict:
    """Fetch price history for a batch of market codes in one API call."""
    params = ["api-method=pricesForAllSeriesGet", "resource=MarketPriceGraph_Block"]
    for i, code in enumerate(market_codes):
        params.append(f"mc{i}={code}")
    payload = "&".join(params)

    resp = requests.post(API_URL, data=payload, headers=HEADERS, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    return data.get("api", {})


def parse_port_data(market_code: str, port_data: dict) -> list[tuple]:
    """Convert raw API response for one port into (date, port_name, fuel_grade, price) rows."""
    rows = []
    port_name = PORTS.get(market_code, market_code)
    country = market_code.split()[0]  # e.g. "SG" from "SG SIN"

    inner = port_data.get("data", {})
    day_lists = inner.get("day_list", {})
    prices_by_product = inner.get("prices", {})

    for sb_grade, canonical_grade in FUEL_MAP.items():
        if sb_grade not in prices_by_product:
            continue
        day_list = day_lists.get(sb_grade, {})
        dayprice = prices_by_product[sb_grade].get("dayprice", [])

        for day_idx, price in dayprice:
            if price is None:
                continue
            ts_ms = day_list.get(str(day_idx))
            if ts_ms is None:
                continue
            dt = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc).date()
            rows.append((dt.isoformat(), port_name, country, canonical_grade, float(price)))

    return rows


def store_rows(conn: sqlite3.Connection, rows: list[tuple]) -> int:
    fetched_at = datetime.now(timezone.utc).isoformat()
    inserted = 0
    for date_str, port_name, country, fuel_grade, price in rows:
        # Use port_name as port_id for S&B (no integer ID available)
        port_id = abs(hash(port_name)) % 1_000_000 + 900_000  # avoid clashing with Integr8 IDs
        conn.execute(
            """INSERT OR IGNORE INTO bunker_prices
               (date, port_id, port_name, country, fuel_grade, price_usd, fetched_at, source)
               VALUES (?, ?, ?, ?, ?, ?, ?, 'shipandbunker')""",
            (date_str, port_id, port_name, country, fuel_grade, price, fetched_at),
        )
        inserted += conn.execute("SELECT changes()").fetchone()[0]
    conn.commit()
    return inserted


def print_summary(conn: sqlite3.Connection) -> None:
    rows = conn.execute("""
        SELECT port_name,
               MIN(date) AS earliest,
               MAX(date) AS latest,
               COUNT(DISTINCT date) AS days,
               GROUP_CONCAT(DISTINCT fuel_grade) AS grades
        FROM bunker_prices
        WHERE source = 'shipandbunker'
        GROUP BY port_name
        ORDER BY port_name
    """).fetchall()

    print(f"\n{'Port':<22} {'From':<12} {'To':<12} {'Days':>5}  Grades")
    print("-" * 70)
    for port, earliest, latest, days, grades in rows:
        print(f"{port:<22} {earliest:<12} {latest:<12} {days:>5}  {grades}")


def main():
    log.info("Fetching Ship & Bunker price history for %d ports", len(PORTS))

    all_rows = []
    market_codes = list(PORTS.keys())
    api_data = {}

    # Fetch in batches — API silently drops ports beyond BATCH_SIZE
    for i in range(0, len(market_codes), BATCH_SIZE):
        batch = market_codes[i : i + BATCH_SIZE]
        try:
            result = fetch_port_batch(batch)
            api_data.update(result)
            log.info("Batch %d/%d: got %d/%d ports", i // BATCH_SIZE + 1,
                     -(-len(market_codes) // BATCH_SIZE), len(result), len(batch))
        except Exception as exc:
            log.error("Batch fetch failed: %s", exc)
            sys.exit(1)

    for code in market_codes:
        if code not in api_data:
            log.warning("No data returned for %s", code)
            continue
        rows = parse_port_data(code, api_data[code])
        log.info("  %-8s %-22s → %d price records", code, PORTS[code], len(rows))
        all_rows.extend(rows)

    log.info("Total rows parsed: %d", len(all_rows))

    with sqlite3.connect(DB_PATH) as conn:
        init_db(conn)
        n = store_rows(conn, all_rows)
        log.info("Inserted %d new records into %s", n, DB_PATH)
        print_summary(conn)


if __name__ == "__main__":
    main()
