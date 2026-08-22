#!/usr/bin/env python3
"""
Integr8 bunker price scraper.

Fetches daily HSFO, VLSFO, LSMGO prices for all ports from the Integr8
world bunker prices API and stores them in a SQLite time series database.

Run daily via cron:
    0 10 * * * /usr/bin/python3 /root/bunkervision/integr8_scraper.py

Query the Hi-5 spread (VLSFO - HSFO) per port:
    SELECT date, port_name,
           MAX(CASE WHEN fuel_grade='VLSFO' THEN price_usd END) -
           MAX(CASE WHEN fuel_grade='HSFO'  THEN price_usd END) AS hi5_spread
    FROM bunker_prices
    GROUP BY date, port_name
    HAVING hi5_spread IS NOT NULL
    ORDER BY date DESC, port_name;
"""

import logging
import sqlite3
import sys
from datetime import date, datetime, timezone
from pathlib import Path

import requests

AJAX_URL = "https://integr8fuels.com/wp-admin/admin-ajax.php"
DB_PATH = Path(__file__).parent / "bunker_prices.db"

FUEL_GRADES = {"HSFO", "VLSFO", "LSMGO", "MGO"}

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
            PRIMARY KEY (date, port_id, fuel_grade)
        )
    """)
    # Convenience view for the Hi-5 spread (VLSFO minus HSFO)
    conn.execute("""
        CREATE VIEW IF NOT EXISTS hi5_spread AS
        SELECT
            date,
            port_id,
            port_name,
            country,
            MAX(CASE WHEN fuel_grade = 'VLSFO' THEN price_usd END) AS vlsfo,
            MAX(CASE WHEN fuel_grade = 'HSFO'  THEN price_usd END) AS hsfo,
            MAX(CASE WHEN fuel_grade = 'LSMGO' THEN price_usd END) AS lsmgo,
            MAX(CASE WHEN fuel_grade = 'VLSFO' THEN price_usd END) -
            MAX(CASE WHEN fuel_grade = 'HSFO'  THEN price_usd END) AS spread
        FROM bunker_prices
        GROUP BY date, port_id
    """)
    conn.commit()


def fetch_world_prices() -> list:
    resp = requests.get(
        AJAX_URL,
        params={"action": "getWorldBunkerPrices"},
        timeout=30,
        headers={
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36",
            "Referer": "https://integr8fuels.com/bunkering-ports/bunkering-singapore/",
            "X-Requested-With": "XMLHttpRequest",
            "Accept": "application/json, text/javascript, */*; q=0.01",
        },
    )
    resp.raise_for_status()
    data = resp.json()
    if data.get("status") != "OK":
        raise ValueError(f"API non-OK: {data}")
    return data["data"]["ports"]


def store_prices(conn: sqlite3.Connection, ports: list, price_date: date) -> int:
    fetched_at = datetime.now(timezone.utc).isoformat()
    date_str = price_date.isoformat()
    inserted = 0

    for port in ports:
        info = port["port"]
        for price in port.get("prices", []):
            grade = price.get("fuelGroupName")
            if grade not in FUEL_GRADES:
                continue
            conn.execute(
                """INSERT OR IGNORE INTO bunker_prices
                   (date, port_id, port_name, country, fuel_grade, price_usd, fetched_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    date_str,
                    info["portId"],
                    info["portName"],
                    info["countryName"],
                    grade,
                    price["currentPrice"],
                    fetched_at,
                ),
            )
            inserted += conn.execute("SELECT changes()").fetchone()[0]

    conn.commit()
    return inserted


def print_summary(conn: sqlite3.Connection, for_date: date) -> None:
    rows = conn.execute(
        """SELECT port_name, country,
                  MAX(CASE WHEN fuel_grade='VLSFO' THEN price_usd END) AS vlsfo,
                  MAX(CASE WHEN fuel_grade='HSFO'  THEN price_usd END) AS hsfo,
                  MAX(CASE WHEN fuel_grade='LSMGO' THEN price_usd END) AS lsmgo
           FROM bunker_prices
           WHERE date = ?
           GROUP BY port_name, country
           ORDER BY port_name""",
        (for_date.isoformat(),),
    ).fetchall()

    print(f"\n{'Port':<22} {'Country':<18} {'VLSFO':>7} {'HSFO':>7} {'LSMGO':>7} {'Hi-5':>7}")
    print("-" * 75)
    for port, country, vlsfo, hsfo, lsmgo in rows:
        hi5 = f"{vlsfo - hsfo:>7.0f}" if vlsfo and hsfo else "     --"
        v = f"{vlsfo:>7.0f}" if vlsfo else "     --"
        h = f"{hsfo:>7.0f}" if hsfo else "     --"
        m = f"{lsmgo:>7.0f}" if lsmgo else "     --"
        print(f"{port:<22} {country:<18} {v} {h} {m} {hi5}")
    print(f"\nAll prices USD/MT  |  {for_date}")


def main():
    today = date.today()
    log.info("Fetching Integr8 world bunker prices for %s", today)

    try:
        ports = fetch_world_prices()
        log.info("Received data for %d ports", len(ports))
    except Exception as exc:
        log.error("Fetch failed: %s", exc)
        sys.exit(1)

    with sqlite3.connect(DB_PATH) as conn:
        init_db(conn)
        n = store_prices(conn, ports, today)
        log.info("Inserted %d new records into %s", n, DB_PATH)
        print_summary(conn, today)


if __name__ == "__main__":
    main()
