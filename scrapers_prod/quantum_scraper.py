#!/usr/bin/env python3
"""
QCIntel Quantum API scraper for cargo and bunker prices.

Fetches daily spot (SPO laycan) assessments for 380 CST HSFO and Marine
Fuel 0.5%S from Singapore and Rotterdam/ARA, plus selected bunker codes,
and stores them in the bunkervision SQLite database for comparison with
Integr8 and Ship & Bunker delivered-bunker prices.

Run daily after market close (Mon–Fri):
    0 18 * * 1-5 /usr/bin/python3 /root/bunkervision/quantum_scraper.py

Backfill:
    python3 quantum_scraper.py --backfill         # from 2024-01-01
    python3 quantum_scraper.py --date 2026-04-22  # single date
    python3 quantum_scraper.py --check            # show DB status
"""

import argparse
import json
import logging
import sqlite3
import time
import urllib.request
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

# Load credentials from shared env file if present
_env_path = Path("/opt/scripts/.env")
if _env_path.exists():
    import os
    for _line in _env_path.read_text().splitlines():
        _line = _line.strip()
        if "=" in _line and not _line.startswith("#"):
            _k, _, _v = _line.partition("=")
            os.environ.setdefault(_k.strip(), _v.strip())

import os

DB_PATH   = Path(__file__).parent / "bunker_prices.db"
API_BASE  = "https://www.qcintel.com/api/"
API_TOKEN = os.getenv("QCINTEL_API_TOKEN", "0P1eWf4H6uXmI4TGCdV18jTgcsJxeUhk")

# Assessment codes: prime_code → human description
# The -SPO suffix in the user codes = "SPO" laycan (spot)
ASSESSMENTS: dict[str, str] = {
    # ── Singapore ──────────────────────────────────────────────────────────────
    "MFFSIS": "Marine Fuel 0.5%S FOB Singapore",      # VLSFO cargo Singapore
    "H3FSIS": "HSFO 380 CST FOB Singapore",            # 380 CST cargo Singapore
    "MFESIC": "Marine Fuel 0.5%S Singapore Cargo",     # 0.5%S bunker/cargo SIN
    "B3ESIC": "Bunker 380 CST Singapore Cargo",        # 380 bunker SIN cargo
    # ── Rotterdam / ARA ────────────────────────────────────────────────────────
    "H3FARS": "HSFO 380 CST FOB Barge ARA",            # 380 CST cargo ARA
    "MFFARS": "Marine Fuel 0.5%S FOB Barge ARA",       # VLSFO cargo ARA
    "MDFARC": "Marine Diesel ARA Cargo",               # MDO/MGO ARA cargo
    "LSFNWS": "Low Sulfur Fuel 0.5% NW Europe",        # LS fuel NWE spot
    "LSFMDC": "Low Sulfur Fuel 0.5% Med",              # LS fuel Med spot
    # ── Other hubs ─────────────────────────────────────────────────────────────
    "MFECNC": "Marine Fuel 0.5%S NE China",
    "MFEAGC": "Marine Fuel 0.5%S East Africa / Gulf",
    "B3EAGC": "Bunker 380 CST East Africa / Gulf",
    "BHEARA": "Bunker HSFO Europe ARA",
}

LAYCAN        = "SPO"
BACKFILL_FROM = date(2024, 1, 1)
UA            = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


# ── Database ──────────────────────────────────────────────────────────────────

def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS quantum_prices (
            date         TEXT NOT NULL,
            prime_code   TEXT NOT NULL,
            name         TEXT NOT NULL,
            laycan_code  TEXT NOT NULL,
            value        REAL,
            low          REAL,
            high         REAL,
            fetched_at   TEXT NOT NULL,
            PRIMARY KEY (date, prime_code, laycan_code)
        );
        CREATE INDEX IF NOT EXISTS qp_code_date
            ON quantum_prices(prime_code, date);
    """)
    conn.commit()


# ── API ───────────────────────────────────────────────────────────────────────

def _fetch(url: str, retries: int = 3) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "*/*"})
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return r.read()
        except Exception as exc:
            if attempt == retries - 1:
                raise
            log.warning("Retry %d/%d: %s", attempt + 1, retries, exc)
            time.sleep(2 ** attempt)


def fetch_date(target: date) -> list[dict]:
    codes = ",".join(ASSESSMENTS.keys())
    url = (
        f"{API_BASE}?action=Prices&format=json&product=ENER"
        f"&date={target.isoformat()}&price_codes={codes}&token={API_TOKEN}"
    )
    return json.loads(_fetch(url))


def fetch_range_csv(from_date: date, to_date: date) -> str:
    codes = ",".join(ASSESSMENTS.keys())
    url = (
        f"{API_BASE}?action=Prices&format=csv&CSVDelimiter=%09&product=ENER"
        f"&historical=1&from_date={from_date.isoformat()}&date={to_date.isoformat()}"
        f"&price_codes={codes}&token={API_TOKEN}"
    )
    raw = _fetch(url)
    try:
        return raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        return raw.decode("latin-1")


# ── Parse & insert ────────────────────────────────────────────────────────────

def insert_json(conn: sqlite3.Connection, records: list[dict]) -> int:
    fetched_at = datetime.now(timezone.utc).isoformat()
    rows = []
    for rec in records:
        prime = rec.get("prime_code", "")
        if prime not in ASSESSMENTS:
            continue
        name = ASSESSMENTS[prime]
        for v in rec.get("values", []):
            if v.get("laycan_code") != LAYCAN:
                continue
            val = v.get("value")
            if val is None:
                continue
            rows.append((
                rec["date"], prime, name, LAYCAN,
                val, v.get("low"), v.get("high"), fetched_at,
            ))
    if rows:
        conn.executemany(
            """INSERT OR IGNORE INTO quantum_prices
               (date, prime_code, name, laycan_code, value, low, high, fetched_at)
               VALUES (?,?,?,?,?,?,?,?)""",
            rows,
        )
        conn.commit()
    return len(rows)


def insert_csv(conn: sqlite3.Connection, csv_text: str) -> int:
    import csv, io
    fetched_at = datetime.now(timezone.utc).isoformat()
    rows = []
    reader = csv.DictReader(io.StringIO(csv_text), delimiter="\t")
    for r in reader:
        prime = r.get("prime_code", "").strip()
        if prime not in ASSESSMENTS:
            continue
        if r.get("laycan_code", "").strip() != LAYCAN:
            continue
        val_str = r.get("value", "").strip()
        if not val_str:
            continue
        try:
            val = float(val_str)
        except ValueError:
            continue
        lo_s = r.get("low", "").strip()
        hi_s = r.get("high", "").strip()
        rows.append((
            r.get("date", ""), prime, ASSESSMENTS[prime], LAYCAN,
            val, float(lo_s) if lo_s else None, float(hi_s) if hi_s else None,
            fetched_at,
        ))
    if rows:
        conn.executemany(
            """INSERT OR IGNORE INTO quantum_prices
               (date, prime_code, name, laycan_code, value, low, high, fetched_at)
               VALUES (?,?,?,?,?,?,?,?)""",
            rows,
        )
        conn.commit()
    return len(rows)


# ── Sync logic ────────────────────────────────────────────────────────────────

def _trading_days(start: date, end: date):
    d = start
    while d <= end:
        if d.weekday() < 5:
            yield d
        d += timedelta(days=1)


def dates_in_db(conn: sqlite3.Connection) -> set[date]:
    rows = conn.execute("SELECT DISTINCT date FROM quantum_prices").fetchall()
    return {date.fromisoformat(r[0]) for r in rows}


def smart_sync(conn: sqlite3.Connection, force_backfill: bool = False) -> None:
    today = date.today()
    yesterday = today - timedelta(days=1)
    while yesterday.weekday() >= 5:
        yesterday -= timedelta(days=1)

    existing = dates_in_db(conn)

    if force_backfill or not existing:
        log.info("Backfill from %s to %s", BACKFILL_FROM, yesterday)
        text = fetch_range_csv(BACKFILL_FROM, yesterday)
        n = insert_csv(conn, text)
        log.info("Backfill inserted %d rows", n)
        return

    expected = set(_trading_days(BACKFILL_FROM, yesterday))
    missing  = sorted(expected - existing)

    if not missing:
        log.info("DB up to date (latest: %s)", max(existing))
        return

    if len(missing) > 10:
        log.info("Bulk pull: %s to %s (%d missing days)", missing[0], missing[-1], len(missing))
        text = fetch_range_csv(missing[0], missing[-1])
        n = insert_csv(conn, text)
        log.info("Inserted %d rows", n)
    else:
        for d in missing:
            log.info("Pulling %s …", d)
            records = fetch_date(d)
            n = insert_json(conn, records)
            log.info("  %d rows", n)


# ── Summary ───────────────────────────────────────────────────────────────────

def print_summary(conn: sqlite3.Connection) -> None:
    rows = conn.execute("""
        SELECT prime_code, name,
               MIN(date) AS first, MAX(date) AS last,
               COUNT(DISTINCT date) AS days,
               ROUND(AVG(value), 2) AS avg_val,
               ROUND(MIN(value), 2) AS min_val,
               ROUND(MAX(value), 2) AS max_val
        FROM quantum_prices
        WHERE laycan_code = ?
        GROUP BY prime_code, name
        ORDER BY name
    """, (LAYCAN,)).fetchall()

    print(f"\n{'Code':<10} {'Name':<40} {'From':<12} {'To':<12} {'Days':>5} {'Avg':>8} {'Lo':>8} {'Hi':>8}")
    print("-" * 110)
    for code, name, first, last, days, avg, lo, hi in rows:
        print(f"{code:<10} {name:<40} {first:<12} {last:<12} {days:>5} {avg:>8} {lo:>8} {hi:>8}")
    print("\nAll prices USD/MT")


def print_latest(conn: sqlite3.Connection) -> None:
    """Print the most recent available spot prices in a clean table."""
    latest_date = conn.execute("SELECT MAX(date) FROM quantum_prices").fetchone()[0]
    if not latest_date:
        print("No data in DB.")
        return

    rows = conn.execute("""
        SELECT prime_code, name, value, low, high
        FROM quantum_prices
        WHERE date = ? AND laycan_code = ?
        ORDER BY name
    """, (latest_date, LAYCAN)).fetchall()

    print(f"\nQCIntel spot prices — {latest_date}\n")
    print(f"{'Code':<10} {'Name':<40} {'Mid':>8} {'Low':>8} {'High':>8}")
    print("-" * 80)
    for code, name, val, lo, hi in rows:
        lo_s = f"{lo:>8.2f}" if lo is not None else "      --"
        hi_s = f"{hi:>8.2f}" if hi is not None else "      --"
        print(f"{code:<10} {name:<40} {val:>8.2f} {lo_s} {hi_s}")
    print("\nAll prices USD/MT")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Sync QCIntel Quantum spot prices into bunkervision DB")
    parser.add_argument("--backfill", action="store_true", help="Force full historical pull from 2024-01-01")
    parser.add_argument("--date",     help="Pull a single date (YYYY-MM-DD)")
    parser.add_argument("--check",    action="store_true", help="Show DB status only")
    args = parser.parse_args()

    with sqlite3.connect(DB_PATH) as conn:
        init_db(conn)

        if args.check:
            print_summary(conn)
            return

        if args.date:
            target = date.fromisoformat(args.date)
            log.info("Pulling %s …", target)
            records = fetch_date(target)
            n = insert_json(conn, records)
            log.info("Inserted %d rows", n)
        else:
            log.info("Smart sync (backfill=%s)", args.backfill)
            smart_sync(conn, force_backfill=args.backfill)

        print_latest(conn)


if __name__ == "__main__":
    main()
