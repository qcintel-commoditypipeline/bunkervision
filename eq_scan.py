#!/usr/bin/env python3
"""
Standalone Equasis bunker tanker scanner.

Reads AIS tanker candidates (type_code 80-89) from the DB, checks each on
Equasis, and auto-adds confirmed "Bunkering Tanker" vessels to the registry.

Usage:
  python eq_scan.py                      # all ports, 10 per port
  python eq_scan.py rotterdam            # one port
  python eq_scan.py all --days 30        # extend AIS lookback window
  python eq_scan.py rotterdam --limit 5  # cap how many to check
  python eq_scan.py rotterdam --dry-run  # preview only, no DB writes

Tips:
  - Equasis has a daily page-view quota. 10 lookups/port = ~80-100 page views.
    Run one or two ports per session, not all 15 at once.
  - Each lookup is rate-limited to 10s to stay well inside Equasis limits.
  - Ctrl-C is safe — partial results are already written to DB.
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Make sure we import from this directory's venv / site-packages
sys.path.insert(0, str(Path(__file__).parent))

import db
import equasis as eq
from config import PORTS

eq._MIN_GAP = 10.0  # extra-conservative: 10s between lookups


def scan_port(port_key: str, cfg: dict, days: int, limit: int, dry_run: bool) -> None:
    bb     = cfg["bbox"]
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)

    # Use Python cutoff to avoid DuckDB NOW() / pytz issues entirely
    rows = db.query("""
        SELECT vp.mmsi, vc.imo, COALESCE(vc.name, vp.name) AS name,
               vc.type_code, vp.sog, vp.lat, vp.lon
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
        LIMIT ?
    """, [bb[0][0], bb[1][0], bb[0][1], bb[1][1], cutoff, limit])

    port_label = cfg.get("name", port_key)
    print(f"\n{'='*60}")
    print(f"  {port_label}  ({port_key})  —  {len(rows)} candidate(s) to check")
    print(f"{'='*60}")

    if not rows:
        print("  Nothing to scan. Try --days 30 to widen the AIS window.")
        return

    added  = 0
    today  = datetime.now(timezone.utc).date()

    for i, (mmsi, imo, name, type_code, sog, lat, lon) in enumerate(rows, 1):
        print(f"\n[{i}/{len(rows)}]  MMSI={mmsi}  IMO={imo}  AIS name={name or '—'}  SOG={sog}")

        info = eq.lookup_imo(str(imo))

        if not info:
            # Possible causes: session expired, rate-limited, vessel not on Equasis
            # equasis.py already logs the detail; just note it here
            print("  Equasis: no result (session issue or IMO not found)")
            continue

        ship_type = info.get("ship_type", "")
        flag      = info.get("flag", "")
        gt        = info.get("gt")
        cl        = eq.classify_ship_type(ship_type)
        disp_name = info.get("name") or name or mmsi

        star = "★ BUNKERING TANKER" if cl["level"] == "confirmed" else f"  {cl['label'] or ship_type or 'unknown'}"
        print(f"  {star}  |  {disp_name}  |  flag={flag}  |  {gt} GT  |  confidence={cl['level']}")

        if cl["level"] == "confirmed":
            if dry_run:
                print("  [DRY RUN] would add to registry")
            else:
                db.execute("""
                    INSERT INTO bunker_vessels (imo, mmsi, name, port, licensee, last_updated)
                    VALUES (?, ?, ?, ?, ?, ?)
                    ON CONFLICT (imo) DO UPDATE SET
                        mmsi         = excluded.mmsi,
                        name         = COALESCE(excluded.name, bunker_vessels.name),
                        port         = excluded.port,
                        last_updated = excluded.last_updated
                """, [str(imo), mmsi, disp_name, port_key, "Equasis-AUTO", today])
                print(f"  → Added to {port_key} registry")
                added += 1

    print(f"\n{port_label}: checked {len(rows)}, added {added}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Equasis bunker tanker scan — check AIS tanker candidates against Equasis"
    )
    parser.add_argument(
        "port", nargs="?", default="all",
        help=f"Port key or 'all'. Valid ports: {', '.join(PORTS.keys())}"
    )
    parser.add_argument("--days",    type=int, default=14, help="AIS lookback window in days (default 14)")
    parser.add_argument("--limit",   type=int, default=10, help="Max vessels to check per port (default 10)")
    parser.add_argument("--dry-run", action="store_true",  help="Preview only — no DB writes")
    args = parser.parse_args()

    port_keys = list(PORTS.keys()) if args.port == "all" else [args.port]

    unknown = [p for p in port_keys if p not in PORTS]
    if unknown:
        print(f"Unknown port(s): {unknown}")
        print(f"Valid ports: {', '.join(PORTS.keys())}")
        sys.exit(1)

    if args.dry_run:
        print("[DRY RUN MODE — no DB writes]")

    print(f"Settings: days={args.days}  limit={args.limit}/port  rate-limit=10s")

    # Test Equasis login once upfront before burning page views
    print("\nTesting Equasis login... ", end="", flush=True)
    session = eq._get_session()
    if session is None:
        print("FAILED — check EQUASIS_EMAIL/PASSWORD in .env and that account is active")
        sys.exit(1)
    print("OK")

    for port_key in port_keys:
        scan_port(port_key, PORTS[port_key], args.days, args.limit, args.dry_run)

    print("\nDone.")


if __name__ == "__main__":
    main()
