#!/usr/bin/env python3
"""
eq_build.py — two faster ways to build bunker vessel registries.

MODE 1: fleet  (operator/company name search)
  Search Equasis by fleet/operator name. One page view returns 20-50 vessels
  with ship types already included — no per-IMO lookups needed for filtering.
  Best for known operators whose vessels share a name prefix.

  eq_build.py fleet panama "Monjasa"
  eq_build.py fleet rotterdam "Veder"
  eq_build.py fleet rotterdam "Koole"
  eq_build.py fleet singapore "Beng Kuang"

MODE 2: slow  (any vessel type, stopped/slow in bbox)
  Finds any vessel seen with SOG < 1kt in the port bounding box, looks each up
  on Equasis. Catches independent operators regardless of AIS type code.
  Uses one Equasis page view per candidate.

  eq_build.py slow panama
  eq_build.py slow fujairah --days 30 --limit 15

Flags:
  --dry-run    Preview only — no DB writes
  --days N     AIS lookback window for slow mode (default 14)
  --limit N    Max candidates to check in slow mode (default 15)

Tips:
  - fleet mode: ~1-2 page views per search term, very quota-friendly
  - slow mode: ~1 page view per candidate, use --limit to stay safe
  - Ctrl-C is safe — partial results already written
  - Do 2-3 runs per day max across both modes combined
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import db
import equasis as eq
from config import PORTS

eq._MIN_GAP = 10.0


def _add_vessel(imo: str, mmsi: str | None, name: str, port_key: str) -> None:
    today = datetime.now(timezone.utc).date()
    db.execute("""
        INSERT INTO bunker_vessels (imo, mmsi, name, port, licensee, last_updated)
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT (imo) DO UPDATE SET
            mmsi         = COALESCE(excluded.mmsi, bunker_vessels.mmsi),
            name         = COALESCE(excluded.name, bunker_vessels.name),
            port         = excluded.port,
            last_updated = excluded.last_updated
    """, [str(imo), mmsi, name, port_key, "Equasis-AUTO", today])


def _already_in_registry(imo: str, mmsi: str | None) -> bool:
    rows = db.query(
        "SELECT 1 FROM bunker_vessels WHERE imo = ? OR (mmsi IS NOT NULL AND mmsi = ?) LIMIT 1",
        [str(imo), mmsi]
    )
    return bool(rows)


# ── MODE 1: fleet name search ─────────────────────────────────────────────────

def run_fleet(port_key: str, term: str, dry_run: bool) -> None:
    port_label = PORTS[port_key].get("name", port_key)
    print(f"\n{'='*60}")
    print(f"  Fleet search: {term!r}  →  {port_label}")
    print(f"{'='*60}")

    results = eq.search(term)
    if not results:
        print("  No results — session issue or term not found on Equasis")
        return

    print(f"  {len(results)} vessel(s) returned by Equasis search")

    added = skipped = other = 0
    for r in results:
        imo       = r.get("imo", "")
        name      = r.get("name", "")
        ship_type = r.get("ship_type", "")
        flag      = r.get("flag", "")
        gt        = r.get("gt")
        cl        = eq.classify_ship_type(ship_type)

        if not imo:
            continue

        status = "★ CONFIRMED" if cl["level"] == "confirmed" else f"  {cl['label'] or ship_type or '—'}"
        print(f"  {status:<22}  IMO={imo}  {name:<28}  {flag}  {gt or '?'} GT")

        if cl["level"] != "confirmed":
            other += 1
            continue

        if _already_in_registry(imo, None):
            print(f"    → already in registry, skipping")
            skipped += 1
            continue

        if dry_run:
            print(f"    [DRY RUN] would add to {port_key}")
        else:
            _add_vessel(imo, None, name, port_key)
            print(f"    → Added to {port_key} registry")
        added += 1

    print(f"\n  Result: {added} added, {skipped} already present, {other} not bunkering tankers")


# ── MODE 2: slow-vessel AIS scan ──────────────────────────────────────────────

def run_slow(port_key: str, days: int, limit: int, dry_run: bool) -> None:
    cfg        = PORTS[port_key]
    bb         = cfg["bbox"]
    port_label = cfg.get("name", port_key)
    cutoff     = datetime.now(timezone.utc) - timedelta(days=days)

    # Rank by how many times a vessel has been seen slow in the bbox —
    # bunker barges are port regulars; transiting ships appear once or twice.
    rows = db.query("""
        SELECT vp.mmsi, vc.imo, COALESCE(vc.name, vp.name) AS name,
               vc.type_code, COUNT(*) AS appearances
        FROM vessel_positions vp
        JOIN vessel_cache vc ON vc.mmsi = vp.mmsi
        WHERE vp.sog < 1.0
          AND vp.lat BETWEEN ? AND ?
          AND vp.lon BETWEEN ? AND ?
          AND vp.ts  >= ?
          AND vc.imo IS NOT NULL AND vc.imo != ''
          AND NOT EXISTS (
              SELECT 1 FROM bunker_vessels bv
              WHERE bv.mmsi = vp.mmsi OR bv.imo = vc.imo
          )
        GROUP BY vp.mmsi, vc.imo, vc.name, vp.name, vc.type_code
        ORDER BY appearances DESC
        LIMIT ?
    """, [bb[0][0], bb[1][0], bb[0][1], bb[1][1], cutoff, limit])

    print(f"\n{'='*60}")
    print(f"  Slow-vessel scan  →  {port_label}  ({len(rows)} candidate(s))")
    print(f"{'='*60}")

    if not rows:
        print("  No slow/stopped vessels found. Try --days 30 to widen window.")
        return

    added = 0
    for i, (mmsi, imo, name, type_code, appearances) in enumerate(rows, 1):
        print(f"\n[{i}/{len(rows)}]  MMSI={mmsi}  IMO={imo}  AIS={name or '—'}  type={type_code}  seen={appearances}x")

        info = eq.lookup_imo(str(imo))
        if not info:
            print("  Equasis: no result")
            continue

        ship_type = info.get("ship_type", "")
        flag      = info.get("flag", "")
        gt        = info.get("gt")
        cl        = eq.classify_ship_type(ship_type)
        disp_name = info.get("name") or name or str(mmsi)

        star = "★ BUNKERING TANKER" if cl["level"] == "confirmed" else f"  {cl['label'] or ship_type or '—'}"
        print(f"  {star}  |  {disp_name}  |  flag={flag}  |  {gt} GT")

        if cl["level"] == "confirmed":
            if dry_run:
                print("  [DRY RUN] would add to registry")
            else:
                _add_vessel(str(imo), mmsi, disp_name, port_key)
                print(f"  → Added to {port_key} registry")
                added += 1

    print(f"\n{port_label}: checked {len(rows)}, added {added}")


# ── MODE 3: global operator list ─────────────────────────────────────────────

# Known major bunker operators — searched globally, vessels auto-assigned to
# whichever port bboxes they've actually been seen in via AIS.
KNOWN_OPERATORS = [
    "Bunker Holdings",
    "Dan-Bunkering",
    "Peninsula Petroleum",
    "Minerva Bunkering",
    "World Fuel Services",
    "Vitol Bunkers",
    "TFG Marine",
    "Monjasa",
    "Integr8",
    "Fratelli Cosulich",
]


def _ports_for_imo(imo: str) -> list[str]:
    """Return port keys where this IMO has been seen in a port bbox via AIS."""
    matched = []
    rows = db.query(
        "SELECT lat, lon FROM vessel_positions vp JOIN vessel_cache vc ON vc.mmsi = vp.mmsi WHERE vc.imo = ? LIMIT 500",
        [str(imo)]
    )
    if not rows:
        return []
    for port_key, cfg in PORTS.items():
        bb = cfg["bbox"]
        min_lat, min_lon = bb[0]
        max_lat, max_lon = bb[1]
        for lat, lon in rows:
            if min_lat <= lat <= max_lat and min_lon <= lon <= max_lon:
                matched.append(port_key)
                break
    return matched


def run_operators(operators: list[str], dry_run: bool) -> None:
    print(f"\nSearching {len(operators)} operator(s) on Equasis")
    print("Vessels confirmed as Bunkering Tanker will be auto-assigned to ports")
    print("where they've been seen in AIS bounding boxes.\n")

    total_added = 0

    for term in operators:
        print(f"\n{'─'*60}")
        print(f"  Operator: {term!r}")
        print(f"{'─'*60}")

        results = eq.search(term)
        if not results:
            print("  No results or session error — skipping")
            continue

        confirmed = [r for r in results if eq.classify_ship_type(r.get("ship_type", ""))["level"] == "confirmed"]
        print(f"  {len(results)} vessels returned, {len(confirmed)} confirmed Bunkering Tanker")

        for r in confirmed:
            imo  = r.get("imo", "")
            name = r.get("name", "")
            flag = r.get("flag", "")
            gt   = r.get("gt")

            if not imo:
                continue

            if _already_in_registry(imo, None):
                print(f"  {name} (IMO {imo}) — already in registry")
                continue

            ports_seen = _ports_for_imo(imo)
            if not ports_seen:
                print(f"  {name} (IMO {imo}) — no AIS sightings in any configured port bbox, skipping")
                continue

            print(f"  ★ {name:<30}  IMO={imo}  {flag}  {gt or '?'} GT  →  ports: {', '.join(ports_seen)}")

            if dry_run:
                print(f"    [DRY RUN] would add to: {', '.join(ports_seen)}")
            else:
                for port_key in ports_seen:
                    _add_vessel(imo, None, name, port_key)
                print(f"    → Added to {', '.join(ports_seen)}")
                total_added += 1

    print(f"\n{'='*60}")
    print(f"  Done — {total_added} vessel(s) added across all operators")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build bunker vessel registries via Equasis — fleet search, slow-vessel scan, or global operator list"
    )
    sub = parser.add_subparsers(dest="mode", required=True)

    # fleet subcommand
    fp = sub.add_parser("fleet", help="Search by operator/fleet name, assign to a specific port")
    fp.add_argument("port",  help=f"Port key. Valid: {', '.join(PORTS.keys())}")
    fp.add_argument("term",  help="Operator or fleet name to search on Equasis")
    fp.add_argument("--dry-run", action="store_true")

    # slow subcommand
    sp = sub.add_parser("slow", help="Find stopped/slow vessels in bbox, look up on Equasis")
    sp.add_argument("port",    help=f"Port key. Valid: {', '.join(PORTS.keys())}")
    sp.add_argument("--days",  type=int, default=14)
    sp.add_argument("--limit", type=int, default=15)
    sp.add_argument("--dry-run", action="store_true")

    # operators subcommand
    op = sub.add_parser("operators", help="Search known major operators, auto-assign vessels to ports by AIS sightings")
    op.add_argument("terms", nargs="*",
                    help="Operator name(s) to search. Defaults to built-in list if omitted.")
    op.add_argument("--dry-run", action="store_true")

    args = parser.parse_args()

    if args.mode in ("fleet", "slow") and args.port not in PORTS:
        print(f"Unknown port: {args.port!r}")
        print(f"Valid ports: {', '.join(PORTS.keys())}")
        sys.exit(1)

    if args.dry_run:
        print("[DRY RUN — no DB writes]")

    print("Testing Equasis login... ", end="", flush=True)
    if eq._get_session() is None:
        print("FAILED — check EQUASIS_EMAIL/PASSWORD in .env")
        sys.exit(1)
    print("OK")

    if args.mode == "fleet":
        run_fleet(args.port, args.term, args.dry_run)
    elif args.mode == "slow":
        run_slow(args.port, args.days, args.limit, args.dry_run)
    else:
        terms = args.terms if args.terms else KNOWN_OPERATORS
        run_operators(terms, args.dry_run)

    print("\nDone.")


if __name__ == "__main__":
    main()
