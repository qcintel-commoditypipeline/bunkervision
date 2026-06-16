"""
Manually enter an official port-authority TOTAL for one month, then re-grade.

Use this to seed a known official figure ahead of the upstream source (e.g. MPA
is known before Stackhero publishes it). Writes the same lane the automatic
loaders write — `port_bunker_sales` (authoritative for grading) plus
`mpa_monthly` for Singapore — so when the source later publishes the same value
it is a harmless idempotent overwrite. Re-grades immediately so the dashboard
track record reflects the new actual without waiting for the daily job.

Usage:
    python set_official.py --port singapore --year 2026 --month 5 --mt 4548000
    python set_official.py --port singapore --year 2026 --month 5 --kt 4548
"""
from __future__ import annotations

import argparse
from datetime import date

import db
from accuracy import _db_port, _official_total, grade_on_refresh

_PBS_UPSERT = """
    INSERT INTO port_bunker_sales (port, month, fuel_type, volume_mt, ships_count)
    VALUES (?, ?, 'TOTAL', ?, NULL)
    ON CONFLICT (port, month, fuel_type) DO UPDATE SET
        volume_mt = excluded.volume_mt
"""
_MPA_UPSERT = """
    INSERT INTO mpa_monthly (month, fuel_type, volume_mt, ships_count)
    VALUES (?, 'TOTAL', ?, NULL)
    ON CONFLICT (month, fuel_type) DO UPDATE SET volume_mt = excluded.volume_mt
"""


def set_official(port: str, year: int, month: int, volume_mt: float) -> dict:
    month_dt = date(year, month, 1)
    before = _official_total(port, year, month)

    db.execute(_PBS_UPSERT, [_db_port(port), month_dt, float(volume_mt)])
    if port == "singapore":
        db.execute(_MPA_UPSERT, [month_dt, float(volume_mt)])

    after = _official_total(port, year, month)
    graded = grade_on_refresh(port)
    return {"before": before, "after": after, "graded": graded}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", default="singapore")
    ap.add_argument("--year", type=int, required=True)
    ap.add_argument("--month", type=int, required=True)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--mt", type=float, help="official TOTAL in tonnes")
    g.add_argument("--kt", type=float, help="official TOTAL in '000 tonnes")
    args = ap.parse_args()

    volume_mt = args.mt if args.mt is not None else args.kt * 1000
    r = set_official(args.port, args.year, args.month, volume_mt)
    print(f"{args.port} {args.year}-{args.month:02d} official TOTAL: "
          f"{r['before']} -> {r['after']:,.0f} mt  ({r['graded']} month(s) re-graded)")


if __name__ == "__main__":
    main()
