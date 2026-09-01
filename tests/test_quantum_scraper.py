import sqlite3
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scrapers_prod import quantum_scraper as q


def test_confirmed_no_data_dates_are_part_of_sync_coverage():
    conn = sqlite3.connect(":memory:")
    q.init_db(conn)

    assert q.mark_confirmed_no_data(conn, {date(2026, 1, 1)}) == 1
    assert date(2026, 1, 1) in q.dates_in_db(conn)


def test_confirmed_no_data_date_is_idempotent():
    conn = sqlite3.connect(":memory:")
    q.init_db(conn)

    q.mark_confirmed_no_data(conn, {date(2026, 1, 1)})
    q.mark_confirmed_no_data(conn, {date(2026, 1, 1)})

    assert conn.execute("SELECT count(*) FROM quantum_no_data_dates").fetchone()[0] == 1


def test_fetch_refuses_missing_token(monkeypatch):
    monkeypatch.setattr(q, "API_TOKEN", None)

    try:
        q.fetch_date(date(2026, 9, 1))
    except RuntimeError as exc:
        assert "QCINTEL_API_TOKEN" in str(exc)
    else:
        raise AssertionError("missing API token was accepted")
