"""/healthz liveness gate — every transition with a fake clock, no sleeping.

The incident this fences (2026-08-19..22): aisstream went silent for days and the
service stayed `active` / HTTP 200 while receiving zero messages.
Run: python -m pytest -q tests
"""
from __future__ import annotations

import os
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import duckdb  # noqa: E402
import pytest  # noqa: E402

import ais_client  # noqa: E402
import db  # noqa: E402
import liveness  # noqa: E402

T0 = 1_700_000_000.0          # arbitrary fixed epoch
MIN = 60.0
OK_DB = {"ok": True, "via": "conn", "error": None}
BAD_DB = {"ok": False, "via": "conn", "error": "FatalException: database has been invalidated"}


def _alive_thread():
    ev = threading.Event()
    t = threading.Thread(target=ev.wait, daemon=True, name="fake-ais-client")
    t.start()
    return t, ev


@pytest.fixture
def ingest(monkeypatch):
    """Fresh ais_client counters with the ingest 'enabled' and a live thread."""
    ais_client._reset_for_tests()
    monkeypatch.setattr(ais_client, "AISSTREAM_API_KEY", "test-key")
    t, ev = _alive_thread()
    ais_client._client_thread = t
    with ais_client._stats_lock:
        ais_client._stats["started_epoch"] = T0
    yield {"client": t, "client_stop": ev}
    ev.set()
    ais_client._reset_for_tests()


def _msg(now, n=1):
    with ais_client._stats_lock:
        ais_client._stats["received"] += n
        ais_client._stats["last_msg_epoch"] = now
    ais_client._window.add("received", n=n, now=now)


def _snap(now, **over):
    s = ais_client.ingest_snapshot(now=now)
    s.update(over)
    return s


# ── WindowCounter ─────────────────────────────────────────────────────────────

def test_window_counter_counts_only_trailing_window():
    w = liveness.WindowCounter()
    w.add("received", 5, now=T0)
    w.add("received", 3, now=T0 + 10 * MIN)
    assert w.total("received", 30 * MIN, now=T0 + 10 * MIN) == 8
    assert w.total("received", 30 * MIN, now=T0 + 31 * MIN) == 3
    assert w.total("received", 30 * MIN, now=T0 + 45 * MIN) == 0


# ── evaluate(): the pure verdict ─────────────────────────────────────────────

def test_fresh_messages_are_ok(ingest):
    _msg(T0 + 5 * MIN, n=40)
    body = liveness.evaluate(_snap(T0 + 6 * MIN), OK_DB, now=T0 + 6 * MIN, stale_min=30)
    assert body["ok"] is True and body["status"] == "ok" and body["reasons"] == []
    assert body["received_last_30m"] == 40 and body["write_errors_last_30m"] == 0
    assert body["db_ok"] is True and body["silence_seconds"] == 60
    assert "flush_thread_alive" not in body          # no flusher in this design


def test_31_minutes_of_silence_is_degraded(ingest):
    _msg(T0 + 1 * MIN, n=10)
    now = T0 + 32 * MIN
    body = liveness.evaluate(_snap(now), OK_DB, now=now, stale_min=30)
    assert body["ok"] is False and body["status"] == "degraded"
    assert any(r.startswith("no_ais_message_for_31m") for r in body["reasons"]), body["reasons"]
    assert body["received_last_30m"] == 0
    assert body["silence_seconds"] == 31 * 60


def test_29_minutes_of_silence_is_still_ok(ingest):
    _msg(T0 + 1 * MIN)
    now = T0 + 30 * MIN
    assert liveness.evaluate(_snap(now), OK_DB, now=now, stale_min=30)["ok"] is True


def test_startup_grace_uses_thread_start_not_first_message(ingest):
    assert liveness.evaluate(_snap(T0 + 5 * MIN), OK_DB, now=T0 + 5 * MIN, stale_min=30)["ok"]
    late = liveness.evaluate(_snap(T0 + 31 * MIN), OK_DB, now=T0 + 31 * MIN, stale_min=30)
    assert not late["ok"] and late["last_message_at"] is None


def test_write_errors_in_window_are_degraded(ingest):
    now = time.time()
    _msg(now, n=100)                                    # feed flowing …
    ais_client._note_write_error("vessel_positions", RuntimeError("database has been invalidated"))
    body = liveness.evaluate(_snap(now), OK_DB, now=now, stale_min=30)
    assert body["ok"] is False
    assert "write_errors_last_30m=1" in body["reasons"]
    assert ais_client.ingest_stats()["write_errors"] == 1


def test_write_errors_age_out_and_gate_recovers(ingest):
    ais_client._window.add("write_errors", now=T0)
    _msg(T0 + 40 * MIN)
    body = liveness.evaluate(_snap(T0 + 40 * MIN), OK_DB, now=T0 + 40 * MIN, stale_min=30)
    assert body["ok"] is True and body["write_errors_last_30m"] == 0


def test_db_invalidated_is_degraded_even_with_live_feed(ingest):
    _msg(T0 + 1)
    body = liveness.evaluate(_snap(T0 + 2), BAD_DB, now=T0 + 2, stale_min=30)
    assert body["ok"] is False and body["db_ok"] is False
    assert any(r.startswith("db_error: FatalException: database has been invalidated")
               for r in body["reasons"])


def test_missing_ingest_thread_is_degraded_not_ok(ingest):
    _msg(T0 + 1)
    ais_client._client_thread = None
    body = liveness.evaluate(_snap(T0 + 2), OK_DB, now=T0 + 2, stale_min=30)
    assert body["ok"] is False
    assert any(r.startswith("ingest_thread_missing") for r in body["reasons"])


def test_dead_ingest_thread_is_degraded(ingest):
    ingest["client_stop"].set()
    ingest["client"].join(timeout=2)
    _msg(T0 + 1)
    body = liveness.evaluate(_snap(T0 + 2), OK_DB, now=T0 + 2, stale_min=30)
    assert body["ok"] is False and body["ingest_thread_alive"] is False


def test_disabled_ingest_is_degraded(monkeypatch):
    ais_client._reset_for_tests()
    monkeypatch.setattr(ais_client, "AISSTREAM_API_KEY", "")
    body = liveness.evaluate(ais_client.ingest_snapshot(now=T0), OK_DB, now=T0, stale_min=30)
    assert body["ok"] is False and body["ingest_enabled"] is False
    assert any("ingest disabled" in r for r in body["reasons"])


# ── threshold is a setting ───────────────────────────────────────────────────

def test_threshold_from_env(monkeypatch, ingest):
    monkeypatch.setenv(liveness.STALE_MINUTES_ENV, "45")
    assert liveness.STALE_MINUTES_ENV == "BUNKERVISION_INGEST_STALE_MINUTES"
    assert liveness.stale_minutes() == 45
    _msg(T0)
    now = T0 + 40 * MIN
    body = liveness.evaluate(_snap(now), OK_DB, now=now)
    assert body["ok"] is True and body["stale_after_minutes"] == 45
    assert "received_last_45m" in body
    monkeypatch.setenv(liveness.STALE_MINUTES_ENV, "garbage")
    assert liveness.stale_minutes() == liveness.DEFAULT_STALE_MINUTES == 30


# ── HealthGate: transition log ───────────────────────────────────────────────

def test_gate_logs_warning_on_each_transition(ingest):
    warnings: list[str] = []
    gate = liveness.HealthGate("bunkervision", log_fn=warnings.append)
    _msg(T0)
    assert gate.check(_snap(T0 + 1), OK_DB, now=T0 + 1, stale_min=30)["status"] == "ok"
    assert warnings == [] and gate.transitions == 0

    r = gate.check(_snap(T0 + 35 * MIN), OK_DB, now=T0 + 35 * MIN, stale_min=30)
    assert r["status"] == "degraded" and gate.transitions == 1
    assert len(warnings) == 1 and "ok -> DEGRADED" in warnings[0]
    assert "no_ais_message_for_35m" in warnings[0]

    gate.check(_snap(T0 + 36 * MIN), OK_DB, now=T0 + 36 * MIN, stale_min=30)
    assert len(warnings) == 1                            # no spam while still degraded

    _msg(T0 + 37 * MIN, n=5)
    r = gate.check(_snap(T0 + 37 * MIN), OK_DB, now=T0 + 37 * MIN, stale_min=30)
    assert r["status"] == "ok" and gate.transitions == 2
    assert len(warnings) == 2 and "degraded -> ok (recovered)" in warnings[1]


def test_gate_first_verdict_degraded_is_warned():
    ais_client._reset_for_tests()
    warnings: list[str] = []
    gate = liveness.HealthGate("bunkervision", log_fn=warnings.append)
    r = gate.check(ais_client.ingest_snapshot(now=T0), OK_DB, now=T0, stale_min=30)
    assert r["status"] == "degraded"
    assert warnings and "start -> DEGRADED" in warnings[0]


def test_module_gate_routes_warning_to_loguru():
    from loguru import logger
    seen: list[str] = []
    sink_id = logger.add(lambda m: seen.append(str(m)), level="WARNING")
    try:
        ais_client._reset_for_tests()
        g = liveness.HealthGate("bunkervision", log_fn=liveness._loguru_warning)
        g.check(ais_client.ingest_snapshot(now=T0), OK_DB, now=T0, stale_min=30)
    finally:
        logger.remove(sink_id)
    assert any("DEGRADED" in s for s in seen)


# ── db.ping(): the probe actually exercises the connection ───────────────────

def test_db_ping_ok_on_live_connection():
    old = db._conn
    try:
        db._conn = duckdb.connect(":memory:")
        assert db.ping() == {"ok": True, "via": "conn", "error": None}
    finally:
        db._conn = old


def test_db_ping_fails_on_unusable_connection():
    old = db._conn
    try:
        c = duckdb.connect(":memory:")
        c.close()                       # closed ≈ invalidated: any execute raises
        db._conn = c
        p = db.ping()
        assert p["ok"] is False and p["error"]
    finally:
        db._conn = old


def test_db_ping_does_not_queue_behind_a_held_lock():
    old = db._conn
    try:
        db._conn = duckdb.connect(":memory:")
        held, release = threading.Event(), threading.Event()

        def holder():
            with db._lock:
                held.set()
                release.wait(10)
        th = threading.Thread(target=holder, daemon=True)
        th.start()
        assert held.wait(5)
        try:
            t = time.time()
            p = db.ping(lock_timeout=0.2)
            assert time.time() - t < 5
            assert p["ok"] is True and p["via"] == "cursor"
        finally:
            release.set()
            th.join(5)
    finally:
        db._conn = old


# ── the Flask route ──────────────────────────────────────────────────────────

@pytest.fixture
def client():
    old = db._conn
    db._conn = duckdb.connect(":memory:")
    import app as app_module
    app_module.app.config["TESTING"] = True
    with app_module.app.test_client() as c:
        yield c
    db._conn = old


def test_healthz_503_when_ingest_absent(client):
    ais_client._reset_for_tests()
    r = client.get("/healthz")
    assert r.status_code == 503
    j = r.get_json()
    assert j["ok"] is False and j["status"] == "degraded" and j["project"] == "bunkervision"
    assert j["reasons"] and j["db_ok"] is True


def test_healthz_200_when_ingest_live_and_verbose_view(client, ingest):
    now = time.time()
    _msg(now, n=3)
    r = client.get("/healthz")
    assert r.status_code == 200, r.get_json()
    j = r.get_json()
    assert j["ok"] is True and j["status"] == "ok" and j["received_last_30m"] == 3
    assert "ingest" not in j
    v = client.get("/healthz?verbose=1").get_json()
    assert v["ingest"]["received"] == 3 and v["ingest"]["enabled"] is True
    assert "threads" in v and v["gate"]["stale_minutes_env"] == "BUNKERVISION_INGEST_STALE_MINUTES"


def test_healthz_503_on_db_error_and_on_write_errors(client, ingest, monkeypatch):
    now = time.time()
    _msg(now)
    monkeypatch.setattr(db, "ping", lambda *a, **k: dict(BAD_DB))
    r = client.get("/healthz")
    assert r.status_code == 503 and any(x.startswith("db_error") for x in r.get_json()["reasons"])
    monkeypatch.setattr(db, "ping", lambda *a, **k: dict(OK_DB))
    assert client.get("/healthz").status_code == 200
    ais_client._note_write_error("ais_positions", RuntimeError("boom"))
    r = client.get("/healthz")
    assert r.status_code == 503 and "write_errors_last_30m=1" in r.get_json()["reasons"]


def test_error_frame_is_logged_not_counted_as_message(ingest):
    from loguru import logger
    seen: list[str] = []
    sink_id = logger.add(lambda m: seen.append(str(m)), level="WARNING")
    try:
        before = ais_client.ingest_stats()["received"]
        ais_client._on_message(None, '{"error": "Api Key Is Not Valid"}')
        assert ais_client.ingest_stats()["received"] == before
        assert any("ERROR frame" in s and "Api Key Is Not Valid" in s for s in seen)
    finally:
        logger.remove(sink_id)
