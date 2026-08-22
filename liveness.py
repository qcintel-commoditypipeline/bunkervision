"""Ingest liveness gate for /healthz.

Why this exists (2026-08-19 → 08-22): aisstream.io went silent on 08-19 06:18 UTC
(provider outage, aisstream/issues #279/#282) and BunkerVision sat for days with
`systemctl is-active` = active and HTTP 200 while receiving ZERO messages — the
classic silent failure: a process that is up but not doing its job, reporting the
first and not the second. (Sibling WAKE additionally sat on an invalidated DuckDB
with 9,391 write errors behind a 200.)

The gate turns "healthy" into a claim that has to be EARNED every time it is asked:

  degraded (HTTP 503) when ANY of
    * the ingest thread is missing (disabled: no key, or the thread died)
    * no AIS message for > BUNKERVISION_INGEST_STALE_MINUTES (default 30; the feed
      is beta / no-SLA, so it is a setting, not a constant)
    * any DB write error inside that same window (the per-message writes in
      ais_client used to be `except Exception: logger.debug(...)` — invisible)
    * the DB connection is invalidated / a trivial SELECT fails
  ok (HTTP 200) otherwise.

Silence is measured from the LATER of (last message, ingest-thread start) so a
fresh restart is "ok" until the threshold elapses, but it is NOT reset on
reconnects: a client that reconnects forever and receives nothing (revoked key,
second client on the key, provider down) goes degraded exactly like a dropped
socket.

Operator reaction to a 503 — this is a SIGNAL, not a restart trigger:
  * systemd does NOT restart on 503 (Restart=always fires only on process exit;
    there is no WatchdogSec / ExecStartPost probe). Deliberately so: a provider
    outage is not fixed by restarting, and reconnect storms hurt the per-key
    throttle. Read `/healthz?verbose=1` (or the journal WARNING on the transition)
    and act on the reason:
      - no_ais_message_for_*        → check aisstream status / the key / a second
                                       client on the same key; nothing to restart
      - write_errors_*  / db_error  → the DuckDB file is unusable (invalidated,
                                       locked, disk full): `systemctl restart
                                       bunkervision` re-opens it; journal has the
                                       first error
      - ingest_thread_missing       → AISSTREAM_API_KEY empty in .env, or the
                                       thread crashed (journal has the traceback)

Design: counters live in ais_client (hot path); this module owns the windowed
buckets, the threshold, the verdict and the transition log. Everything takes an
injectable clock (`now`) so tests drive the state machine without sleeping.
(Same module as wake/liveness.py with the project name/env prefix changed.)
"""
from __future__ import annotations

import logging
import os
import threading
import time
from datetime import datetime, timezone

__all__ = ["WindowCounter", "HealthGate", "evaluate", "stale_minutes", "gate"]

_log = logging.getLogger("bunkervision.health")

STALE_MINUTES_ENV = "BUNKERVISION_INGEST_STALE_MINUTES"
DEFAULT_STALE_MINUTES = 30


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name, "")
    try:
        v = int(raw)
    except (TypeError, ValueError):
        return default
    return v if v > 0 else default


def stale_minutes() -> int:
    """Silence threshold / counter window, minutes. Read at call time (not import)
    so an .env change or a test monkeypatch takes effect without a restart."""
    return _env_int(STALE_MINUTES_ENV, DEFAULT_STALE_MINUTES)


def _iso(epoch: float | None) -> str | None:
    if epoch is None:
        return None
    return datetime.fromtimestamp(epoch, tz=timezone.utc).isoformat(timespec="seconds")


class WindowCounter:
    """Per-minute event buckets — `add` on the hot path is a dict increment under a
    lock; `total(key, window_s)` sums the buckets that overlap the trailing window
    and prunes the rest. Bucket granularity is one minute, so a window of W seconds
    may include up to 59 s extra — fine for a liveness threshold of tens of minutes."""

    def __init__(self) -> None:
        self._buckets: dict[int, dict[str, int]] = {}
        self._lock = threading.Lock()

    def add(self, key: str, n: int = 1, now: float | None = None) -> None:
        minute = int((time.time() if now is None else now) // 60)
        with self._lock:
            d = self._buckets.setdefault(minute, {})
            d[key] = d.get(key, 0) + n

    def total(self, key: str, window_s: float, now: float | None = None) -> int:
        now = time.time() if now is None else now
        first = int((now - window_s) // 60)
        with self._lock:
            stale = [m for m in self._buckets if m < first]
            for m in stale:
                del self._buckets[m]
            return sum(d.get(key, 0) for m, d in self._buckets.items() if m >= first)

    def reset(self) -> None:
        with self._lock:
            self._buckets.clear()


def evaluate(snapshot: dict, db_probe: dict, now: float | None = None,
             stale_min: int | None = None) -> dict:
    """Pure verdict. `snapshot` comes from ais_client.ingest_snapshot(), `db_probe`
    from db.ping(). Returns the /healthz body (minus project-specific extras)."""
    now = time.time() if now is None else now
    w = stale_minutes() if stale_min is None else int(stale_min)
    window_s = w * 60
    reasons: list[str] = []

    enabled = bool(snapshot.get("enabled"))
    thread_alive = snapshot.get("thread_alive")
    flusher_alive = snapshot.get("flusher_alive")  # None = n/a (no flusher design)
    if not enabled:
        reasons.append("ingest_thread_missing: ingest disabled (no AISSTREAM_API_KEY)")
    elif not thread_alive:
        reasons.append("ingest_thread_missing: ais-client thread is not alive")
    elif flusher_alive is False:
        reasons.append("flush_thread_dead: positions buffer in RAM, nothing reaches the DB")

    last_msg = snapshot.get("last_msg_epoch")
    started = snapshot.get("started_epoch")
    refs = [t for t in (last_msg, started) if t is not None]
    silence_s = (now - max(refs)) if refs else None
    if silence_s is not None and silence_s > window_s:
        reasons.append(f"no_ais_message_for_{int(silence_s // 60)}m (threshold {w}m)")
    elif silence_s is None and not reasons:
        reasons.append("ingest_not_started: no start time and no message ever")

    recv_w = int(snapshot.get("received_window", 0) or 0)
    werr_w = int(snapshot.get("write_errors_window", 0) or 0)
    if werr_w > 0:
        reasons.append(f"write_errors_last_{w}m={werr_w}")

    db_ok = bool(db_probe.get("ok"))
    if not db_ok:
        reasons.append(f"db_error: {db_probe.get('error') or 'probe failed'}")

    sched_error = snapshot.get("sched_error")
    if sched_error:
        reasons.append(f"scheduler_error: {sched_error}")

    status = "ok" if not reasons else "degraded"
    body = {
        "ok": status == "ok",
        "status": status,
        "reasons": reasons,
        "last_message_at": _iso(last_msg),
        "silence_seconds": None if silence_s is None else int(silence_s),
        f"received_last_{w}m": recv_w,
        f"write_errors_last_{w}m": werr_w,
        "window_minutes": w,
        "stale_after_minutes": w,
        "db_ok": db_ok,
        "db_error": db_probe.get("error"),
        "db_probe_via": db_probe.get("via"),
        "ingest_enabled": enabled,
        "ingest_thread_alive": thread_alive,
        "checked_at": _iso(now),
    }
    if flusher_alive is not None:
        body["flush_thread_alive"] = flusher_alive
    return body


class HealthGate:
    """Holds the last verdict and logs a WARNING on every ok↔degraded transition
    (and on the first degraded verdict after start), so the journal shows WHEN
    the feed died / recovered without anyone tailing the counters."""

    def __init__(self, name: str = "bunkervision", log_fn=None) -> None:
        self.name = name
        self._log_fn = log_fn or _log.warning
        self._lock = threading.Lock()
        self.last_status: str | None = None
        self.last_change_epoch: float | None = None
        self.transitions = 0

    def check(self, snapshot: dict, db_probe: dict, now: float | None = None,
              stale_min: int | None = None) -> dict:
        now = time.time() if now is None else now
        body = evaluate(snapshot, db_probe, now=now, stale_min=stale_min)
        with self._lock:
            prev, cur = self.last_status, body["status"]
            changed = prev != cur
            if changed:
                self.last_status = cur
                self.last_change_epoch = now
                if prev is not None:
                    self.transitions += 1
        if changed:
            if cur == "degraded":
                self._log_fn(f"[{self.name}.health] {prev or 'start'} -> DEGRADED: "
                             + "; ".join(body["reasons"]))
            elif prev is not None:
                self._log_fn(f"[{self.name}.health] {prev} -> ok (recovered)")
            else:
                _log.info("[%s.health] start -> ok", self.name)
        body["since"] = _iso(self.last_change_epoch)
        return body


def _loguru_warning(msg: str) -> None:
    """BunkerVision logs through loguru (stderr → journal); route the transition
    WARNING there so it sits next to the ais_client lines. Falls back to stdlib."""
    try:
        from loguru import logger
        logger.warning(msg)
    except Exception:  # pragma: no cover
        _log.warning(msg)


# process-wide gate used by the Flask /healthz route
gate = HealthGate("bunkervision", log_fn=_loguru_warning)
