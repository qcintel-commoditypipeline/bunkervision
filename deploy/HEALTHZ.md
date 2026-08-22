# `/healthz` — liveness gate, not an "is the process up" ping

Background (2026-08-19 → 08-22): aisstream.io went silent (provider outage,
aisstream/issues #279/#282) and BunkerVision sat for days `active` and answering
HTTP 200 while receiving zero AIS messages. A process that is up but not doing
its job must not report healthy. `/healthz` now has to **earn** its 200
(`liveness.py`, `ais_client.ingest_snapshot()`, `db.ping()`).

| verdict | HTTP | when |
|---|---|---|
| `{"ok": true, "status": "ok"}` | 200 | ingest thread alive, an AIS message inside the window, zero DB write errors in the window, `SELECT 1` works |
| `{"ok": false, "status": "degraded", "reasons": [...]}` | **503** | any of: `ingest_thread_missing` (no key / thread died), `no_ais_message_for_Nm`, `write_errors_last_Nm=k`, `db_error: …` |

The body always carries `last_message_at`, `silence_seconds`, `received_last_Nm`,
`write_errors_last_Nm`, `db_ok`, `db_error`; `/healthz?verbose=1` adds the
cumulative ingest counters (`received`, `write_errors`, `connects`, `started_at`,
live vessel count, SSE queue depth), the thread table and the gate's transition
count.

- **Threshold is a setting**: `BUNKERVISION_INGEST_STALE_MINUTES` (default `30`)
  in `/opt/scripts/bunkervision/.env` — silence threshold *and* counter window.
  The feed is beta / no-SLA; tune it rather than hard-coding.
- Silence is measured from the later of *last message* and *ingest-thread start*,
  so a restart is `ok` for the first N minutes; it is **not** reset on reconnects,
  so a client that reconnects forever and receives nothing goes degraded like a
  dead socket.
- DB write errors that used to vanish into `logger.debug("… write skipped")` are now
  counted (all five per-message writes) and logged at WARNING once a minute.
- aisstream `{"error": …}` frames (bad key, second client on the key, throttle) are
  now logged at WARNING — they are exactly what explains a silent feed.
- Each ok→degraded and degraded→ok transition logs **one WARNING** to the journal
  (`[bunkervision.health] ok -> DEGRADED: …` / `degraded -> ok (recovered)`), not
  one per probe.

## Operator reaction to a 503 — it is a signal, not a restart trigger

`bunkervision.service` has `Restart=always` (fires on *process exit* only), no
`WatchdogSec`, no `ExecStartPost` probe: a 503 never makes systemd restart or
loop the service — deliberately, because a provider outage is not fixed by
restarting and reconnect storms hurt the per-key throttle. Nothing on the box
probes this endpoint automatically (nginx `/bunkervision/` only proxies;
`deploy_hardening.sh` curls `/api/demand`, not healthz). Read the `reasons`:

| reason | do |
|---|---|
| `no_ais_message_for_…` | aisstream status / key revoked / a second client on the same key (one connection per key — do **not** point WAKE at this key) — `journalctl -u bunkervision \| grep -i "aisstream"` shows connects and error frames; nothing to restart |
| `write_errors_…` / `db_error: …` | the DuckDB file is unusable (invalidated, locked, disk full): `systemctl restart bunkervision` re-opens it; the journal has the first `DB write failed` line |
| `ingest_thread_missing` | `AISSTREAM_API_KEY` empty in `.env`, or the thread crashed (traceback in the journal) |

One-line checks on the box:

```bash
curl -s -o /dev/null -w '%{http_code}\n' 127.0.0.1:5100/healthz
curl -s '127.0.0.1:5100/healthz?verbose=1' | python3 -m json.tool
```

Tests: `python -m pytest -q tests` (fake clock, no network, in-memory DuckDB).
