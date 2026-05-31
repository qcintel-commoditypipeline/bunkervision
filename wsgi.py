"""Gunicorn entrypoint for BunkerVision.

Run with a SINGLE worker only:

    gunicorn --workers 1 --threads 16 --worker-class gthread \
             --timeout 120 --bind 127.0.0.1:5100 wsgi:app

The AIS websocket listener, the event detector, and the APScheduler jobs all
run as threads *inside this process* (see app.start_background). Multiple
gunicorn workers would each open their own AIS stream and double-write the
single-writer DuckDB file, so concurrency must come from --threads, never
--workers. Do NOT use --preload (threads do not survive the fork).
"""
from app import app, start_background

# Launch AIS/detector/scheduler in this worker process.
start_background()
