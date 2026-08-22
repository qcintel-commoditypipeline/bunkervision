#!/usr/bin/env bash
# Deploy the calibrated-nowcast salvage to the BunkerVision droplet.
#
# Ships the runtime files, installs the new deps (statsmodels/scipy), and restarts.
# USE_CALIBRATED_NOWCAST ships as True: the live headline becomes the calibrated
# nowcast, the dashboard track record shows the nowcast's out-of-sample record, the
# /backtest shadow panel is live, and the AIS signal test starts logging.
# Reversible: the current versions are backed up first (and the flag can be set
# False to revert behaviour without redeploying).
#
# Run from a machine with SSH access to the droplet, from the repo root:
#   bash deploy/deploy_hardening.sh
#
# Ships whatever is checked out locally (no `git checkout` inside this script) —
# always run it from an up-to-date `main` checkout. `main` is the branch that
# ships; see README.md#branch-convention. (Historical note: this script and its
# name predate that convention — it was written against a branch called
# `hardening`, since merged into `main`.)
set -euo pipefail

SERVER="${SERVER:-root@165.232.110.29}"
WORKDIR="${WORKDIR:-/opt/scripts/bunkervision}"
VENV="${VENV:-/opt/scripts/venv}"
TS="$(date +%Y%m%d_%H%M%S)"
# Disable SSH connection multiplexing (ControlMaster from ~/.ssh/config fails in
# this environment); each scp/ssh opens its own direct connection.
SSH_OPTS="-o ControlMaster=no -o ControlPath=none"

# Includes the /healthz liveness gate (liveness.py + the counters in ais_client.py and
# db.ping(); see deploy/HEALTHZ.md). .env and the systemd unit are NOT shipped:
# server secrets stay on the box, and the unit change in git is comment-only.
FILES="app.py config.py demand_model.py nowcast_model.py backtest_nowcast.py \
ais_signal.py monthly_study.py set_official.py requirements.txt \
ais_client.py db.py liveness.py \
templates/backtest.html templates/dashboard.html"

echo "==> Bundling runtime files"
tar czf "/tmp/bv_hardening_${TS}.tgz" $FILES

echo "==> Copying to ${SERVER}"
scp $SSH_OPTS "/tmp/bv_hardening_${TS}.tgz" "${SERVER}:/tmp/"

echo "==> Deploying on ${SERVER} (backup -> extract -> pip install -> restart)"
ssh $SSH_OPTS "${SERVER}" "TS='${TS}' WORKDIR='${WORKDIR}' VENV='${VENV}' FILES='${FILES}' bash -s" <<'REMOTE'
set -euo pipefail
cd "$WORKDIR"
# Back up only the files that already exist (new files are skipped, not fatal).
tar czf "/root/bunkervision_backup_${TS}.tgz" --ignore-failed-read $FILES || true
tar xzf "/tmp/bv_hardening_${TS}.tgz" -C "$WORKDIR"
"$VENV/bin/pip" install -q -r "$WORKDIR/requirements.txt"
systemctl restart bunkervision.service
sleep 3
echo -n "service: "; systemctl is-active bunkervision.service
curl -fsS "127.0.0.1:5100/api/demand?port=singapore" | grep -q calibrated_nowcast \
  && echo "live /api/demand OK — headline = calibrated nowcast (flag ON)" \
  || echo "WARN: /api/demand did not report calibrated_nowcast — check the flag"
# /healthz is a liveness GATE (deploy/HEALTHZ.md): 200 inside the startup grace window,
# 503 "degraded" once the AIS feed has been silent > BUNKERVISION_INGEST_STALE_MINUTES.
# Either code means the route is live; only a connection failure is a deploy problem.
echo -n "healthz: "; curl -s -o /dev/null -w '%{http_code}\n' 127.0.0.1:5100/healthz || echo "UNREACHABLE"
echo "backup at /root/bunkervision_backup_${TS}.tgz"
REMOTE

echo "==> Done. Shadow panel: https://<host>/backtest  (the slow first load builds the backtest)"
echo "    AIS signal test now runnable on the server; ais_signal_log fills on the monthly job."
