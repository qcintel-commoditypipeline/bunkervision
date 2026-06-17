#!/usr/bin/env bash
# Deploy the calibrated-nowcast salvage to the BunkerVision droplet.
#
# Ships the runtime files, installs the new deps (statsmodels/scipy), and restarts.
# USE_CALIBRATED_NOWCAST ships as False, so the LIVE headline is UNCHANGED — this
# only makes the read-only /backtest shadow panel live and the AIS signal test
# runnable. Reversible: the current versions are backed up first.
#
# Run from a machine with SSH access to the droplet, from the repo root:
#   bash deploy/deploy_hardening.sh
set -euo pipefail

SERVER="${SERVER:-root@165.232.110.29}"
WORKDIR="${WORKDIR:-/opt/scripts/bunkervision}"
VENV="${VENV:-/opt/scripts/venv}"
TS="$(date +%Y%m%d_%H%M%S)"

FILES="app.py config.py demand_model.py nowcast_model.py backtest_nowcast.py \
ais_signal.py monthly_study.py set_official.py requirements.txt templates/backtest.html"

echo "==> Bundling runtime files"
tar czf "/tmp/bv_hardening_${TS}.tgz" $FILES

echo "==> Copying to ${SERVER}"
scp "/tmp/bv_hardening_${TS}.tgz" "${SERVER}:/tmp/"

echo "==> Deploying on ${SERVER} (backup -> extract -> pip install -> restart)"
ssh "${SERVER}" "TS='${TS}' WORKDIR='${WORKDIR}' VENV='${VENV}' FILES='${FILES}' bash -s" <<'REMOTE'
set -euo pipefail
cd "$WORKDIR"
# Back up only the files that already exist (new files are skipped, not fatal).
tar czf "/root/bunkervision_backup_${TS}.tgz" --ignore-failed-read $FILES || true
tar xzf "/tmp/bv_hardening_${TS}.tgz" -C "$WORKDIR"
"$VENV/bin/pip" install -q -r "$WORKDIR/requirements.txt"
systemctl restart bunkervision.service
sleep 3
echo -n "service: "; systemctl is-active bunkervision.service
curl -fsS "127.0.0.1:5100/api/demand?port=singapore" >/dev/null && echo "live /api/demand OK (headline unchanged, flag off)"
echo "backup at /root/bunkervision_backup_${TS}.tgz"
REMOTE

echo "==> Done. Shadow panel: https://<host>/backtest  (the slow first load builds the backtest)"
echo "    AIS signal test now runnable on the server; ais_signal_log fills on the monthly job."
