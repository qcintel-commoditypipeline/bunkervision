# BunkerVision

Bottom-up bunker-fuel demand nowcast from AIS + official series calibration.
Production runs on `165.232.110.29` under `bunkervision.service`.

## Branch convention

**`main` is the branch that ships.** Production is deployed straight off `main`
via `deploy/deploy_hardening.sh`, which tars the runtime file list from your
current local checkout and ships it — the script does not check out a branch
by name, so always run it from a `main` checkout that is up to date with
`origin/main`.

Team convention is **direct push to `main`**, not PRs-then-merge: commit
locally, `git push origin main`, then `bash deploy/deploy_hardening.sh` to
ship. `/healthz` (see `deploy/HEALTHZ.md`) is the liveness gate that proves a
deploy actually landed.

### History note (2026-08-22)

Production was carried on a branch named `hardening` for a period while `main`
drifted behind (`main` was missing the ingest-liveness-gate work that
`hardening` had independently re-landed). `hardening` was merged back into
`main` to resolve the drift — `main` now carries everything `hardening` had,
byte-for-byte. `hardening` is retired; do not push new work to it. If it still
exists in `git branch -a`, it is safe to delete once you've confirmed no one
has uncommitted work depending on it.
