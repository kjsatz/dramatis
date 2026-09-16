#!/usr/bin/env bash
# Stream finished runs home from a rented box while it trains: rsync is incremental, so pulling
# every minute copies each run as it lands; a file caught mid-write is simply re-copied next
# pass (its size changes). Run on the laptop in its own terminal; Ctrl-C when the box is done
# and one last pass has completed.
#   BOX=root@<host> PORT=<port> bash scripts/pull_runs.sh
set -eu
cd "$(dirname "$0")/.."
: "${BOX:?set BOX=root@<host>}" "${PORT:?set PORT=<ssh port>}"
EVERY="${EVERY:-60}"
while true; do
  # plain -a: macOS openrsync lacks --info; no -z: checkpoints don't compress. A transient ssh
  # failure just waits for the next pass.
  rsync -a -e "ssh -p $PORT" "$BOX:~/dramatis/runs/" runs/ || echo "--- pull failed; will retry"
  echo "--- $(date '+%H:%M:%S') $(ls -d runs/*_story-10m_*-s*/ 2>/dev/null | wc -l | tr -d ' ') run dirs local, $(ls runs/*_story-10m_*-s*/summary.json 2>/dev/null | wc -l | tr -d ' ') finished; next pull in ${EVERY}s"
  sleep "$EVERY"
done
