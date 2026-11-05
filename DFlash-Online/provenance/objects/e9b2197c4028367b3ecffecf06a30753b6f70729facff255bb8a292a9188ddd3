#!/usr/bin/env bash
set -u
kr_root=/data/sources/minhao/runs/dflash_reference_replay_20261028
date -u +%FT%TZ > "$kr_root/STARTED.utc"
timeout --signal=TERM --kill-after=120s 7200 bash "$kr_root/run_campaign.sh" > "$kr_root/launch.log" 2>&1
confirmation_status=$?
printf '%s\n' "$confirmation_status" > "$kr_root/EXIT_CODE"
date -u +%FT%TZ > "$kr_root/FINISHED.utc"
exit "$confirmation_status"
