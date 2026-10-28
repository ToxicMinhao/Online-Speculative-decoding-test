#!/usr/bin/env bash
set -u
continuation_root=/data/sources/minhao/runs/dflash_ok_continuation_20261027
date -u +%FT%TZ > "$continuation_root/STARTED.utc"
timeout --signal=TERM --kill-after=120s 10800 bash "$continuation_root/run_campaign.sh" > "$continuation_root/launch.log" 2>&1
confirmation_status=$?
printf '%s\n' "$confirmation_status" > "$continuation_root/EXIT_CODE"
date -u +%FT%TZ > "$continuation_root/FINISHED.utc"
exit "$confirmation_status"
