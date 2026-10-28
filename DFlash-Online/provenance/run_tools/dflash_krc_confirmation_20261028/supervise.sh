#!/usr/bin/env bash
set -u
confirmation_root=/data/sources/minhao/runs/dflash_krc_confirmation_20261028
date -u +%FT%TZ > "$confirmation_root/STARTED.utc"
timeout --signal=TERM --kill-after=120s 7200 bash "$confirmation_root/run_campaign.sh" > "$confirmation_root/launch.log" 2>&1
confirmation_status=$?
printf '%s\n' "$confirmation_status" > "$confirmation_root/EXIT_CODE"
date -u +%FT%TZ > "$confirmation_root/FINISHED.utc"
exit "$confirmation_status"
