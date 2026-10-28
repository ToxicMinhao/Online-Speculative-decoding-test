#!/usr/bin/env bash
set -u
kc_root=/data/sources/minhao/runs/dflash_krc_event_20261027
date -u +%FT%TZ > "$kc_root/STARTED.utc"
timeout --signal=TERM --kill-after=120s 10800 bash "$kc_root/run_campaign.sh" > "$kc_root/launch.log" 2>&1
kc_status=$?
printf '%s\n' "$kc_status" > "$kc_root/EXIT_CODE"
date -u +%FT%TZ > "$kc_root/FINISHED.utc"
exit "$kc_status"
