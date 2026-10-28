#!/usr/bin/env bash
set -u
krc34_root=/data/sources/minhao/runs/dflash_krc_events_3_4_20261028
date -u +%FT%TZ > "$krc34_root/STARTED.utc"
timeout --signal=TERM --kill-after=120s 10800 bash "$krc34_root/run_campaign.sh" > "$krc34_root/launch.log" 2>&1
confirmation_status=$?
printf '%s\n' "$confirmation_status" > "$krc34_root/EXIT_CODE"
date -u +%FT%TZ > "$krc34_root/FINISHED.utc"
exit "$confirmation_status"
