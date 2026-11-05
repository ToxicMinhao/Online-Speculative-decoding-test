#!/usr/bin/env bash
set -u
krc_off_root=/data/sources/minhao/runs/dflash_krc_off_20261028
date -u +%FT%TZ > "$krc_off_root/STARTED.utc"
timeout --signal=TERM --kill-after=120s 7200 bash "$krc_off_root/run_campaign.sh" > "$krc_off_root/launch.log" 2>&1
confirmation_status=$?
printf '%s\n' "$confirmation_status" > "$krc_off_root/EXIT_CODE"
date -u +%FT%TZ > "$krc_off_root/FINISHED.utc"
exit "$confirmation_status"
