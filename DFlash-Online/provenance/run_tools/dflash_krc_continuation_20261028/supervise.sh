#!/usr/bin/env bash
set -u
krc2_root=/data/sources/minhao/runs/dflash_krc_continuation_20261028
date -u +%FT%TZ > "$krc2_root/STARTED.utc"
timeout --signal=TERM --kill-after=120s 7200 bash "$krc2_root/run_campaign.sh" > "$krc2_root/launch.log" 2>&1
confirmation_status=$?
printf '%s\n' "$confirmation_status" > "$krc2_root/EXIT_CODE"
date -u +%FT%TZ > "$krc2_root/FINISHED.utc"
exit "$confirmation_status"
