#!/usr/bin/env bash
set -u
margin_root=/data/sources/minhao/runs/dflash_reference_margin_execution_20261026
date -u +%FT%TZ > "$margin_root/STARTED.utc"
timeout --signal=TERM --kill-after=120s 10800 bash "$margin_root/run_campaign.sh" > "$margin_root/launch.log" 2>&1
margin_status=$?
printf '%s\n' "$margin_status" > "$margin_root/EXIT_CODE"
date -u +%FT%TZ > "$margin_root/FINISHED.utc"
exit "$margin_status"
