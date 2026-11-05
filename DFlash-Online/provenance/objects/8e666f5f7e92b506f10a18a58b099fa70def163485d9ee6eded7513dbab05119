#!/usr/bin/env bash
set -u
first_error_root=/data/sources/minhao/runs/dflash_first_error_execution_20261026
date -u +%FT%TZ > "$first_error_root/STARTED.utc"
timeout --signal=TERM --kill-after=120s 10800 bash "$first_error_root/run_campaign.sh" > "$first_error_root/launch.log" 2>&1
first_error_status=$?
printf '%s\n' "$first_error_status" > "$first_error_root/EXIT_CODE"
date -u +%FT%TZ > "$first_error_root/FINISHED.utc"
exit "$first_error_status"
