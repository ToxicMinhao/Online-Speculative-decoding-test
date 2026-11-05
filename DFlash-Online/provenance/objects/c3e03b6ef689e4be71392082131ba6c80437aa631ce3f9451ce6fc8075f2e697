#!/usr/bin/env bash
set -u
check_root=/data/sources/minhao/runs/dflash_prefix_ranking_execution_20261026
date -u +%FT%TZ > "$check_root/STARTED.utc"
timeout --signal=TERM --kill-after=120s 10800 bash "$check_root/run_campaign.sh" > "$check_root/launch.log" 2>&1
status=$?
printf '%s\n' "$status" > "$check_root/EXIT_CODE"
date -u +%FT%TZ > "$check_root/FINISHED.utc"
exit "$status"
