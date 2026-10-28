#!/usr/bin/env bash
set -u
diagnostic_root=/data/sources/minhao/runs/dflash_prefix_ranking_gpu_diagnostic_20261026
date -u +%FT%TZ > "$diagnostic_root/STARTED.utc"
timeout --signal=TERM --kill-after=120s 1800 bash "$diagnostic_root/run_diagnostic.sh" > "$diagnostic_root/launch.log" 2>&1
status=$?
printf '%s\n' "$status" > "$diagnostic_root/EXIT_CODE"
date -u +%FT%TZ > "$diagnostic_root/FINISHED.utc"
exit "$status"
