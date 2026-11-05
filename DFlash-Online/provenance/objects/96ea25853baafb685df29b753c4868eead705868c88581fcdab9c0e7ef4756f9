#!/usr/bin/env bash
set -u
diagnostic_root=/data/sources/minhao/runs/dflash_prefix_ranking_gpu_diagnostic_20261026
date -u +%FT%TZ > "$diagnostic_root/BACKWARD_STARTED.utc"
timeout --signal=TERM --kill-after=60s 900 bash "$diagnostic_root/run_backward.sh" > "$diagnostic_root/backward.log" 2>&1
status=$?
printf '%s\n' "$status" > "$diagnostic_root/BACKWARD_EXIT_CODE"
date -u +%FT%TZ > "$diagnostic_root/BACKWARD_FINISHED.utc"
exit "$status"
