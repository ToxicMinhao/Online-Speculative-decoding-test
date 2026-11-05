#!/usr/bin/env bash
set -u
correction_root=/data/sources/minhao/runs/dflash_prefix_correction_step_20261027
date -u +%FT%TZ > "$correction_root/STARTED.utc"
timeout --signal=TERM --kill-after=120s 1800 bash "$correction_root/run_diagnostic.sh" > "$correction_root/launch.log" 2>&1
correction_status=$?
printf '%s\n' "$correction_status" > "$correction_root/EXIT_CODE"
date -u +%FT%TZ > "$correction_root/FINISHED.utc"
exit "$correction_status"
