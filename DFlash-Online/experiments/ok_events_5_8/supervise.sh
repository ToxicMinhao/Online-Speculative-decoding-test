#!/usr/bin/env bash
set -u
ok58_root=${1:?An explicit sealed run root is required}
date -u +%FT%TZ > "$ok58_root/STARTED.utc"
timeout --signal=TERM --kill-after=120s 18000 bash \
    /data/sources/minhao/DFlash-Online/experiments/ok_events_5_8/run_a800.sh \
    "$ok58_root" > "$ok58_root/launch.log" 2>&1
ok58_status=$?
printf '%s\n' "$ok58_status" > "$ok58_root/EXIT_CODE"
date -u +%FT%TZ > "$ok58_root/FINISHED.utc"
exit "$ok58_status"
