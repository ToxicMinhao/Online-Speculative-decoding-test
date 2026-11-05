#!/usr/bin/env bash
set -u
coding_root=${1:?An explicit sealed run root is required}
coding_seconds=${2:?An explicit wall-time bound in seconds is required}
if ! [[ "$coding_seconds" =~ ^[1-9][0-9]*$ ]]; then
    echo 'The wall-time bound must be a positive integer number of seconds.' >&2
    exit 1
fi
if [ -e "$coding_root/STARTED.utc" ]; then
    echo 'Preserve the existing attempt; use explicit verified-boundary recovery.' >&2
    exit 1
fi
date -u +%FT%TZ > "$coding_root/STARTED.utc"
printf '%s\n' "$coding_seconds" > "$coding_root/WALL_TIME_BOUND_SECONDS"
timeout --signal=TERM --kill-after=120s "$coding_seconds" bash \
    /data/sources/minhao/DFlash-Online/experiments/ok_coding/run_a800.sh \
    "$coding_root" > "$coding_root/launch.log" 2>&1
coding_status=$?
printf '%s\n' "$coding_status" > "$coding_root/EXIT_CODE"
date -u +%FT%TZ > "$coding_root/FINISHED.utc"
exit "$coding_status"
