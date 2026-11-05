#!/usr/bin/env bash
set -u
kw_root=${1:?An explicit sealed run root is required}
kw_seconds=${2:-28800}
# Execution bound is explicit and never extends itself on a failure.
if [ -e "$kw_root/STARTED.utc" ]; then
    echo 'Supervisor already ran; preserve its exit and use explicit reviewed boundary recovery.' >&2
    exit 1
fi
date -u +%FT%TZ > "$kw_root/STARTED.utc"
timeout --signal=TERM --kill-after=120s "$kw_seconds" bash \
    /data/sources/minhao/DFlash-Online/experiments/kw_v1/run_a800.sh \
    "$kw_root" > "$kw_root/launch.log" 2>&1
kw_status=$?
printf '%s\n' "$kw_status" > "$kw_root/EXIT_CODE"
date -u +%FT%TZ > "$kw_root/FINISHED.utc"
exit "$kw_status"
