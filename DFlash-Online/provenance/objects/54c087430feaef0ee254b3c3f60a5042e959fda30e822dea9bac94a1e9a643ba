#!/usr/bin/env bash
set -uo pipefail
source /data/sources/minhao/EAGLE_SPEC/setup/a800/a800_env.sh || exit 1
ops=/data/sources/minhao/runs/dflash_online_gpu_check_20261022
run=/data/sources/minhao/runs/dflash_online_foundation_20261022/refactor_qualification_001
date -u +%FT%TZ > "$ops/STARTED_UTC"
TIMEFORMAT='wall_s=%R user_s=%U system_s=%S'
{ time timeout --signal=INT --kill-after=30s 1800 \
    bash /data/sources/minhao/DFlash-Online/run_qualification.sh "$run" \
    > "$ops/QUALIFICATION.log" 2>&1; } 2> "$ops/RESOURCE_USAGE.txt"
result=$?
printf '%s\n' "$result" > "$ops/EXIT_CODE"
date -u +%FT%TZ > "$ops/FINISHED_UTC"
exit "$result"
