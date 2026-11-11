#!/usr/bin/env bash
set -u
source /data/sources/minhao/EAGLE_SPEC/setup/a800/a800_env.sh || exit 1
TASK_ROOT=$1
TASK_CODE=$2
TASK_PYTHON=/data/conda_envs/minhao/dflash_specforge_20261022/bin/python
cd "$TASK_CODE" || exit 1
export PYTHONPATH="$TASK_CODE"
export CUDA_VISIBLE_DEVICES=0
date -u +%FT%TZ > "$TASK_ROOT/STARTED_AT"
printf '%s\n' "$$" > "$TASK_ROOT/SUPERVISOR_PID"
timeout --signal=TERM --kill-after=60s 4h "$TASK_PYTHON" -u -m experiments.survival_confirmation.runtime "$TASK_ROOT" > "$TASK_ROOT/RUN.log" 2>&1
TASK_EXIT=$?
printf '%s\n' "$TASK_EXIT" > "$TASK_ROOT/EXECUTION_EXIT_CODE"
"$TASK_PYTHON" -m experiments.survival_confirmation.report "$TASK_ROOT" --output "$TASK_ROOT/final_report" > "$TASK_ROOT/REPORT.log" 2>&1
TASK_REPORT_EXIT=$?
printf '%s\n' "$TASK_REPORT_EXIT" > "$TASK_ROOT/REPORT_EXIT_CODE"
if [ "$TASK_EXIT" -eq 0 ] && [ "$TASK_REPORT_EXIT" -ne 0 ]; then TASK_EXIT=$TASK_REPORT_EXIT; fi
printf '%s\n' "$TASK_EXIT" > "$TASK_ROOT/EXIT_CODE"
date -u +%FT%TZ > "$TASK_ROOT/FINISHED_AT"
exit "$TASK_EXIT"
