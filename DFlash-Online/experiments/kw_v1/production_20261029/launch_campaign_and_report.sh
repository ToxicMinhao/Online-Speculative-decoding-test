#!/usr/bin/env bash
set -u
source /data/sources/minhao/EAGLE_SPEC/setup/a800/a800_env.sh || exit 1
kw_run_root=/data/sources/minhao/runs/dflash_kw_v1_20261029/production_001
kw_source=/data/sources/minhao/DFlash-Online
# The approved supervisor runs once, with an eight-hour bound and no retry.
bash "$kw_source/experiments/kw_v1/supervise.sh" "$kw_run_root" 28800
kw_execution_status=$?
printf '%s\n' "$kw_execution_status" > "$kw_run_root/CAMPAIGN_EXIT_CODE"
# Reporting reads retained records only, after the campaign has stopped.
export CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$kw_source:$kw_source/training:$kw_source/inference:$kw_source/tests"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
/data/conda_envs/minhao/dflash_specforge_20261022/bin/python -m experiments.kw_v1.report \
    --root "$kw_run_root" --output "$kw_run_root/report" \
    --execution-commit 708e974e06f5d9f4b186c32607c16b37cbd4b14f \
    > "$kw_run_root/REPORT_GENERATION.log" 2>&1
kw_report_status=$?
printf '%s\n' "$kw_report_status" > "$kw_run_root/REPORT_EXIT_CODE"
date -u +%FT%TZ > "$kw_run_root/REPORT_FINISHED.utc"
if [ "$kw_execution_status" -ne 0 ]; then exit "$kw_execution_status"; fi
exit "$kw_report_status"
