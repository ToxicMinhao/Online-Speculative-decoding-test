#!/usr/bin/env bash
set -u
source /data/sources/minhao/EAGLE_SPEC/setup/a800/a800_env.sh || exit 1
TASK_ROOT=$1
TASK_CODE=$2
TASK_PYTHON=/data/conda_envs/minhao/dflash_specforge_20261022/bin/python
cd "$TASK_CODE" || exit 1
export PYTHONPATH="$TASK_CODE"
export PYTHONDONTWRITEBYTECODE=1
export CONDA_PREFIX=/data/conda_envs/minhao/dflash_specforge_20261022
export PATH="$CONDA_PREFIX/bin:$PATH"
# Match the completed pilot's compiler and CPU-thread environment. SGLang's
# startup can link cached CUDA objects, even though this run performs no training.
export CC=/data/conda_envs/minhao/dflash_tree_qualification_v2/bin/x86_64-conda-linux-gnu-gcc
export CXX=/data/conda_envs/minhao/dflash_tree_qualification_v2/bin/x86_64-conda-linux-gnu-g++
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
test -x "$CC" && test -x "$CXX" || exit 1
export CUDA_VISIBLE_DEVICES=0
date -u +%FT%TZ > "$TASK_ROOT/STARTED_AT"
printf '%s\n' "$$" > "$TASK_ROOT/SUPERVISOR_PID"
timeout --signal=TERM --kill-after=60s 4h "$TASK_PYTHON" -u -m experiments.successor_shuffle.runtime "$TASK_ROOT" > "$TASK_ROOT/RUN.log" 2>&1
TASK_EXIT=$?
printf '%s\n' "$TASK_EXIT" > "$TASK_ROOT/EXECUTION_EXIT_CODE"
"$TASK_PYTHON" -m experiments.successor_shuffle.report "$TASK_ROOT" --output "$TASK_ROOT/final_report" > "$TASK_ROOT/REPORT.log" 2>&1
TASK_REPORT_EXIT=$?
printf '%s\n' "$TASK_REPORT_EXIT" > "$TASK_ROOT/REPORT_EXIT_CODE"
if [ "$TASK_EXIT" -eq 0 ] && [ "$TASK_REPORT_EXIT" -ne 0 ]; then TASK_EXIT=$TASK_REPORT_EXIT; fi
printf '%s\n' "$TASK_EXIT" > "$TASK_ROOT/EXIT_CODE"
date -u +%FT%TZ > "$TASK_ROOT/FINISHED_AT"
exit "$TASK_EXIT"
