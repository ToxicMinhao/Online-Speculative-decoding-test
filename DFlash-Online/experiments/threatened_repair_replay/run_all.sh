#!/usr/bin/env bash
set -u
source /data/sources/minhao/EAGLE_SPEC/setup/a800/a800_env.sh || exit 1
TASK_ROOT=$1
TASK_CODE=$2
TASK_PYTHON=/data/conda_envs/minhao/dflash_specforge_20261022/bin/python
case "${CUDA_VISIBLE_DEVICES-}" in ''|0) ;; *) echo 'Inherited GPU assignment differs from the sealed physical card' >&2; exit 1 ;; esac
cd "$TASK_CODE" || exit 1
export PYTHONPATH="$TASK_CODE:$TASK_CODE/training"
export PYTHONDONTWRITEBYTECODE=1
export CONDA_PREFIX=/data/conda_envs/minhao/dflash_specforge_20261022
export PATH="$CONDA_PREFIX/bin:$PATH"
export CC=/data/conda_envs/minhao/dflash_tree_qualification_v2/bin/x86_64-conda-linux-gnu-gcc
export CXX=/data/conda_envs/minhao/dflash_tree_qualification_v2/bin/x86_64-conda-linux-gnu-g++
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=0
test -x "$CC" && test -x "$CXX" || exit 1
test ! -e "$TASK_ROOT/STARTED_AT" || exit 1
date -u +%FT%TZ > "$TASK_ROOT/STARTED_AT"
printf '%s\n' "$$" > "$TASK_ROOT/SUPERVISOR_PID"
"$TASK_PYTHON" -u -m experiments.threatened_repair_replay.prepare --spec "$TASK_ROOT/SPEC.json" > "$TASK_ROOT/PREPARATION.log" 2>&1
TASK_EXIT=$?
if [ "$TASK_EXIT" -eq 0 ]; then
    timeout --signal=TERM --kill-after=60s 4h "$TASK_PYTHON" -u -m experiments.threatened_repair_replay.campaign --plan "$TASK_ROOT/sealed/PLAN.json" > "$TASK_ROOT/RUN.log" 2>&1
    TASK_EXIT=$?
fi
printf '%s\n' "$TASK_EXIT" > "$TASK_ROOT/EXIT_CODE"
date -u +%FT%TZ > "$TASK_ROOT/FINISHED_AT"
exit "$TASK_EXIT"
