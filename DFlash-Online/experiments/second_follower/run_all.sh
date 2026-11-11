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
export CC=/data/conda_envs/minhao/dflash_tree_qualification_v2/bin/x86_64-conda-linux-gnu-gcc
export CXX=/data/conda_envs/minhao/dflash_tree_qualification_v2/bin/x86_64-conda-linux-gnu-g++
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ] && [ "$CUDA_VISIBLE_DEVICES" != 0 ]; then
  echo 'Existing GPU allocation differs from sealed physical GPU 0' >&2
  exit 1
fi
export CUDA_VISIBLE_DEVICES=0
printf '%s\n' "$$" > "$TASK_ROOT/SUPERVISOR_PID"
date -u +%FT%TZ > "$TASK_ROOT/STARTED_AT"
"$TASK_PYTHON" -u -m experiments.second_follower.runtime "$TASK_ROOT" > "$TASK_ROOT/RUN.log" 2>&1
TASK_EXIT=$?
printf '%s\n' "$TASK_EXIT" > "$TASK_ROOT/EXIT_CODE"
date -u +%FT%TZ > "$TASK_ROOT/FINISHED_AT"
exit "$TASK_EXIT"
