#!/usr/bin/env bash
# Use the same compiler/runtime startup as the qualified K+ experiments.
set -eu
source /data/sources/minhao/EAGLE_SPEC/setup/a800/a800_env.sh
cd /data/sources/minhao/DFlash-Online
export PYTHONPATH=/data/sources/minhao/DFlash-Online
export CUDA_VISIBLE_DEVICES=0 PYTHONDONTWRITEBYTECODE=1
export PATH=/data/conda_envs/minhao/dflash_specforge_20261022/bin:$PATH
export CONDA_PREFIX=/data/conda_envs/minhao/dflash_specforge_20261022
export CC=/data/conda_envs/minhao/dflash_tree_qualification_v2/bin/x86_64-conda-linux-gnu-gcc
export CXX=/data/conda_envs/minhao/dflash_tree_qualification_v2/bin/x86_64-conda-linux-gnu-g++
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
test -x "$CC"
test -x "$CXX"
plan="$1"
output="$2"
receipt="$3"
test ! -e "$output"
test ! -e "$receipt"
mkdir -p "$receipt"
date -u +%FT%TZ > "$receipt/STARTED_UTC"
echo "$$" > "$receipt/SUPERVISOR_PID"
trap 'result=$?; date -u +%FT%TZ > "$receipt/FINISHED_UTC"; echo "$result" > "$receipt/EXIT_CODE"' EXIT
# Operational ceiling only; no efficacy stopping rule or automatic retries.
timeout --signal=TERM --kill-after=120s 1h \
  /data/conda_envs/minhao/dflash_specforge_20261022/bin/python -u \
  -m experiments.block_depth.qualify --plan "$plan" --output "$output" \
  > "$receipt/qualification.log" 2>&1
