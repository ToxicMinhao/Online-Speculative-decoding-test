#!/usr/bin/env bash
set -eu
source /data/sources/minhao/EAGLE_SPEC/setup/a800/a800_env.sh
cd /data/sources/minhao/headroom_source_20261112_001/DFlash-Online
export PYTHONPATH="$PWD:$PWD/training:$PWD/inference:$PWD/tests"
export PYTHONDONTWRITEBYTECODE=1
export CONDA_PREFIX=/data/conda_envs/minhao/adaflash_author
export PATH="$CONDA_PREFIX/bin:$PATH"
export CC=/data/conda_envs/minhao/dflash_tree_qualification_v2/bin/x86_64-conda-linux-gnu-gcc
export CXX=/data/conda_envs/minhao/dflash_tree_qualification_v2/bin/x86_64-conda-linux-gnu-g++
export CUDAHOSTCXX="$CXX" NVCC_CCBIN="$CXX"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ] && [ "$CUDA_VISIBLE_DEVICES" != 0 ]; then
  echo 'Existing GPU allocation conflicts with sealed GPU 0'; exit 2
fi
export CUDA_VISIBLE_DEVICES=0
run_root=/data/sources/minhao/runs/dflash_longer_run_headroom_20261112
set +e
"$CONDA_PREFIX/bin/python" -u -m experiments.longer_run_headroom.run --inputs "$run_root/prepared/PLAN.json" --output "$run_root/diagnostic_001"
run_status=$?
printf '%s\n' "$run_status" > "$run_root/EXIT_CODE"
exit "$run_status"
