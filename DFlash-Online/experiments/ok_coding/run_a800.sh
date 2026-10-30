#!/usr/bin/env bash
set -eu
source /data/sources/minhao/EAGLE_SPEC/setup/a800/a800_env.sh
cd /data/sources/minhao/DFlash-Online
coding_root=${1:?An explicit sealed run root is required}
coding_resume=${2:-}
if [ -n "$coding_resume" ] && [ "$coding_resume" != '--resume' ]; then
    echo 'The only optional mode is explicit --resume.' >&2
    exit 1
fi
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ] && [ "$CUDA_VISIBLE_DEVICES" != '1' ]; then
    echo 'Inherited GPU allocation differs; preserve it and stop.' >&2
    exit 1
fi
if [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; then
    echo 'GPU jobs are active; isolated training and evaluation are required.' >&2
    exit 1
fi
export CUDA_VISIBLE_DEVICES=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=/data/sources/minhao/DFlash-Online:/data/sources/minhao/DFlash-Online/training:/data/sources/minhao/DFlash-Online/inference:/data/sources/minhao/DFlash-Online/tests
export PATH=/data/conda_envs/minhao/dflash_specforge_20261022/bin:$PATH
export CONDA_PREFIX=/data/conda_envs/minhao/dflash_specforge_20261022
export CC=/data/conda_envs/minhao/dflash_tree_qualification_v2/bin/x86_64-conda-linux-gnu-gcc
export CXX=/data/conda_envs/minhao/dflash_tree_qualification_v2/bin/x86_64-conda-linux-gnu-g++
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
if [ -z "$coding_resume" ]; then
    CUDA_VISIBLE_DEVICES='' python -m experiments.ok_coding.preflight "$coding_root"
fi
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m unittest \
    discover -s tests -p 'test_ok_coding_*.py' -v > "$coding_root/CPU_REMOTE.log" 2>&1
# There is no model warmup/qualification stage. The actual first jobs and included
# serving gates supply native evidence. The campaign never retries failed work.
set +e
if [ -n "$coding_resume" ]; then
    python -u -m experiments.ok_coding.campaign "$coding_root" --resume
else
    python -u -m experiments.ok_coding.campaign "$coding_root"
fi
coding_status=$?
CUDA_VISIBLE_DEVICES='' python -m experiments.ok_coding.preflight "$coding_root" --postflight > "$coding_root/POSTFLIGHT.log" 2>&1
coding_postflight=$?
if [ "$coding_status" -ne 0 ]; then exit "$coding_status"; fi
exit "$coding_postflight"
