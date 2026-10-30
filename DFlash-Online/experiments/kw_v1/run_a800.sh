#!/usr/bin/env bash
set -eu
source /data/sources/minhao/EAGLE_SPEC/setup/a800/a800_env.sh
cd /data/sources/minhao/DFlash-Online
kw_root=${1:?An explicit sealed run root is required}
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ] && [ "$CUDA_VISIBLE_DEVICES" != '1' ]; then
    echo 'Inherited GPU allocation differs; preserve it and stop.' >&2
    exit 1
fi
if [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; then
    echo 'GPU jobs active; isolated qualification/training/evaluation required.' >&2
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
CUDA_VISIBLE_DEVICES='' python -m experiments.kw_v1.preflight "$kw_root"
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m unittest \
    experiments.kw_v1.test_campaign experiments.kw_v1.test_resume experiments.kw_v1.test_evaluate experiments.kw_v1.test_report \
    experiments.ok_events_5_8.test_resume experiments.ok_continuation.test_resume \
    test_response_weighting -v > "$kw_root/CPU_REMOTE.log" 2>&1
# The separately bounded qualification must already have passed. This launcher
# never retries it and cannot silently add model/optimizer qualification work.
set +e
python -u -m experiments.kw_v1.campaign "$kw_root"
kw_status=$?
CUDA_VISIBLE_DEVICES='' python -m experiments.kw_v1.preflight "$kw_root" --postflight > "$kw_root/POSTFLIGHT.log" 2>&1
kw_postflight=$?
if [ "$kw_status" -ne 0 ]; then exit "$kw_status"; fi
exit "$kw_postflight"
