#!/usr/bin/env bash
set -eu
source /data/sources/minhao/EAGLE_SPEC/setup/a800/a800_env.sh
cd /data/sources/minhao/DFlash-Online
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ] && [ "$CUDA_VISIBLE_DEVICES" != '1' ]; then
    echo 'Inherited GPU allocation differs; stop.' >&2
    exit 1
fi
if [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; then
    echo 'GPU jobs active; isolated training/evaluation required.' >&2
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
krc_off_root=/data/sources/minhao/runs/dflash_krc_off_20261028
CUDA_VISIBLE_DEVICES='' python "$krc_off_root/preflight.py" "$krc_off_root"
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m unittest test_krc_off experiments.krc_off.test_evaluate experiments.krc_off.test_campaign -v > "$krc_off_root/CPU_REMOTE.log" 2>&1
set +e
python -u -m experiments.krc_off.campaign "$krc_off_root"
campaign_status=$?
CUDA_VISIBLE_DEVICES='' python "$krc_off_root/preflight.py" "$krc_off_root" --postflight > "$krc_off_root/POSTFLIGHT.log" 2>&1
postflight_status=$?
if [ "$campaign_status" -ne 0 ]; then exit "$campaign_status"; fi
exit "$postflight_status"
