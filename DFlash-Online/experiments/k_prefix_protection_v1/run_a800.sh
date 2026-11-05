#!/usr/bin/env bash
set -eu
source /data/sources/minhao/EAGLE_SPEC/setup/a800/a800_env.sh
cd /data/sources/minhao/DFlash-Online
kp_root=${1:?Explicit sealed experiment root required}
kp_resume=${2:-}
if [ -n "$kp_resume" ] && [ "$kp_resume" != '--resume' ]; then exit 2; fi
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ] && [ "$CUDA_VISIBLE_DEVICES" != '1' ]; then
    echo 'Inherited GPU allocation differs; preserve it and stop.' >&2
    exit 1
fi
if [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; then
    echo 'Active GPU jobs prevent isolated execution.' >&2
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
if [ -z "$kp_resume" ]; then
    CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m unittest discover -s tests -p 'test_k_prefix_protection_*.py' -v > "$kp_root/CPU_REMOTE.log" 2>&1
    CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m unittest discover -s tests -p 'test_prefix_protection_objective.py' -v >> "$kp_root/CPU_REMOTE.log" 2>&1
    CUDA_VISIBLE_DEVICES='' python -m experiments.k_prefix_protection_v1.preflight "$kp_root" > "$kp_root/PREFLIGHT.log" 2>&1
    # One sole qualification suite. Any failure exits before production; no retry.
    python -u -m experiments.k_prefix_protection_v1.launch_qualification "$kp_root"
fi
set +e
python -u -m experiments.k_prefix_protection_v1.campaign "$kp_root" ${kp_resume:+--resume}
kp_status=$?
CUDA_VISIBLE_DEVICES='' python -m experiments.k_prefix_protection_v1.report "$kp_root" > "$kp_root/REPORT.log" 2>&1
kp_report=$?
CUDA_VISIBLE_DEVICES='' python -m experiments.k_prefix_protection_v1.preflight "$kp_root" --postflight > "$kp_root/POSTFLIGHT.log" 2>&1
kp_postflight=$?
if [ "$kp_status" -ne 0 ]; then exit "$kp_status"; fi
if [ "$kp_report" -ne 0 ]; then exit "$kp_report"; fi
exit "$kp_postflight"
