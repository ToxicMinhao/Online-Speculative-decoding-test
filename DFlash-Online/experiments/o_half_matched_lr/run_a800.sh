#!/usr/bin/env bash
set -eu
source /data/sources/minhao/EAGLE_SPEC/setup/a800/a800_env.sh
cd /data/sources/minhao/DFlash-Online
oh_root=${1:?Explicit sealed experiment root required}
oh_resume=${2:-}
if [ -n "$oh_resume" ] && [ "$oh_resume" != '--resume' ]; then exit 2; fi
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
if [ -z "$oh_resume" ]; then
    CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m unittest discover -s tests -p 'test_o_half_*.py' -v > "$oh_root/CPU_REMOTE.log" 2>&1
    CUDA_VISIBLE_DEVICES='' python -m experiments.o_half_matched_lr.preflight "$oh_root" > "$oh_root/PREFLIGHT.log" 2>&1
fi
set +e
python -u -m experiments.o_half_matched_lr.campaign "$oh_root" ${oh_resume:+--resume}
oh_status=$?
CUDA_VISIBLE_DEVICES='' python -m experiments.o_half_matched_lr.report "$oh_root" > "$oh_root/REPORT.log" 2>&1
oh_report=$?
CUDA_VISIBLE_DEVICES='' python -m experiments.o_half_matched_lr.preflight "$oh_root" --postflight > "$oh_root/POSTFLIGHT.log" 2>&1
oh_postflight=$?
if [ "$oh_status" -ne 0 ]; then exit "$oh_status"; fi
if [ "$oh_report" -ne 0 ]; then exit "$oh_report"; fi
exit "$oh_postflight"
