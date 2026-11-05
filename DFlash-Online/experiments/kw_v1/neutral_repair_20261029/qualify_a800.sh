#!/usr/bin/env bash
set -eu
source /data/sources/minhao/EAGLE_SPEC/setup/a800/a800_env.sh
cd /data/sources/minhao/DFlash-Online
kw_root=/data/sources/minhao/runs/dflash_kw_v1_20261029/neutral_repair_001
if [ -e "$kw_root/STARTED.utc" ]; then exit 1; fi
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ] && [ "$CUDA_VISIBLE_DEVICES" != '1' ]; then exit 1; fi
if [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; then exit 1; fi
export CUDA_VISIBLE_DEVICES=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=/data/sources/minhao/DFlash-Online:/data/sources/minhao/DFlash-Online/training:/data/sources/minhao/DFlash-Online/inference:/data/sources/minhao/DFlash-Online/tests
export PATH=/data/conda_envs/minhao/dflash_specforge_20261022/bin:$PATH
export CONDA_PREFIX=/data/conda_envs/minhao/dflash_specforge_20261022
export CC=/data/conda_envs/minhao/dflash_tree_qualification_v2/bin/x86_64-conda-linux-gnu-gcc
export CXX=/data/conda_envs/minhao/dflash_tree_qualification_v2/bin/x86_64-conda-linux-gnu-g++
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export TORCH_LOGS=recompiles
date -u +%FT%TZ > "$kw_root/STARTED.utc"
set +e
CUDA_VISIBLE_DEVICES='' python -u "$kw_root/repair_preflight.py" "$kw_root" > "$kw_root/PREFLIGHT.log" 2>&1
kw_status=$?
if [ "$kw_status" -eq 0 ]; then
    CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m unittest experiments.kw_v1.test_diagnostics test_response_weighting test_kw_training -q > "$kw_root/CPU_REMOTE.log" 2>&1
    kw_status=$?
fi
if [ "$kw_status" -eq 0 ]; then
    timeout --signal=TERM --kill-after=120s 3600 python -u -m experiments.kw_v1.qualify --plan "$kw_root/PLAN.json" --root "$kw_root" > "$kw_root/QUALIFICATION.log" 2>&1
    kw_status=$?
fi
printf '%s\n' "$kw_status" > "$kw_root/EXIT_CODE"
date -u +%FT%TZ > "$kw_root/FINISHED.utc"
exit "$kw_status"
