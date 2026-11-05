#!/usr/bin/env bash
set -eu
source /data/sources/minhao/EAGLE_SPEC/setup/a800/a800_env.sh
cd /data/sources/minhao/DFlash-Online
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ] && [ "$CUDA_VISIBLE_DEVICES" != '0' ]; then
    echo 'Inherited allocation differs; preserve it and stop.' >&2
    exit 1
fi
if [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; then
    echo 'GPU jobs active; isolated serving must not overlap them.' >&2
    exit 1
fi
export CUDA_VISIBLE_DEVICES=0 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=/data/sources/minhao/DFlash-Online:/data/sources/minhao/DFlash-Online/training:/data/sources/minhao/DFlash-Online/inference
export PATH=/data/conda_envs/minhao/dflash_specforge_20261022/bin:$PATH
export CONDA_PREFIX=/data/conda_envs/minhao/dflash_specforge_20261022
export CC=/data/conda_envs/minhao/dflash_tree_qualification_v2/bin/x86_64-conda-linux-gnu-gcc
export CXX=/data/conda_envs/minhao/dflash_tree_qualification_v2/bin/x86_64-conda-linux-gnu-g++
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
confirmation_root=/data/sources/minhao/runs/dflash_first_error_fresh_confirmation_20261027
python "$confirmation_root/preflight.py" "$confirmation_root"
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  PYTHONPATH="$PYTHONPATH:/data/sources/minhao/DFlash-Online/tests" \
  python -m unittest experiments.fresh_confirmation.test_evaluate -v \
  > "$confirmation_root/CPU_TESTS_REMOTE.log" 2>&1
(cd "$confirmation_root" && CUDA_VISIBLE_DEVICES='' python -m unittest test_campaign -v \
  >> "$confirmation_root/CPU_TESTS_REMOTE.log" 2>&1)
python -u "$confirmation_root/campaign.py" "$confirmation_root"
