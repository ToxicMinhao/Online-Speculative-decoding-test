#!/usr/bin/env bash
set -eu
source /data/sources/minhao/EAGLE_SPEC/setup/a800/a800_env.sh
cd /data/sources/minhao/DFlash-Online
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ] && [ "$CUDA_VISIBLE_DEVICES" != '1' ]; then
    echo 'Inherited allocation differs; preserve it and stop.' >&2
    exit 1
fi
if [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; then
    echo 'GPU jobs active; isolated serving must not overlap them.' >&2
    exit 1
fi
export CUDA_VISIBLE_DEVICES=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=/data/sources/minhao/DFlash-Online:/data/sources/minhao/DFlash-Online/training:/data/sources/minhao/DFlash-Online/inference
export PATH=/data/conda_envs/minhao/dflash_specforge_20261022/bin:$PATH
export CONDA_PREFIX=/data/conda_envs/minhao/dflash_specforge_20261022
export CC=/data/conda_envs/minhao/dflash_tree_qualification_v2/bin/x86_64-conda-linux-gnu-gcc
export CXX=/data/conda_envs/minhao/dflash_tree_qualification_v2/bin/x86_64-conda-linux-gnu-g++
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
continuation_root=/data/sources/minhao/runs/dflash_ok_continuation_20261027
CUDA_VISIBLE_DEVICES='' python "$continuation_root/preflight.py" "$continuation_root"
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  PYTHONPATH="$PYTHONPATH:/data/sources/minhao/DFlash-Online/tests" \
  python -m unittest experiments.ok_continuation.test_evaluate experiments.ok_continuation.test_resume -v \
  > "$continuation_root/CPU_TESTS_REMOTE.log" 2>&1
(cd "$continuation_root" && CUDA_VISIBLE_DEVICES='' python -m unittest test_campaign -v \
  >> "$continuation_root/CPU_TESTS_REMOTE.log" 2>&1)
python -u "$continuation_root/campaign.py" "$continuation_root"

CUDA_VISIBLE_DEVICES='' python "$continuation_root/preflight.py" "$continuation_root" --postflight
