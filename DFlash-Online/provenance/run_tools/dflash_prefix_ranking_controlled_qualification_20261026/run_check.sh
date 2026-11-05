#!/usr/bin/env bash
set -eu
source /data/sources/minhao/EAGLE_SPEC/setup/a800/a800_env.sh
cd /data/sources/minhao/DFlash-Online
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ] && [ "$CUDA_VISIBLE_DEVICES" != 1 ]; then
    echo 'Inherited allocation differs; preserve it and stop.' >&2
    exit 1
fi
if [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; then
    echo 'Active GPU work; stop before qualification.' >&2
    exit 1
fi
export PYTHONPATH=/data/sources/minhao/DFlash-Online:/data/sources/minhao/DFlash-Online/training:/data/sources/minhao/DFlash-Online/inference
export PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=1
export PATH=/data/conda_envs/minhao/dflash_specforge_20261022/bin:$PATH
export CONDA_PREFIX=/data/conda_envs/minhao/dflash_specforge_20261022
export CC=/data/conda_envs/minhao/dflash_tree_qualification_v2/bin/x86_64-conda-linux-gnu-gcc
export CXX=/data/conda_envs/minhao/dflash_tree_qualification_v2/bin/x86_64-conda-linux-gnu-g++
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
check_root=/data/sources/minhao/runs/dflash_prefix_ranking_controlled_qualification_20261026
ordinary_root=/data/sources/minhao/runs/dflash_osd_one_event_20261025
prior_diagnostic=/data/sources/minhao/runs/dflash_prefix_ranking_gpu_diagnostic_20261026
python "$check_root/preflight.py" "$check_root"
CUDA_VISIBLE_DEVICES='' PYTHONPATH="$PYTHONPATH:/data/sources/minhao/DFlash-Online/tests:$prior_diagnostic" python -m unittest discover -s "$check_root" -p 'test_*.py' -v > "$check_root/CPU_TESTS.txt" 2>&1
python -u "$check_root/runner.py" \
    --config "$ordinary_root/sealed/config.yaml" \
    --ordinary-receipt "$ordinary_root/campaign_001/trainer/events/event_001/RESULT.json" \
    --pinned-initial "$prior_diagnostic/trials/O2/initial.pt" \
    --diagnostic-runner "$prior_diagnostic/runner.py" \
    --output "$check_root/qualification"
