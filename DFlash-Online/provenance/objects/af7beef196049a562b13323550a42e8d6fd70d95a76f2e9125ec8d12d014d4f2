#!/usr/bin/env bash
set -eu
source /data/sources/minhao/EAGLE_SPEC/setup/a800/a800_env.sh
cd /data/sources/minhao/DFlash-Online
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ] && [ "$CUDA_VISIBLE_DEVICES" != 1 ]; then
    echo 'Inherited allocation differs; preserve it and stop.' >&2
    exit 1
fi
if [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; then
    echo 'Active GPU work; stop before diagnostic.' >&2
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
diagnostic_root=/data/sources/minhao/runs/dflash_prefix_ranking_gpu_diagnostic_20261026
ordinary_root=/data/sources/minhao/runs/dflash_osd_one_event_20261025
python "$diagnostic_root/preflight.py" "$diagnostic_root"
PYTHONPATH="$PYTHONPATH:/data/sources/minhao/DFlash-Online/tests" python -m unittest discover -s "$diagnostic_root" -p test_runner.py -v > "$diagnostic_root/CPU_TESTS.txt" 2>&1
python -u "$diagnostic_root/runner.py" \
    --config "$ordinary_root/sealed/config.yaml" \
    --ordinary-receipt "$ordinary_root/campaign_001/trainer/events/event_001/RESULT.json" \
    --output "$diagnostic_root/trials"
python -u "$diagnostic_root/evaluate.py" \
    --config "$ordinary_root/sealed/config.yaml" --manifest "$ordinary_root/sealed/MANIFEST.json" \
    --trial-root "$diagnostic_root/trials" --output "$diagnostic_root/evaluation"
