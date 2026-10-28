#!/usr/bin/env bash
set -eu
source /data/sources/minhao/EAGLE_SPEC/setup/a800/a800_env.sh
cd /data/sources/minhao/DFlash-Online
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ] && [ "$CUDA_VISIBLE_DEVICES" != "1" ]; then
    echo 'Inherited allocation differs; preserve it and stop.' >&2
    exit 1
fi
if [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; then
    echo 'GPU jobs active; do not overlap this isolated experiment.' >&2
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
ranking_root=/data/sources/minhao/runs/dflash_prefix_ranking_execution_20261026
ordinary_root=/data/sources/minhao/runs/dflash_osd_one_event_20261025
/data/conda_envs/minhao/dflash_specforge_20261022/bin/python "$ranking_root/preflight.py" "$ranking_root"
CUDA_VISIBLE_DEVICES='' PYTHONPATH="$PYTHONPATH:/data/sources/minhao/DFlash-Online/tests" python -m unittest test_prefix_ranking_controlled_startup test_prefix_ranking_evaluation test_prefix_ranking_run -v > "$ranking_root/CPU_TESTS.txt" 2>&1
timeout --signal=TERM --kill-after=120s 10800 \
 /data/conda_envs/minhao/dflash_specforge_20261022/bin/python -u -m experiments.prefix_ranking.run \
 --config "$ordinary_root/sealed/config.yaml" --manifest "$ordinary_root/sealed/MANIFEST.json" \
 --ordinary-receipt "$ordinary_root/campaign_001/trainer/events/event_001/RESULT.json" \
 --ordinary-measurements "$ordinary_root/campaign_001/evaluation/MEASUREMENTS.jsonl" \
 --worker-python /data/conda_envs/minhao/adaflash_author/bin/python --serving-gpu 0 \
 --qualification-receipt /data/sources/minhao/runs/dflash_prefix_ranking_controlled_qualification_20261026/qualification/RESULT.json \
 --qualification-seal /data/sources/minhao/runs/dflash_prefix_ranking_controlled_qualification_20261026/SEAL.json \
 --qualification-runner /data/sources/minhao/runs/dflash_prefix_ranking_controlled_qualification_20261026/runner.py \
 --diagnostic-runner /data/sources/minhao/runs/dflash_prefix_ranking_gpu_diagnostic_20261026/runner.py \
 --pinned-initial /data/sources/minhao/runs/dflash_prefix_ranking_gpu_diagnostic_20261026/trials/O2/initial.pt \
 --output "$ranking_root/campaign_001"
