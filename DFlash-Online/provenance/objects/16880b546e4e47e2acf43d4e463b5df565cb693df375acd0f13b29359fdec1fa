#!/usr/bin/env bash
set -uo pipefail
source /data/sources/minhao/EAGLE_SPEC/setup/a800/a800_env.sh || exit 1
export PYTHONPATH=/data/sources/minhao/DFlash-Online:/data/sources/minhao/DFlash-Online/training:/data/sources/minhao/DFlash-Online/inference
export PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=0
export CC=/data/conda_envs/minhao/dflash_tree_qualification_v2/bin/x86_64-conda-linux-gnu-gcc
export CXX=/data/conda_envs/minhao/dflash_tree_qualification_v2/bin/x86_64-conda-linux-gnu-g++
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
code=/data/sources/minhao/EAGLE_SPEC/research/projected_target_feasibility_20261022
ops=/data/sources/minhao/runs/projected_target_feasibility_20261022
cd /data/sources/minhao/DFlash-Online || exit 1
date -u +%FT%TZ > "$ops/STARTED_UTC"
TIMEFORMAT='wall_s=%R user_s=%U system_s=%S'
# Safety ceiling only; no retry or scientific early stopping based on results.
{ time timeout --signal=INT --kill-after=60s 7200 \
    /data/conda_envs/minhao/dflash_specforge_20261022/bin/python -u "$code/runner.py" \
    --manifest "$code/manifest.json" --config "$code/execution_config.yaml" \
    --target /data/sources/minhao/models/qwen3_dflash_foundation_20261022/target \
    --draft /data/sources/minhao/models/qwen3_dflash_foundation_20261022/draft \
    --output "$ops/campaign_001" > "$ops/CAMPAIGN.log" 2>&1; } 2> "$ops/RESOURCE_USAGE.txt"
result=$?
printf '%s\n' "$result" > "$ops/EXIT_CODE"
date -u +%FT%TZ > "$ops/FINISHED_UTC"
exit "$result"
