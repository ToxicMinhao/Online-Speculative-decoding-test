#!/bin/bash
set -euo pipefail
source /data/sources/minhao/EAGLE_SPEC/setup/a800/a800_env.sh
source /opt/conda/etc/profile.d/conda.sh
conda activate /data/conda_envs/minhao/dflash_specforge_20261022
source /data/sources/minhao/EAGLE_SPEC/setup/a800/a800_env.sh
export CUDA_VISIBLE_DEVICES=''
export CC=/data/conda_envs/minhao/dflash_tree_qualification_v2/bin/x86_64-conda-linux-gnu-gcc
export CXX=/data/conda_envs/minhao/dflash_tree_qualification_v2/bin/x86_64-conda-linux-gnu-g++
export PYTHONPATH=/data/sources/minhao/frontier_preservation_source_20261125_009/DFlash-Online:/data/sources/minhao/frontier_preservation_source_20261125_009/DFlash-Online/training:/data/sources/minhao/frontier_preservation_source_20261125_009/DFlash-Online/inference
export PYTHONDONTWRITEBYTECODE=1
cd /data/sources/minhao/frontier_preservation_source_20261125_009/DFlash-Online
FRONTIER_RUN=/data/sources/minhao/runs/dflash_frontier_preservation_20261125/production_001
trap 'frontier_exit_code=$?; printf "%s\n" "$frontier_exit_code" > "$FRONTIER_RUN/PROCESS_EXIT_CODE"' EXIT
printf '%s\n' "$$" > "$FRONTIER_RUN/PROCESS_PID"
/data/conda_envs/minhao/dflash_specforge_20261022/bin/python -u -m experiments.frontier_preservation_pilot.allocation_campaign --plan "$FRONTIER_RUN/sealed/PLAN.json"
