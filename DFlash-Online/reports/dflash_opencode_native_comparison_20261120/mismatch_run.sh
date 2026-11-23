#!/bin/bash
set -euo pipefail
source /data/sources/minhao/EAGLE_SPEC/setup/a800/a800_env.sh
export CUDA_VISIBLE_DEVICES=''
export CC=/data/conda_envs/minhao/dflash_tree_qualification_v2/bin/x86_64-conda-linux-gnu-gcc
export CXX=/data/conda_envs/minhao/dflash_tree_qualification_v2/bin/x86_64-conda-linux-gnu-g++
export PYTHONDONTWRITEBYTECODE=1
exec /data/conda_envs/minhao/dflash_specforge_20261022/bin/python -u /data/sources/minhao/runs/dflash_opencode_native_comparison_20261120/diagnostic_001/RUN.py
