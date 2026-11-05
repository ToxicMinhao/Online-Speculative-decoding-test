#!/usr/bin/env bash
set -eu
source /data/sources/minhao/EAGLE_SPEC/setup/a800/a800_env.sh
cd /data/sources/minhao/DFlash-Online
export PYTHONPATH=/data/sources/minhao/DFlash-Online:/data/sources/minhao/DFlash-Online/training:/data/sources/minhao/DFlash-Online/inference
export PYTHONDONTWRITEBYTECODE=1
export CUDA_VISIBLE_DEVICES=1
export PATH=/data/conda_envs/minhao/dflash_specforge_20261022/bin:$PATH
export CONDA_PREFIX=/data/conda_envs/minhao/dflash_specforge_20261022
export CC=/data/conda_envs/minhao/dflash_tree_qualification_v2/bin/x86_64-conda-linux-gnu-gcc
export CXX=/data/conda_envs/minhao/dflash_tree_qualification_v2/bin/x86_64-conda-linux-gnu-g++
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
/data/conda_envs/minhao/dflash_specforge_20261022/bin/python -u -m online_feedback.qualify_sglang \
 --target /data/sources/minhao/models/qwen3_dflash_foundation_20261022/target \
 --draft /data/sources/minhao/models/qwen3_dflash_foundation_20261022/draft \
 --config /data/sources/minhao/DFlash-Online/configs/osd.yaml \
 --worker-python /data/conda_envs/minhao/adaflash_author/bin/python \
 --serving-gpu 0 --output "$1" "${@:2}"
