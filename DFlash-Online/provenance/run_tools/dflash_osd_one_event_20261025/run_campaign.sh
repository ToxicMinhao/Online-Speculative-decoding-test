#!/usr/bin/env bash
set -eu
source /data/sources/minhao/EAGLE_SPEC/setup/a800/a800_env.sh
cd /data/sources/minhao/DFlash-Online
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ] && [ "$CUDA_VISIBLE_DEVICES" != "1" ]; then
    echo "Inherited GPU allocation differs; preserve it and review before launch." >&2
    exit 1
fi
if [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; then
    echo "GPU jobs are active; this isolated run requires both assigned GPUs free." >&2
    exit 1
fi
export PYTHONPATH=/data/sources/minhao/DFlash-Online:/data/sources/minhao/DFlash-Online/training:/data/sources/minhao/DFlash-Online/inference
export PYTHONDONTWRITEBYTECODE=1
export CUDA_VISIBLE_DEVICES=1
export PATH=/data/conda_envs/minhao/dflash_specforge_20261022/bin:$PATH
export CONDA_PREFIX=/data/conda_envs/minhao/dflash_specforge_20261022
export CC=/data/conda_envs/minhao/dflash_tree_qualification_v2/bin/x86_64-conda-linux-gnu-gcc
export CXX=/data/conda_envs/minhao/dflash_tree_qualification_v2/bin/x86_64-conda-linux-gnu-g++
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
one_event_root=/data/sources/minhao/runs/dflash_osd_one_event_20261025
/data/conda_envs/minhao/dflash_specforge_20261022/bin/python "$one_event_root/preflight.py" "$one_event_root"
timeout --signal=TERM --kill-after=120s 7200 \
 /data/conda_envs/minhao/dflash_specforge_20261022/bin/python -u -m experiments.osd_one_event.run \
 --target /data/sources/minhao/models/qwen3_dflash_foundation_20261022/target \
 --draft /data/sources/minhao/models/qwen3_dflash_foundation_20261022/draft \
 --worker-python /data/conda_envs/minhao/adaflash_author/bin/python \
 --serving-gpu 0 --manifest "$one_event_root/sealed/MANIFEST.json" \
 --config "$one_event_root/sealed/config.yaml" --output "$one_event_root/campaign_001"
