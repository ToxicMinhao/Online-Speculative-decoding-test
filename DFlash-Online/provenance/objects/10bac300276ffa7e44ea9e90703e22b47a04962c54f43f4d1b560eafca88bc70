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
test "$(cat "$diagnostic_root/EXIT_CODE")" = 0
python - "$diagnostic_root" <<'CHECK'
import hashlib,json,shutil,sys
from pathlib import Path
root=Path(sys.argv[1]); seal=json.loads((root/'BACKWARD_SEAL.json').read_text())
for name,expected in seal.items():
 assert hashlib.sha256((root/name).read_bytes()).hexdigest()==expected,name
assert shutil.disk_usage(root).free > 22*2**30
CHECK
PYTHONPATH="$PYTHONPATH:/data/sources/minhao/DFlash-Online/tests" python -m unittest discover -s "$diagnostic_root" -p test_backward_control.py -v > "$diagnostic_root/CPU_BACKWARD_A800.txt" 2>&1
python -u "$diagnostic_root/backward_control.py" --config "$ordinary_root/sealed/config.yaml" --ordinary-receipt "$ordinary_root/campaign_001/trainer/events/event_001/RESULT.json" --output "$diagnostic_root/backward_control"
