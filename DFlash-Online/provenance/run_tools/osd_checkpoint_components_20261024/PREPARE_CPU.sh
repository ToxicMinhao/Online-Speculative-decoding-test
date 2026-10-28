#!/usr/bin/env bash
set -eu
source /data/sources/minhao/EAGLE_SPEC/setup/a800/a800_env.sh
cd /data/sources/minhao/DFlash-Online
export CUDA_VISIBLE_DEVICES=''
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=/data/sources/minhao/DFlash-Online:/data/sources/minhao/DFlash-Online/training:/data/sources/minhao/DFlash-Online/inference
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
/data/conda_envs/minhao/dflash_specforge_20261022/bin/python - <<'PY'
import hashlib,json
from pathlib import Path
root=Path('/data/sources/minhao/runs/osd_checkpoint_components_20261024')
manifest=json.loads((root/'inputs/IMPLEMENTATION_SHA256.json').read_text())
checks={p:hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in manifest.items()}
(root/'TRANSFER_CHECK.json').write_text(json.dumps(checks,indent=2)+'\n')
assert all(checks.values()), checks
print('All',len(checks),'transferred files match.')
PY
/data/conda_envs/minhao/dflash_specforge_20261022/bin/python -m unittest discover -s tests -v > /data/sources/minhao/runs/osd_checkpoint_components_20261024/ALL_CPU_TESTS.log 2>&1
tail -5 /data/sources/minhao/runs/osd_checkpoint_components_20261024/ALL_CPU_TESTS.log
/data/conda_envs/minhao/dflash_specforge_20261022/bin/python -u -m experiments.osd_components.prepare \
  --prior-root /data/sources/minhao/runs/dflash_online_osd_rerun_20261023 \
  --runtime-pin /data/sources/minhao/runs/osd_checkpoint_components_20261024/inputs/RUNTIME_PIN.json \
  --model-pin /data/sources/minhao/runs/osd_checkpoint_components_20261024/inputs/MODEL_PIN.json \
  --worker-python /data/conda_envs/minhao/adaflash_author/bin/python \
  --output /data/sources/minhao/runs/osd_checkpoint_components_20261024/prepared
