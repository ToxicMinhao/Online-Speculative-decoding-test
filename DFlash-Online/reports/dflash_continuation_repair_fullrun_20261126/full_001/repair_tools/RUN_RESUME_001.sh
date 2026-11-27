#!/bin/bash
set -euo pipefail
source /data/sources/minhao/EAGLE_SPEC/setup/a800/a800_env.sh
source /opt/conda/etc/profile.d/conda.sh
conda activate /data/conda_envs/minhao/dflash_specforge_20261022
source /data/sources/minhao/EAGLE_SPEC/setup/a800/a800_env.sh
export CUDA_VISIBLE_DEVICES=''
export CC=/data/conda_envs/minhao/dflash_tree_qualification_v2/bin/x86_64-conda-linux-gnu-gcc
export CXX=/data/conda_envs/minhao/dflash_tree_qualification_v2/bin/x86_64-conda-linux-gnu-g++
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
REPAIR_RUN=/data/sources/minhao/runs/dflash_continuation_repair_fullrun_20261126/full_001
REPAIR_SOURCE=$REPAIR_RUN/infrastructure_sources/repair_001/DFlash-Online
export PYTHONPATH=$REPAIR_SOURCE:$REPAIR_SOURCE/training:$REPAIR_SOURCE/inference:$REPAIR_SOURCE/tests
cd "$REPAIR_SOURCE"
test ! -e "$REPAIR_RUN/PROCESS_EXIT_CODE.before_resume_001"
mv "$REPAIR_RUN/PROCESS_EXIT_CODE" "$REPAIR_RUN/PROCESS_EXIT_CODE.before_resume_001"
trap 'repair_exit=$?; printf "%s\n" "$repair_exit" > "$REPAIR_RUN/PROCESS_EXIT_CODE"' EXIT
printf '%s\n' "$$" > "$REPAIR_RUN/PROCESS_PID"
python -u -m experiments.continuation_repair_fullrun.launch \
 --plan "$REPAIR_RUN/sealed/PLAN.json" --resume \
 --source-amendment "$REPAIR_RUN/sealed/SOURCE_AMENDMENT_001.json"
python -u - <<'PY_ANALYSIS'
import json
from pathlib import Path
from experiments.continuation_repair_fullrun.analysis import run_analysis
root=Path('/data/sources/minhao/runs/dflash_continuation_repair_fullrun_20261126/full_001')
result=run_analysis(json.loads((root/'sealed/PLAN.json').read_text()),root/'public')
print(json.dumps(result['decision']))
PY_ANALYSIS
