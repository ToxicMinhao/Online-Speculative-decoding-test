#!/bin/bash
set -euo pipefail
source /data/sources/minhao/EAGLE_SPEC/setup/a800/a800_env.sh
export CUDA_VISIBLE_DEVICES=''
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
REPAIR_ROOT=/data/sources/minhao/runs/dflash_continuation_repair_fullrun_20261126/full_001
REPAIR_SOURCE=$REPAIR_ROOT/infrastructure_sources/repair_001/DFlash-Online
REPAIR_PYTHON=/data/conda_envs/minhao/dflash_specforge_20261022/bin/python
export PYTHONPATH=$REPAIR_SOURCE:$REPAIR_SOURCE/training:$REPAIR_SOURCE/inference
export PATH=/data/conda_envs/minhao/dflash_specforge_20261022/bin:$PATH
cd "$REPAIR_SOURCE"
"$REPAIR_PYTHON" -m unittest discover -s tests -p 'test_continuation_repair_fullrun*.py' -v
"$REPAIR_PYTHON" reports/dflash_continuation_repair_fullrun_20261126/full_001/repair_tools/recover_packaging.py
"$REPAIR_PYTHON" -m experiments.continuation_repair_fullrun.source_amendment \
 --plan "$REPAIR_ROOT/sealed/PLAN.json" \
 --source-root "$REPAIR_SOURCE" \
 --source-manifest "$REPAIR_ROOT/infrastructure_sources/repair_001/SOURCE.json" \
 --resume-evidence "$REPAIR_ROOT/infrastructure_repairs/packaging_001/CPU_RECOVERY.json" \
 --output "$REPAIR_ROOT/sealed/SOURCE_AMENDMENT_001.json"
