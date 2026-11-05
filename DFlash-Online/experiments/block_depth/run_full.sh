#!/usr/bin/env bash
set -eu
source /data/sources/minhao/EAGLE_SPEC/setup/a800/a800_env.sh
cd /data/sources/minhao/DFlash-Online
export PYTHONPATH=/data/sources/minhao/DFlash-Online:/data/sources/minhao/DFlash-Online/training:/data/sources/minhao/DFlash-Online/inference:/data/sources/minhao/DFlash-Online/tests
export PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=''
export PATH=/data/conda_envs/minhao/dflash_specforge_20261022/bin:$PATH
export CONDA_PREFIX=/data/conda_envs/minhao/dflash_specforge_20261022
export CC=/data/conda_envs/minhao/dflash_tree_qualification_v2/bin/x86_64-conda-linux-gnu-gcc
export CXX=/data/conda_envs/minhao/dflash_tree_qualification_v2/bin/x86_64-conda-linux-gnu-g++
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
run_root="$1"
run_python=/data/conda_envs/minhao/dflash_specforge_20261022/bin/python
test -f "$run_root/PLAN.json"
test ! -e "$run_root/START_UTC"
date -u +%FT%TZ > "$run_root/START_UTC"
echo "$$" > "$run_root/SUPERVISOR_PID"
trap 'result=$?; date -u +%FT%TZ > "$run_root/END_UTC"; echo "$result" > "$run_root/EXIT_CODE"' EXIT
"$run_python" -m unittest discover -s tests -p 'test_block_depth*.py' -q > "$run_root/CPU_TESTS.log" 2>&1
set +e
timeout --signal=TERM --kill-after=120s 12h "$run_python" -u -m experiments.block_depth.full_campaign "$run_root/PLAN.json"
campaign_exit=$?
echo "$campaign_exit" > "$run_root/CAMPAIGN_EXIT_CODE"
"$run_python" -c 'import json,sys; from pathlib import Path; from experiments.block_depth.full_plan import preflight; p=Path(sys.argv[1]); preflight(json.loads((p/"PLAN.json").read_text()),p/"POSTFLIGHT.json",postflight=True)' "$run_root"
postflight_exit=$?
"$run_python" -m experiments.block_depth.full_report "$run_root"
report_exit=$?
echo "$report_exit" > "$run_root/REPORT_EXIT_CODE"
if [ "$campaign_exit" -ne 0 ]; then exit "$campaign_exit"; fi
if [ "$postflight_exit" -ne 0 ]; then exit "$postflight_exit"; fi
exit "$report_exit"
