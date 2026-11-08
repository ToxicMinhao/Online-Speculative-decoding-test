#!/usr/bin/env bash
# Qualification only: this entry point cannot launch events 5–6 or endpoint timing.
set -eu
source /data/sources/minhao/EAGLE_SPEC/setup/a800/a800_env.sh
cd /data/sources/minhao/fsg_source_20261108/DFlash-Online
export PYTHONPATH="$PWD:$PWD/training:$PWD/inference:$PWD/tests"
export PYTHONDONTWRITEBYTECODE=1
export CUDA_VISIBLE_DEVICES=1
export PATH=/data/conda_envs/minhao/dflash_specforge_20261022/bin:$PATH
export CONDA_PREFIX=/data/conda_envs/minhao/dflash_specforge_20261022
export CC=/data/conda_envs/minhao/dflash_tree_qualification_v2/bin/x86_64-conda-linux-gnu-gcc
export CXX=/data/conda_envs/minhao/dflash_tree_qualification_v2/bin/x86_64-conda-linux-gnu-g++
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
run_root="$1"
run_python=/data/conda_envs/minhao/dflash_specforge_20261022/bin/python
test -f "$run_root/sealed/SEAL.json"
test ! -e "$run_root/START_UTC"
date -u +%FT%TZ > "$run_root/START_UTC"
echo "$$" > "$run_root/SUPERVISOR_PID"
trap 'result=$?; date -u +%FT%TZ > "$run_root/END_UTC"; echo "$result" > "$run_root/EXIT_CODE"' EXIT
CUDA_VISIBLE_DEVICES='' "$run_python" -m unittest -v \
  test_functional_geometry_math test_functional_geometry_runtime test_functional_geometry_oracles \
  test_fsg_continuation test_fsg_work test_fsg_qualification test_fsg_seal \
  test_kplus_objective test_role_loss_objective \
  > "$run_root/CPU_TESTS.log" 2>&1
"$run_python" -u -m experiments.functional_step_geometry.preflight "$run_root"
timeout --signal=TERM --kill-after=120s 30m \
  "$run_python" -u -m experiments.functional_step_geometry.prepare_feedback "$run_root"
timeout --signal=TERM --kill-after=120s 2h \
  "$run_python" -u -m experiments.functional_step_geometry.qualify "$run_root"
"$run_python" -u -m experiments.functional_step_geometry.preflight "$run_root" --postflight
