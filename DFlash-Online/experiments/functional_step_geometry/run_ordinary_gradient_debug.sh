#!/usr/bin/env bash
# Separately authorized zero-Adam startup/control diagnosis; no pilot continuation.
set -eu
source /data/sources/minhao/EAGLE_SPEC/setup/a800/a800_env.sh
cd /data/sources/minhao/fsg_ordinary_debug_20261109_002/DFlash-Online
export PYTHONPATH="$PWD:$PWD/training:$PWD/inference:$PWD/tests"
export PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=1
export CONDA_PREFIX=/data/conda_envs/minhao/dflash_specforge_20261022
export PATH="$CONDA_PREFIX/bin:$PATH"
export CC=/data/conda_envs/minhao/dflash_tree_qualification_v2/bin/x86_64-conda-linux-gnu-gcc
export CXX=/data/conda_envs/minhao/dflash_tree_qualification_v2/bin/x86_64-conda-linux-gnu-g++
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
run_root="$1"
run_python="$CONDA_PREFIX/bin/python"
test -f "$run_root/PLAN.json"
test ! -e "$run_root/START_UTC"
date -u +%FT%TZ > "$run_root/START_UTC"
echo "$$" > "$run_root/SUPERVISOR_PID"
finish() {
    result=$?
    date -u +%FT%TZ > "$run_root/END_UTC"
    echo "$result" > "$run_root/EXIT_CODE"
    exit "$result"
}
trap finish EXIT
CUDA_VISIBLE_DEVICES='' "$run_python" -m unittest -v \
    test_fsg_ordinary_gradient_checks test_fsg_ordinary_startup \
    test_fsg_qualification test_fsg_revised_qualification \
    > "$run_root/CPU_TESTS.log" 2>&1
timeout --signal=TERM --kill-after=120s 90m \
    "$run_python" -u -m experiments.functional_step_geometry.diagnose_ordinary_gradients "$run_root/PLAN.json"
