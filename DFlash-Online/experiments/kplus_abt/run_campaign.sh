#!/usr/bin/env bash
# Production only: native qualification must already pass in the same seal.
set -eu
source /data/sources/minhao/EAGLE_SPEC/setup/a800/a800_env.sh
run_root="$1"
run_source="$2"
cd "$run_source"
export PYTHONPATH="$PWD:$PWD/training:$PWD/inference:$PWD/tests"
export PYTHONDONTWRITEBYTECODE=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export CONDA_PREFIX=/data/conda_envs/minhao/dflash_specforge_20261022
export PATH="$CONDA_PREFIX/bin:$PATH"
export CC=/data/conda_envs/minhao/dflash_tree_qualification_v2/bin/x86_64-conda-linux-gnu-gcc
export CXX=/data/conda_envs/minhao/dflash_tree_qualification_v2/bin/x86_64-conda-linux-gnu-g++
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
run_python="$CONDA_PREFIX/bin/python"
test -f "$run_root/sealed/SEAL.json"
test -f "$run_root/qualification/ADMISSION.json"
test ! -e "$run_root/PILOT_START_UTC"
date -u +%FT%TZ > "$run_root/PILOT_START_UTC"
echo "$$" > "$run_root/PILOT_SUPERVISOR_PID"
finish() {
    result=$?
    "$run_python" -u -m experiments.kplus_abt.preflight "$run_root" --scope pilot --postflight \
        > "$run_root/pilot_postflight.log" 2>&1 || { if test "$result" -eq 0; then result=1; fi; }
    date -u +%FT%TZ > "$run_root/PILOT_END_UTC"
    echo "$result" > "$run_root/PILOT_EXIT_CODE"
    exit "$result"
}
trap finish EXIT
CUDA_VISIBLE_DEVICES='' "$run_python" -m unittest discover -s tests/kplus_abt -v > "$run_root/PILOT_CPU_TESTS.log" 2>&1
"$run_python" -u -m experiments.kplus_abt.preflight "$run_root" --scope pilot
timeout --signal=TERM --kill-after=120s 24h \
    "$run_python" -u -m experiments.kplus_abt.native_pilot "$run_root"
