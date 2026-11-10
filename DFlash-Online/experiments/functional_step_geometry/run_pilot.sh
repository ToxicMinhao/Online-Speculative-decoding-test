#!/usr/bin/env bash
# Original conditional R1 events-five/six pilot. No qualification or retry here.
set -eu
source /data/sources/minhao/EAGLE_SPEC/setup/a800/a800_env.sh
cd /data/sources/minhao/fsg_source_20261109_pilot_r1/DFlash-Online
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
test -f "$run_root/sealed/SEAL.json"
test ! -e "$run_root/START_UTC"
date -u +%FT%TZ > "$run_root/START_UTC"
echo "$$" > "$run_root/SUPERVISOR_PID"
finish() {
    result=$?
    "$run_python" -u -m experiments.functional_step_geometry.preflight "$run_root" --postflight \
        > "$run_root/postflight.log" 2>&1 || { if test "$result" -eq 0; then result=1; fi; }
    date -u +%FT%TZ > "$run_root/END_UTC"
    echo "$result" > "$run_root/EXIT_CODE"
    exit "$result"
}
trap finish EXIT
CUDA_VISIBLE_DEVICES='' "$run_python" -m unittest -v \
    test_fsg_pilot_seal test_fsg_canonical_bindings test_fsg_native_pilot test_fsg_pilot \
    test_fsg_pilot_diagnostics test_fsg_feasible_realization test_fsg_feasible_runtime \
    test_fsg_continuation test_kplus_objective > "$run_root/CPU_TESTS.log" 2>&1
"$run_python" -u -m experiments.functional_step_geometry.preflight "$run_root"
# The controller independently requires the full 188.444 GB actual-capacity
# admission before collection. Training and serving processes do not overlap.
timeout --signal=TERM --kill-after=120s 48h \
    "$run_python" -u -m experiments.functional_step_geometry.native_pilot "$run_root"
