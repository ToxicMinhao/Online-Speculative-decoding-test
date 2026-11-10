#!/usr/bin/env bash
# Continue only the remaining qualification under the reviewed numerical policy.
set -eu
source /data/sources/minhao/EAGLE_SPEC/setup/a800/a800_env.sh
cd /data/sources/minhao/fsg_source_20261109_revised/DFlash-Online
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
  test_functional_geometry_math test_functional_geometry_runtime test_functional_geometry_oracles \
  test_fsg_continuation test_fsg_work test_fsg_qualification test_fsg_seal test_fsg_repair_seal \
  test_fsg_same_layout_oracle test_fsg_qualification_checks test_fsg_revised_qualification \
  test_fsg_consequence_diagnostic test_fsg_bf16_consequences_runner test_fsg_pilot \
  test_kplus_objective test_role_loss_objective > "$run_root/CPU_TESTS.log" 2>&1
"$run_python" -u -m experiments.functional_step_geometry.preflight "$run_root"
timeout --signal=TERM --kill-after=120s 2h \
    "$run_python" -u -m experiments.functional_step_geometry.resume_qualification "$run_root"
